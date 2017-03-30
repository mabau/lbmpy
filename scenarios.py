import numpy as np
from functools import partial
from pystencils.field import getLayoutOfArray, createNumpyArrayWithLayout
from pystencils.slicing import sliceFromDirection, addGhostLayers, getPeriodicBoundaryFunctor, removeGhostLayers, \
    normalizeSlice, makeSlice
from lbmpy.creationfunctions import createLatticeBoltzmannFunction, updateWithDefaultParameters
from lbmpy.macroscopic_value_kernels import compileMacroscopicValuesGetter, compileMacroscopicValuesSetter
from lbmpy.boundaries import BoundaryHandling, noSlip, ubb, fixedDensity
from lbmpy.stencils import getStencil
from lbmpy.updatekernels import createPdfArray


class Scenario(object):
    def __init__(self, domainSize, methodParameters, optimizationParams, lbmKernel=None,
                 initialVelocity=None, preUpdateFunctions=[], kernelParams={}):
        """
        Constructor for generic scenarios. You probably want to use one of the simpler scenario factory functions
        of this file.

        :param domainSize: tuple, defining the domain size without ghost layers
        :param methodParameters: dict with method parameters, as documented in :mod:`lbmpy.creationfunctions`,
                                 passed to :func:`lbmpy.creationfunction.createLatticeBoltzmannFunction`
        :param optimizationParams: dict with optimization parameters, as documented in :mod:`lbmpy.creationfunctions`,
                                   passed to :func:`lbmpy.creationfunction.createLatticeBoltzmannFunction`
        :param lbmKernel: a lattice boltzmann function can be passed here, if None it is created with the parameters
                          specified above
        :param initialVelocity: tuple with initial velocity of the domain, can either be a constant or a numpy array
                                with first axes shaped like the domain, and the last dimension of size #dimensions
        :param preUpdateFunctions: list of functions that are called before the LBM kernel. They get the pdf array as
                                   only argument. Can be used for custom boundary conditions, periodicity, etc.
        :param kernelParams: dict which is passed to the kernel as additional parameters
        """

        ghostLayers = 1
        domainSizeWithGhostLayer = tuple([s + 2 * ghostLayers for s in domainSize])
        D = len(domainSize)
        if 'stencil' not in methodParameters:
            methodParameters['stencil'] = 'D2Q9' if D == 2 else 'D3Q27'

        if isinstance(initialVelocity, np.ndarray):
            initialVelocity = addGhostLayers(initialVelocity, indexDimensions=1, ghostLayers=1)

        methodParameters, optimizationParams = updateWithDefaultParameters(methodParameters, optimizationParams)

        Q = len(getStencil(methodParameters['stencil']))
        self._pdfArrays = [createPdfArray(domainSize, Q, layout=optimizationParams['fieldLayout']),
                           createPdfArray(domainSize, Q, layout=optimizationParams['fieldLayout'])]

        # Create kernel
        if lbmKernel is None:
            optimizationParams['pdfArr'] = self._pdfArrays[0]
            methodParameters['optimizationParams'] = optimizationParams
            self._lbmKernel = createLatticeBoltzmannFunction(**methodParameters)
        else:
            self._lbmKernel = lbmKernel
        assert D == self._lbmKernel.method.dim, "Domain size and stencil do not match"

        # Macroscopic value input/output
        pdfArrLayout = getLayoutOfArray(self._pdfArrays[0])
        pdfArrLayoutNoIdx = getLayoutOfArray(self._pdfArrays[0], indexDimensionIds=[D])
        self._density = createNumpyArrayWithLayout(domainSizeWithGhostLayer, layout=pdfArrLayoutNoIdx)
        self._velocity = createNumpyArrayWithLayout(list(domainSizeWithGhostLayer) + [D], layout=pdfArrLayout)
        self._getMacroscopic = compileMacroscopicValuesGetter(self._lbmKernel.method, ['density', 'velocity'],
                                                              pdfArr=self._pdfArrays[0], target='cpu')

        self._boundaryHandling = BoundaryHandling(self._pdfArrays[0], domainSize, self.method,
                                                  target=optimizationParams['target'])

        self._preUpdateFunctions = preUpdateFunctions
        self.kernelParams = kernelParams
        self._pdfGpuArrays = []

        if initialVelocity is None:
            initialVelocity = [0] * D

        setMacroscopic = compileMacroscopicValuesSetter(self.method, {'density': 1.0, 'velocity': initialVelocity},
                                                        pdfArr=self._pdfArrays[0], target='cpu')
        setMacroscopic(pdfs=self._pdfArrays[0])

        if optimizationParams['target'] == 'gpu':
            import pycuda.gpuarray as gpuarray
            self._pdfGpuArrays = [gpuarray.to_gpu(a) for a in self._pdfArrays]
        else:
            self._pdfGpuArrays = []

    def run(self, timeSteps=1):
        """Run the scenario for the given amount of time steps"""
        if len(self._pdfGpuArrays) > 0:
            self._gpuTimeloop(timeSteps)
        else:
            self._cpuTimeloop(timeSteps)

    @property
    def velocity(self):
        """Velocity as numpy array"""
        mask = np.logical_not(np.bitwise_and(self.boundaryHandling.flagField, self.boundaryHandling._fluidFlag))
        mask = np.repeat(mask[..., np.newaxis], self.dim, axis=2)
        return removeGhostLayers(np.ma.masked_array(self._velocity, mask), indexDimensions=1)

    @property
    def density(self):
        """Density as numpy array"""
        mask = np.logical_not(np.bitwise_and(self.boundaryHandling.flagField, self.boundaryHandling._fluidFlag))
        return removeGhostLayers(np.ma.masked_array(self._density, mask))

    @property
    def pdfs(self):
        """Particle distribution functions as numpy array"""
        return removeGhostLayers(self._pdfArrays[0], indexDimensions=1)

    @property
    def boundaryHandling(self):
        """Boundary handling instance of the scenario. Use this to change the boundary setup"""
        return self._boundaryHandling

    @property
    def method(self):
        """Lattice boltzmann method description"""
        return self._lbmKernel.method

    @property
    def dim(self):
        """Dimension of the domain"""
        return self.method.dim

    @property
    def ast(self):
        """Returns abstract syntax tree of the kernel"""
        return self._lbmKernel.ast

    @property
    def updateRule(self):
        """Equation collection defining the LBM update rule (already in simplified form)"""
        return self._lbmKernel.updateRule

    def animateVelocity(self, steps=10, **kwargs):
        import lbmpy.plot2d as plt

        def runFunction():
            self.run(steps)
            return self.velocity
        return plt.vectorFieldMagnitudeAnimation(runFunction, **kwargs)

    def plotVelocity(self, **kwargs):
        import lbmpy.plot2d as plt
        if self.dim == 2:
            plt.vectorFieldMagnitude(self.velocity, **kwargs)
            plt.title("Velocity Magnitude")
            plt.colorbar()
            plt.axis('equal')
        elif self.dim == 3:
            idxSlice = normalizeSlice(makeSlice[:, :, 0.5], self.velocity.shape[:3])
            plt.vectorFieldMagnitude(self.velocity[idxSlice], **kwargs)
            plt.title("Velocity Magnitude in x-y (z at half of domain)")
            plt.colorbar()
            plt.axis('equal')
        else:
            raise NotImplementedError("Can only plot 2D and 3D scenarios")

    def plotDensity(self, **kwargs):
        import lbmpy.plot2d as plt
        if self.dim == 2:
            plt.scalarField(self.density, **kwargs)
            plt.title("Density")
            plt.colorbar()
            plt.axis('equal')
        elif self.dim == 3:
            idxSlice = normalizeSlice(makeSlice[:, :, 0.5], self.density.shape)
            plt.scalarField(self.density[idxSlice], **kwargs)
            plt.title("Density in x-y (z at half of domain)")
            plt.colorbar()
            plt.axis('equal')
        else:
            raise NotImplementedError("Can only plot 2D and 3D scenarios")

    def _timeloop(self, pdfArrays, timeSteps):
        for t in range(timeSteps):
            for f in self._preUpdateFunctions:
                f(pdfArrays[0])
            if self._boundaryHandling is not None:
                self._boundaryHandling(pdfs=pdfArrays[0])
            self._lbmKernel(src=pdfArrays[0], dst=pdfArrays[1], **self.kernelParams)

            pdfArrays[0], pdfArrays[1] = pdfArrays[1], pdfArrays[0]  # swap

    def _cpuTimeloop(self, timeSteps):
        self._timeloop(self._pdfArrays, timeSteps)
        self._getMacroscopic(pdfs=self._pdfArrays[0], density=self._density, velocity=self._velocity)

    def _gpuTimeloop(self, timeSteps):
        # Transfer data to gpu
        for cpuArr, gpuArr in zip(self._pdfArrays, self._pdfGpuArrays):
            gpuArr.set(cpuArr)

        self._timeloop(self._pdfGpuArrays, timeSteps)

        # Transfer data from gpu to cpu
        for cpuArr, gpuArr in zip(self._pdfArrays, self._pdfGpuArrays):
            gpuArr.get(cpuArr)

        self._getMacroscopic(pdfs=self._pdfArrays[0], density=self._density, velocity=self._velocity)


# ---------------------------------------- Example Scenarios -----------------------------------------------------------


def createFullyPeriodicFlow(initialVelocity, optimizationParams={}, lbmKernel=None, kernelParams={}, **kwargs):
    """
    Creates a fully periodic setup with prescribed velocity field
    """
    domainSize = initialVelocity.shape[:-1]
    scenario = Scenario(domainSize, kwargs, optimizationParams, lbmKernel, initialVelocity, kernelParams)
    scenario.boundaryHandling.setPeriodicity(True, True, True)
    return scenario


def createLidDrivenCavity(domainSize, lidVelocity=0.005, optimizationParams={}, lbmKernel=None,
                          kernelParams={}, **kwargs):
    """
    Creates a lid driven cavity scenario with prescribed lid velocity in lattice coordinates
    """
    scenario = Scenario(domainSize, kwargs, optimizationParams, lbmKernel=lbmKernel, kernelParams=kernelParams)

    myUbb = partial(ubb, velocity=[lidVelocity, 0, 0][:scenario.method.dim])
    myUbb.name = 'ubb'
    dim = scenario.method.dim
    scenario.boundaryHandling.setBoundary(myUbb, sliceFromDirection('N', dim))
    for direction in ('W', 'E', 'S') if scenario.method.dim == 2 else ('W', 'E', 'S', 'T', 'B'):
        scenario.boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, dim))

    return scenario


def createPressureGradientDrivenChannel(dim, pressureDifference, domainSize=None, radius=None, length=None,
                                        lbmKernel=None, optimizationParams={}, initialVelocity=None,
                                        boundarySetupFunctions=[], kernelParams={}, **kwargs):
    """
    Creates a channel flow in x direction, which is driven by two pressure boundaries.
    """
    assert dim in (2, 3)

    if radius is not None:
        assert length is not None
        if dim == 3:
            domainSize = (length, 2 * radius + 1, 2 * radius + 1)
            roundChannel = True
        else:
            if domainSize is None:
                domainSize = (length, 2 * radius)
    else:
        roundChannel = False

    if domainSize is None:
        raise ValueError("Missing domainSize or radius and length parameters!")

    scenario = Scenario(domainSize, kwargs, optimizationParams, lbmKernel=lbmKernel, kernelParams=kernelParams,
                        initialVelocity=initialVelocity)
    boundaryHandling = scenario.boundaryHandling
    pressureBoundaryInflow = partial(fixedDensity, density=1.0 + pressureDifference)
    pressureBoundaryInflow.__name__ = "Inflow"

    pressureBoundaryOutflow = partial(fixedDensity, density=1.0)
    pressureBoundaryOutflow.__name__ = "Outflow"
    boundaryHandling.setBoundary(pressureBoundaryInflow, sliceFromDirection('W', dim))
    boundaryHandling.setBoundary(pressureBoundaryOutflow, sliceFromDirection('E', dim))

    if dim == 2:
        for direction in ('N', 'S'):
            boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, dim))
    elif dim == 3:
        if roundChannel:
            noSlipIdx = boundaryHandling.addBoundary(noSlip)
            ff = boundaryHandling.flagField
            yMid = ff.shape[1] // 2
            zMid = ff.shape[2] // 2
            y, z = np.meshgrid(range(ff.shape[1]), range(ff.shape[2]))
            ff[(y - yMid) ** 2 + (z - zMid) ** 2 > radius ** 2] = noSlipIdx
        else:
            for direction in ('N', 'S', 'T', 'B'):
                boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, dim))

    for userFunction in boundarySetupFunctions:
        userFunction(boundaryHandling, scenario.method, domainSize)

    return scenario


def createForceDrivenChannel(dim, force, domainSize=None, radius=None, length=None, lbmKernel=None,
                             optimizationParams={}, initialVelocity=None, boundarySetupFunctions=[],
                             kernelParams={}, **kwargs):
    """
    Creates a channel flow in x direction, which is driven by a constant force along the x axis
    """
    assert dim in (2, 3)
    kwargs['force'] = tuple([force, 0, 0][:dim])

    if radius is not None:
        assert length is not None
        if dim == 3:
            domainSize = (length, 2 * radius + 1, 2 * radius + 1)
            roundChannel = True
        else:
            if domainSize is None:
                domainSize = (length, 2 * radius)
    else:
        roundChannel = False

    if 'forceModel' not in kwargs:
        kwargs['forceModel'] = 'guo'

    scenario = Scenario(domainSize, kwargs, optimizationParams, lbmKernel=lbmKernel,
                        initialVelocity=initialVelocity, kernelParams=kernelParams)

    boundaryHandling = scenario.boundaryHandling
    boundaryHandling.setPeriodicity(True, False, False)
    if dim == 2:
        for direction in ('N', 'S'):
            boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, dim))
    elif dim == 3:
        if roundChannel:
            noSlipIdx = boundaryHandling.addBoundary(noSlip)
            ff = boundaryHandling.flagField
            yMid = ff.shape[1] // 2
            zMid = ff.shape[2] // 2
            y, z = np.meshgrid(range(ff.shape[1]), range(ff.shape[2]))
            ff[(y - yMid) ** 2 + (z - zMid) ** 2 > radius ** 2] = noSlipIdx
        else:
            for direction in ('N', 'S', 'T', 'B'):
                boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, dim))
    for userFunction in boundarySetupFunctions:
        userFunction(boundaryHandling, scenario.method, domainSize)

    assert domainSize is not None
    if 'forceModel' not in kwargs:
        kwargs['forceModel'] = 'guo'

    return scenario

if __name__ == '__main__':
    from lbmpy.scenarios import createForceDrivenChannel
    from lbmpy.boundaries.geometry import BlackAndWhiteImageBoundary
    from pystencils.slicing import makeSlice
    from lbmpy.boundaries import noSlip
    import numpy as np
    domainSize = (10, 10)
    scenario = createForceDrivenChannel(dim=2, method='srt', force=0.000002,
                                        domainSize=domainSize,
                                        relaxationRates=[1.92], forceModel='guo',
                                        compressible=True,
                                        optimizationParams={'target': 'gpu'})
    imageSetup = BlackAndWhiteImageBoundary("/home/staff/bauer/opensource.png",
                                            noSlip, targetSlice=makeSlice[2:-2, 1:-2])
    imageSetup(scenario.boundaryHandling, scenario.method, domainSize)
    scenario.boundaryHandling.prepare()
    scenario.run(1)
