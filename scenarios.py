"""
Scenario setup
==============

This module contains functions to set up pre-defined scenarios like a lid-driven cavity or channel flows.
It is a good starting point if you are new to lbmpy.

>>> scenario = createForceDrivenChannel(dim=2, radius=10, length=20, force=1e-5,
...                                     method='srt', relaxationRate=1.9)
>>> scenario.run(100)

All scenarios can be modified, for example you can create a simple channel first, then place an object in it:

>>> from lbmpy.boundaries import noSlip
>>> from pystencils.slicing import makeSlice
>>> scenario.boundaryHandling.setBoundary(noSlip, makeSlice[0.3:0.4, 0.0:0.3])

Functions for scenario setup:
-----------------------------

All of the following scenario creation functions take keyword arguments specifying which LBM method should be used
and a ``optimizationParams`` dictionary, defining performance related options. These parameters are documented
at :mod:`lbmpy.creationfunctions`. The only mandatory keyword parameter is ``relaxationRate``,
that defines the viscosity of the fluid (valid values being between 0 and 2).
"""
import numpy as np
import sympy as sp
from functools import partial
from pystencils.field import getLayoutOfArray, createNumpyArrayWithLayout
from pystencils.slicing import sliceFromDirection, addGhostLayers, removeGhostLayers, normalizeSlice, makeSlice
from lbmpy.creationfunctions import createLatticeBoltzmannFunction, updateWithDefaultParameters
from lbmpy.macroscopic_value_kernels import compileMacroscopicValuesGetter, compileMacroscopicValuesSetter
from lbmpy.boundaries import BoundaryHandling, noSlip, ubb, fixedDensity
from lbmpy.stencils import getStencil
from lbmpy.updatekernels import createPdfArray


# ---------------------------------------- Example Scenarios -----------------------------------------------------------


def createFullyPeriodicFlow(initialVelocity, periodicityInKernel=False,
                            optimizationParams={}, lbmKernel=None, kernelParams={}, **kwargs):
    """
    Creates a fully periodic setup with prescribed velocity field

    :param initialVelocity: numpy array that defines an initial velocity for each cell. The shape of this
                            array determines the domain size.
    :param periodicityInKernel: don't use boundary handling for periodicity, but directly generate the kernel periodic 
    :param optimizationParams: see :mod:`lbmpy.creationfunctions`
    :param lbmKernel: a LBM function, which would otherwise automatically created
    :param kernelParams: additional parameters passed to the sweep
    :param kwargs: other parameters are passed on to the method, see :mod:`lbmpy.creationfunctions`
    :return: instance of :class:`Scenario`
    """
    optimizationParams = optimizationParams.copy()
    domainSize = initialVelocity.shape[:-1]
    if periodicityInKernel:
        optimizationParams['builtinPeriodicity'] = (True, True, True)
    scenario = Scenario(domainSize, kwargs, optimizationParams, lbmKernel, initialVelocity, kernelParams=kernelParams)

    if not periodicityInKernel:
        scenario.boundaryHandling.setPeriodicity(True, True, True)
    return scenario


def createLidDrivenCavity(domainSize, lidVelocity=0.005, optimizationParams={}, lbmKernel=None,
                          kernelParams={}, **kwargs):
    """
    Creates a lid driven cavity scenario

    :param domainSize: tuple specifying the number of cells in each dimension
    :param lidVelocity: x velocity of lid in lattice coordinates.
    :param optimizationParams: see :mod:`lbmpy.creationfunctions`
    :param lbmKernel: a LBM function, which would otherwise automatically created
    :param kernelParams: additional parameters passed to the sweep
    :param kwargs: other parameters are passed on to the method, see :mod:`lbmpy.creationfunctions`
    :return: instance of :class:`Scenario`
    """
    scenario = Scenario(domainSize, kwargs, optimizationParams, lbmKernel=lbmKernel, kernelParams=kernelParams)

    myUbb = partial(ubb, velocity=[lidVelocity, 0, 0][:scenario.method.dim])
    myUbb.name = 'ubb'
    dim = scenario.method.dim
    scenario.boundaryHandling.setBoundary(myUbb, sliceFromDirection('N', dim))
    for direction in ('W', 'E', 'S') if scenario.method.dim == 2 else ('W', 'E', 'S', 'T', 'B'):
        scenario.boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, dim))

    return scenario


def createForceDrivenChannel(force=1e-6, domainSize=None, dim=2, radius=None, length=None, initialVelocity=None,
                             optimizationParams={}, lbmKernel=None, kernelParams={}, **kwargs):
    """
    Creates a channel flow in x direction, which is driven by a constant force along the x axis

    :param force: force in x direction (lattice units) that drives the channel
    :param domainSize: tuple with size of channel in x, y, (z) direction. If not specified, pass dim, radius and length.
                       In 3D, this creates a channel with rectangular cross section
    :param dim: dimension of the channel (only required if domainSize is not passed)
    :param radius: radius in 3D, or half of the height in 2D  (only required if domainSize is not passed).
                   In 3D, this creates a channel with circular cross section
    :param length: extend in x direction (only required if domainSize is not passed)
    :param initialVelocity: initial velocity, either array to specify velocity for each cell or tuple for constant
    :param optimizationParams: see :mod:`lbmpy.creationfunctions`
    :param lbmKernel: a LBM function, which would otherwise automatically created
    :param kernelParams: additional parameters passed to the sweep
    :param kwargs: other parameters are passed on to the method, see :mod:`lbmpy.creationfunctions`
    :return: instance of :class:`Scenario`
    """
    if domainSize is not None:
        dim = len(domainSize)
    else:
        if dim is None or radius is None or length is None:
            raise ValueError("Pass either 'domainSize' or 'dim', 'radius' and 'length'")

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

    assert domainSize is not None
    if 'forceModel' not in kwargs:
        kwargs['forceModel'] = 'guo'

    return scenario


def createPressureGradientDrivenChannel(pressureDifference, domainSize=None, dim=2, radius=None, length=None,
                                        initialVelocity=None, optimizationParams={},
                                        lbmKernel=None, kernelParams={}, **kwargs):
    """
    Creates a channel flow in x direction, which is driven by two pressure boundaries.
    Consider using :func:`createForceDrivenChannel` which does not have artifacts an inflow and outflow.

    :param pressureDifference: pressure drop in channel in lattice units
    :param domainSize: tuple with size of channel in x, y, (z) direction. If not specified, pass dim, radius and length.
                       In 3D, this creates a channel with rectangular cross section
    :param dim: dimension of the channel (only required if domainSize is not passed)
    :param radius: radius in 3D, or half of the height in 2D  (only required if domainSize is not passed).
                   In 3D, this creates a channel with circular cross section
    :param length: extend in x direction (only required if domainSize is not passed)
    :param initialVelocity: initial velocity, either array to specify velocity for each cell or tuple for constant
    :param optimizationParams: see :mod:`lbmpy.creationfunctions`
    :param lbmKernel: a LBM function, which would otherwise automatically created
    :param kernelParams: additional parameters passed to the sweep
    :param kwargs: other parameters are passed on to the method, see :mod:`lbmpy.creationfunctions`
    :return: instance of :class:`Scenario`
    """
    if domainSize is not None:
        dim = len(domainSize)
    else:
        if dim is None or radius is None or length is None:
            raise ValueError("Pass either 'domainSize' or 'dim', 'radius' and 'length'")

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

    assert dim in (2, 3)

    scenario = Scenario(domainSize, kwargs, optimizationParams, lbmKernel=lbmKernel,
                        initialVelocity=initialVelocity, kernelParams=kernelParams)
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

    return scenario


# ------------------------------------------ Scenario Class ------------------------------------------------------------


class Scenario(object):
    """Scenario containing boundary handling and LBM update function

    You probably want to use one of the simpler scenario factory functions of this module instead of using
    this constructor.

    :param domainSize: tuple, defining the domain size without ghost layers
    :param methodParameters: dict with method parameters, as documented in :mod:`lbmpy.creationfunctions`,
                             passed to :func:`lbmpy.creationfunctions.createLatticeBoltzmannFunction`
    :param optimizationParams: dict with optimization parameters, as documented in :mod:`lbmpy.creationfunctions`,
                               passed to :func:`lbmpy.creationfunctions.createLatticeBoltzmannFunction`
    :param lbmKernel: a lattice boltzmann function can be passed here, if None it is created with the parameters
                      specified above
    :param initialVelocity: tuple with initial velocity of the domain, can either be a constant or a numpy array
                            with first axes shaped like the domain, and the last dimension of size #dimensions
    :param preUpdateFunctions: list of functions that are called before the LBM kernel. They get the pdf array as
                               only argument. Can be used for custom boundary conditions, periodicity, etc.
    :param kernelParams: additional parameters passed to the sweep
    """

    def __init__(self, domainSize, methodParameters, optimizationParams, lbmKernel=None,
                 initialVelocity=None, preUpdateFunctions=[], kernelParams={}):
        ghostLayers = 1
        domainSizeWithGhostLayer = tuple([s + 2 * ghostLayers for s in domainSize])
        D = len(domainSize)
        if 'stencil' not in methodParameters:
            methodParameters['stencil'] = 'D2Q9' if D == 2 else 'D3Q27'

        methodParameters, optimizationParams = updateWithDefaultParameters(methodParameters, optimizationParams)

        Q = len(getStencil(methodParameters['stencil']))
        self._pdfArrays = [createPdfArray(domainSize, Q, layout=optimizationParams['fieldLayout']),
                           createPdfArray(domainSize, Q, layout=optimizationParams['fieldLayout'])]

        if isinstance(initialVelocity, np.ndarray):
            initialVelocity = addGhostLayers(initialVelocity, indexDimensions=1, ghostLayers=1,
                                             layout=getLayoutOfArray(self._pdfArrays[0]))

        # Create kernel
        if lbmKernel is None:
            if methodParameters['entropic']:
                newRelaxationRates = []
                for rr in methodParameters['relaxationRates']:
                    if not isinstance(rr, sp.Symbol):
                        dummyVar = sp.Dummy()
                        newRelaxationRates.append(dummyVar)
                        kernelParams[dummyVar.name] = rr
                    else:
                        newRelaxationRates.append(rr)
                if len(newRelaxationRates) < 2:
                    newRelaxationRates.append(sp.Dummy())
                methodParameters['relaxationRates'] = newRelaxationRates

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
                                                  target=optimizationParams['target'],
                                                  openMP=optimizationParams['openMP'])

        self._preUpdateFunctions = preUpdateFunctions
        self.kernelParams = kernelParams
        self._pdfGpuArrays = []
        self.timeStepsRun = 0
        self.domainSize = domainSize

        if initialVelocity is None:
            initialVelocity = [0] * D

        setMacroscopic = compileMacroscopicValuesSetter(self.method, {'density': 1.0, 'velocity': initialVelocity},
                                                        pdfArr=self._pdfArrays[0], target='cpu')
        setMacroscopic(pdfs=self._pdfArrays[0], **self.kernelParams)

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
        self.timeStepsRun += timeSteps

    def benchmarkRun(self, timeSteps):
        from time import perf_counter
        start = perf_counter()
        self.run(timeSteps)
        duration = perf_counter() - start
        durationOfTimeStep = duration / timeSteps
        mlups = self.numberOfCells / durationOfTimeStep * 1e-6
        return mlups

    @property
    def numberOfCells(self):
        result = 1
        for d in self.domainSize:
            result *= d
        return result

    def benchmark(self, timeForBenchmark=5, initTimeSteps=10, numberOfTimeStepsForEstimation=10):
        """
        Returns the number of MLUPS (million lattice update per second) for this scenario
        
        :param timeForBenchmark: number of seconds benchmark should take
        :param initTimeSteps: number of time steps run initially for warm up, to get arrays into cache etc
        :param numberOfTimeStepsForEstimation: time steps run before real benchmarks, to determine number of time steps
                                               that approximately take 'timeForBenchmark'
        :return: MLUP/s: number of cells updated per second times 1e-6 
        """
        from time import perf_counter

        self.run(initTimeSteps)

        # Run a few time step to get first estimate
        start = perf_counter()
        self.run(numberOfTimeStepsForEstimation)
        duration = perf_counter() - start

        # Run for approximately 'timeForBenchmark' seconds
        durationOfTimeStep = duration / numberOfTimeStepsForEstimation
        timeSteps = int(timeForBenchmark / durationOfTimeStep)
        timeSteps = max(timeSteps, 6)
        return self.benchmarkRun(timeSteps)

    @property
    def velocity(self):
        """Velocity as numpy array"""
        mask = np.logical_not(np.bitwise_and(self.boundaryHandling.flagField, self.boundaryHandling._fluidFlag))
        mask = np.repeat(mask[..., np.newaxis], self.dim, axis=2)
        return removeGhostLayers(np.ma.masked_array(self._velocity, mask), indexDimensions=1)

    @property
    def vorticity(self):
        if self.dim != 2:
            raise NotImplementedError("Vorticity only implemented for 2D scenarios")
        vel = self.velocity
        grad_y_of_x = np.gradient(vel[:, :, 0], axis=1)
        grad_x_of_y = np.gradient(vel[:, :, 1], axis=0)
        return grad_x_of_y - grad_y_of_x

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
                self._boundaryHandling(pdfs=pdfArrays[0], **self.kernelParams)
            self._lbmKernel(src=pdfArrays[0], dst=pdfArrays[1], **self.kernelParams)

            pdfArrays[0], pdfArrays[1] = pdfArrays[1], pdfArrays[0]  # swap

    def _cpuTimeloop(self, timeSteps):
        self._timeloop(self._pdfArrays, timeSteps)
        self._getMacroscopic(pdfs=self._pdfArrays[0], density=self._density, velocity=self._velocity,
                             **self.kernelParams)

    def _gpuTimeloop(self, timeSteps):
        # Transfer data to gpu
        for cpuArr, gpuArr in zip(self._pdfArrays, self._pdfGpuArrays):
            gpuArr.set(cpuArr)

        self._timeloop(self._pdfGpuArrays, timeSteps)

        # Transfer data from gpu to cpu
        for cpuArr, gpuArr in zip(self._pdfArrays, self._pdfGpuArrays):
            gpuArr.get(cpuArr)

        self._getMacroscopic(pdfs=self._pdfArrays[0], density=self._density, velocity=self._velocity,
                             **self.kernelParams)
