"""
Scenario setup
==============

This module contains functions to set up pre-defined scenarios like a lid-driven cavity or channel flows.
It is a good starting point if you are new to lbmpy.

>>> scenario = createChannel(domainSize=(20, 10), force=1e-5,
...                          method='srt', relaxationRate=1.9)
>>> scenario.run(100)

All scenarios can be modified, for example you can create a simple channel first, then place an object in it:

>>> from lbmpy.boundaries import NoSlip
>>> from pystencils.slicing import makeSlice
>>> scenario.boundaryHandling.setBoundary(NoSlip(), makeSlice[0.3:0.4, 0.0:0.3])

Functions for scenario setup:
-----------------------------

All of the following scenario creation functions take keyword arguments specifying which LBM method should be used
and a ``optimizationParams`` dictionary, defining performance related options. These parameters are documented
at :mod:`lbmpy.creationfunctions`. The only mandatory keyword parameter is ``relaxationRate``,
that defines the viscosity of the fluid (valid values being between 0 and 2).
"""
import numpy as np
import sympy as sp

from lbmpy.boundaries.forceevaluation import calculateForceOnBoundary, calculateForceOnNoSlipBoundary
from lbmpy.geometry import setupChannelWalls, addParabolicVelocityInflow
from pystencils.field import getLayoutOfArray, createNumpyArrayWithLayout
from pystencils.slicing import sliceFromDirection, addGhostLayers, removeGhostLayers, normalizeSlice, makeSlice
from lbmpy.creationfunctions import createLatticeBoltzmannFunction, updateWithDefaultParameters, \
    switchToSymbolicRelaxationRatesForEntropicMethods
from lbmpy.macroscopic_value_kernels import compileMacroscopicValuesGetter, compileMacroscopicValuesSetter
from lbmpy.boundaries import BoundaryHandling, NoSlip, UBB, FixedDensity
from lbmpy.stencils import getStencil
from lbmpy.updatekernels import createPdfArray


# ---------------------------------------- Example Scenarios -----------------------------------------------------------


def createFullyPeriodicFlow(initialVelocity, periodicityInKernel=False, lbmKernel=None, **kwargs):
    """
    Creates a fully periodic setup with prescribed velocity field

    :param initialVelocity: numpy array that defines an initial velocity for each cell. The shape of this
                            array determines the domain size.
    :param periodicityInKernel: don't use boundary handling for periodicity, but directly generate the kernel periodic 
    :param lbmKernel: a LBM function, which would otherwise automatically created
    :param kwargs: other parameters are passed on to the method, see :mod:`lbmpy.creationfunctions`
    :return: instance of :class:`Scenario`
    """
    if 'optimizationParams' not in kwargs:
        kwargs['optimizationParams'] = {}
    else:
        kwargs['optimizationParams'] = kwargs['optimizationParams'].copy()
    domainSize = initialVelocity.shape[:-1]
    if periodicityInKernel:
        kwargs['optimizationParams']['builtinPeriodicity'] = (True, True, True)
    scenario = Scenario(domainSize, lbmKernel=lbmKernel, initialVelocity=initialVelocity, **kwargs)

    if not periodicityInKernel:
        scenario.boundaryHandling.setPeriodicity(True, True, True)
    return scenario


def createLidDrivenCavity(domainSize, lidVelocity=0.005, lbmKernel=None, **kwargs):
    """
    Creates a lid driven cavity scenario

    :param domainSize: tuple specifying the number of cells in each dimension
    :param lidVelocity: x velocity of lid in lattice coordinates.
    :param lbmKernel: a LBM function, which would otherwise automatically created
    :param kwargs: other parameters are passed on to the method, see :mod:`lbmpy.creationfunctions`
    :return: instance of :class:`Scenario`
    """
    scenario = Scenario(domainSize, lbmKernel=lbmKernel, **kwargs)

    myUbb = UBB(velocity=[lidVelocity, 0, 0][:scenario.method.dim])
    dim = scenario.method.dim
    scenario.boundaryHandling.setBoundary(myUbb, sliceFromDirection('N', dim))
    for direction in ('W', 'E', 'S') if scenario.method.dim == 2 else ('W', 'E', 'S', 'T', 'B'):
        scenario.boundaryHandling.setBoundary(NoSlip(), sliceFromDirection(direction, dim))

    return scenario


def createChannel(domainSize, force=None, pressureDifference=None, u_max=None,
                  diameterCallback=None, duct=False, wallBoundary=NoSlip(), **kwargs):
    """
    Create a channel scenario (2D or 3D)
    :param domainSize: size of the simulation domain. First coordinate is the flow direction.
    
    The channel can be driven by one of the following methods. Please specify exactly one of the following parameters: 
    :param force: Periodic channel, driven by a body force. Pass force in flow direction in lattice units here.
    :param pressureDifference: Inflow and outflow are fixed pressure conditions, with the given pressure difference. 
    :param u_max: Parabolic velocity profile prescribed at inflow, pressure boundary =1.0 at outflow.
    
    Geometry parameters:
    :param diameterCallback: optional callback for channel with varying diameters. Only valid if duct=False.
                             The callback receives x coordinate array and domainSize and returns a
                             an array of diameters of the same shape
    :param duct: if true the channel has rectangular instead of circular cross section
    :param wallBoundary: instance of boundary class that should be set at the channel walls
    :param kwargs: all other keyword parameters are passed directly to scenario class. 
    """
    dim = len(domainSize)
    assert dim in (2, 3)

    if [bool(p) for p in (force, pressureDifference, u_max)].count(True) != 1:
        raise ValueError("Please specify exactly one of the parameters 'force', 'pressureDifference' or 'u_max'")

    if force:
        kwargs['force'] = tuple([force, 0, 0][:dim])
        scenario = Scenario(domainSize, **kwargs)
        scenario.boundaryHandling.setPeriodicity(True, False, False)
    elif pressureDifference:
        scenario = Scenario(domainSize, **kwargs)
        inflow = FixedDensity(1.0 + pressureDifference)
        outflow = FixedDensity(1.0)
        scenario.boundaryHandling.setBoundary(inflow, sliceFromDirection('W', dim))
        scenario.boundaryHandling.setBoundary(outflow, sliceFromDirection('E', dim))
    elif u_max:
        scenario = Scenario(domainSize, **kwargs)
        if duct:
            raise NotImplementedError("Velocity inflow for duct flows not yet implemented")

        diameter = diameterCallback(np.array([0]), domainSize)[0] if diameterCallback else min(domainSize[1:])
        addParabolicVelocityInflow(scenario.boundaryHandling, u_max, sliceFromDirection('W', dim),
                                   velCoord=0, diameter=diameter)
        outflow = FixedDensity(1.0)
        scenario.boundaryHandling.setBoundary(outflow, sliceFromDirection('E', dim))
    else:
        assert False

    setupChannelWalls(scenario.boundaryHandling, diameterCallback, duct, wallBoundary)
    return scenario


# ------------------------------------------ Scenario Class ------------------------------------------------------------


class Scenario(object):
    """Scenario containing boundary handling and LBM update function

    You probably want to use one of the simpler scenario factory functions of this module instead of using
    this constructor.

    :param domainSize: tuple, defining the domain size without ghost layers
    :param optimizationParams: dict with optimization parameters, as documented in :mod:`lbmpy.creationfunctions`,
                               passed to :func:`lbmpy.creationfunctions.createLatticeBoltzmannFunction`
    :param lbmKernel: a lattice boltzmann function can be passed here, if None it is created with the parameters
                      specified above
    :param initialVelocity: tuple with initial velocity of the domain, can either be a constant or a numpy array
                            with first axes shaped like the domain, and the last dimension of size #dimensions
    :param preUpdateFunctions: list of functions that are called before the LBM kernel. They get the pdf array as
                               only argument. Can be used for custom boundary conditions, periodicity, etc.
    :param kernelParams: additional parameters passed to the sweep
    :param kwargs:  dict with method parameters, as documented in :mod:`lbmpy.creationfunctions`,
                             passed to :func:`lbmpy.creationfunctions.createLatticeBoltzmannFunction`
    """

    def __init__(self, domainSize, optimizationParams={}, lbmKernel=None,
                 initialVelocity=None, preUpdateFunctions=[], kernelParams={}, **kwargs):
        methodParameters = kwargs
        ghostLayers = 1
        domainSizeWithGhostLayer = tuple([s + 2 * ghostLayers for s in domainSize])
        D = len(domainSize)
        if 'stencil' not in methodParameters:
            methodParameters['stencil'] = 'D2Q9' if D == 2 else 'D3Q27'

        self.kernelParams = kernelParams

        methodParameters, optimizationParams = updateWithDefaultParameters(methodParameters, optimizationParams)

        # Automatic handling of fixed relaxation rate in entropic scenario
        if methodParameters['entropic']:
            relaxationRates = methodParameters['relaxationRates']
            hasTwoDifferentRates = len(set(relaxationRates)) == 2
            noSymbolicRelaxationRates = all(not isinstance(e, sp.Symbol) for e in relaxationRates)
            if noSymbolicRelaxationRates and hasTwoDifferentRates:
                substitutions = {}
                for i, value in enumerate(set(relaxationRates)):
                    newSymbol = sp.Symbol("omega_ent_%d" % (i,))
                    assert newSymbol not in self.kernelParams
                    self.kernelParams[newSymbol.name] = value
                    substitutions[sp.sympify(value)] = newSymbol
                methodParameters['relaxationRates'] = [sp.sympify(v).subs(substitutions) for v in relaxationRates]

        Q = len(getStencil(methodParameters['stencil']))
        self._pdfArrays = [createPdfArray(domainSize, Q, layout=optimizationParams['fieldLayout']),
                           createPdfArray(domainSize, Q, layout=optimizationParams['fieldLayout'])]

        if isinstance(initialVelocity, np.ndarray):
            initialVelocity = addGhostLayers(initialVelocity, indexDimensions=1, ghostLayers=1,
                                             layout=getLayoutOfArray(self._pdfArrays[0]))

        # Create kernel
        if lbmKernel is None:
            switchToSymbolicRelaxationRatesForEntropicMethods(methodParameters, self.kernelParams)
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
        self.boundaryHandling.prepare() # make sure that boundary setup time does not enter benchmark
        start = perf_counter()
        self.run(timeSteps)
        duration = perf_counter() - start
        durationOfTimeStep = duration / timeSteps
        mlups = self.numberOfCells / durationOfTimeStep * 1e-6
        return mlups

    def writeVTK(self, fileBaseName="vtk"):
        from pystencils.vtk import imageToVTK
        imageToVTK("%s_%06d" % (fileBaseName, self.timeStepsRun,),
                   cellData={
                       'velocity': (np.ascontiguousarray(self.velocity[..., 0].filled(0.0)),
                                    np.ascontiguousarray(self.velocity[..., 1].filled(0.0)),
                                    np.ascontiguousarray(self.velocity[..., 2].filled(0.0))),
                       'density': np.ascontiguousarray(self.density.filled(0.0)),
                   })

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
        mask = np.logical_not(self.boundaryHandling.getMask('fluid'))
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
        mask = np.logical_not(self._boundaryHandling.getMask('fluid'))
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

    def calculateForceOnBoundary(self, boundaryObject):
        """Computes force on boundary using simple momentum exchange method"""
        if isinstance(boundaryObject, NoSlip):
            return calculateForceOnNoSlipBoundary(boundaryObject, self.boundaryHandling, self._pdfArrays[0])
        else:
            self.runBoundaryHandlingOnly()
            return calculateForceOnBoundary(boundaryObject, self.boundaryHandling, self._pdfArrays[0])

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

    def runBoundaryHandlingOnly(self):
        isGpuSimulation = len(self._pdfGpuArrays) > 0
        if isGpuSimulation:
            self._pdfGpuArrays[0].set(self._pdfArrays[0])
            self._boundaryHandling(pdfs=self._pdfGpuArrays[0], **self.kernelParams)
            self._pdfGpuArrays[0].get(self._pdfArrays[0])
        else:
            self._boundaryHandling(pdfs=self._pdfArrays[0], **self.kernelParams)

