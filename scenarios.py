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
----    -------------------------

All of the following scenario creation functions take keyword arguments specifying which LBM method should be used
and a ``optimizationParams`` dictionary, defining performance related options. These parameters are documented
at :mod:`lbmpy.creationfunctions`. The only mandatory keyword parameter is ``relaxationRate``,
that defines the viscosity of the fluid (valid values being between 0 and 2).
"""
import numpy as np
from lbmpy.geometry import setupChannelWalls, addParabolicVelocityInflow
from lbmpy.lbstep import LatticeBoltzmannStep
from pystencils.datahandling import createDataHandling
from pystencils.slicing import sliceFromDirection
from lbmpy.boundaries import NoSlip, UBB, FixedDensity


def createFullyPeriodicFlow(initialVelocity, periodicityInKernel=False, lbmKernel=None,
                            dataHandling=None, parallel=False, **kwargs):
    """
    Creates a fully periodic setup with prescribed velocity field

    :param initialVelocity: numpy array that defines an initial velocity for each cell. The shape of this
                            array determines the domain size.
    :param periodicityInKernel: don't use boundary handling for periodicity, but directly generate the kernel periodic 
    :param lbmKernel: a LBM function, which would otherwise automatically created
    :param parallel: True for distributed memory parallelization with waLBerla
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

    if dataHandling is None:
        dataHandling = createDataHandling(parallel, domainSize, periodicity=not periodicityInKernel,
                                          defaultGhostLayers=1)
    step = LatticeBoltzmannStep(dataHandling=dataHandling, name="periodicScenario", lbmKernel=lbmKernel, **kwargs)
    for b in step.dataHandling.iterate(ghostLayers=False):
        np.copyto(b[step.velocityDataName], initialVelocity[b.globalSlice])
    return step


def createLidDrivenCavity(domainSize=None, lidVelocity=0.005, lbmKernel=None, parallel=False,
                          dataHandling=None, **kwargs):
    """
    Creates a lid driven cavity scenario

    :param domainSize: tuple specifying the number of cells in each dimension
    :param lidVelocity: x velocity of lid in lattice coordinates.
    :param lbmKernel: a LBM function, which would otherwise automatically created
    :param kwargs: other parameters are passed on to the method, see :mod:`lbmpy.creationfunctions`
    :param parallel: True for distributed memory parallelization with waLBerla
    :return: instance of :class:`Scenario`
    """
    assert domainSize is not None or dataHandling is not None
    if dataHandling is None:
        dataHandling = createDataHandling(parallel, domainSize, periodicity=False, defaultGhostLayers=1)
    step = LatticeBoltzmannStep(dataHandling=dataHandling, lbmKernel=lbmKernel, name="lidDrivenCavity" **kwargs)

    myUbb = UBB(velocity=[lidVelocity, 0, 0][:step.method.dim])
    step.boundaryHandling.setBoundary(myUbb, sliceFromDirection('N', step.dim))
    for direction in ('W', 'E', 'S') if step.dim == 2 else ('W', 'E', 'S', 'T', 'B'):
        step.boundaryHandling.setBoundary(NoSlip(), sliceFromDirection(direction, step.dim))

    return step


def createChannel(domainSize=None, force=None, pressureDifference=None, u_max=None, diameterCallback=None,
                  duct=False, wallBoundary=NoSlip(), parallel=False, dataHandling=None, **kwargs):
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
    :param parallel: True for distributed memory parallelization with waLBerla
    :param kwargs: all other keyword parameters are passed directly to scenario class.
    """
    assert domainSize is not None or dataHandling is not None

    dim = len(domainSize)
    assert dim in (2, 3)

    if [bool(p) for p in (force, pressureDifference, u_max)].count(True) != 1:
        raise ValueError("Please specify exactly one of the parameters 'force', 'pressureDifference' or 'u_max'")

    periodicity = (True, False, False) if force else (False, False, False)
    if dataHandling is None:
        dataHandling = createDataHandling(parallel, domainSize, periodicity=periodicity[:dim], defaultGhostLayers=1)

    if force:
        kwargs['force'] = tuple([force, 0, 0][:dim])
        assert dataHandling.periodicity[0]
        step = LatticeBoltzmannStep(dataHandling=dataHandling, name="forceDrivenChannel", **kwargs)
    elif pressureDifference:
        inflow = FixedDensity(1.0 + pressureDifference)
        outflow = FixedDensity(1.0)
        step = LatticeBoltzmannStep(dataHandling=dataHandling, name="pressureDrivenChannel", **kwargs)
        step.boundaryHandling.setBoundary(inflow, sliceFromDirection('W', dim))
        step.boundaryHandling.setBoundary(outflow, sliceFromDirection('E', dim))
    elif u_max:
        if duct:
            raise NotImplementedError("Velocity inflow for duct flows not yet implemented")
        step = LatticeBoltzmannStep(dataHandling=dataHandling, name="velocityDrivenChannel", **kwargs)
        diameter = diameterCallback(np.array([0]), domainSize)[0] if diameterCallback else min(domainSize[1:])
        addParabolicVelocityInflow(step.boundaryHandling, u_max, sliceFromDirection('W', dim),
                                   velCoord=0, diameter=diameter)
        outflow = FixedDensity(1.0)
        step.boundaryHandling.setBoundary(outflow, sliceFromDirection('E', dim))
    else:
        assert False

    setupChannelWalls(step.boundaryHandling, diameterCallback, duct, wallBoundary)
    return step

