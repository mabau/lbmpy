import numpy as np
import sympy as sp

from lbmpy.chapman_enskog import Diff
from pystencils import makeSlice


def plotStatus(phaseFieldStep, fromX=None, toX=None):
    import lbmpy.plot2d as plt

    domainSize = phaseFieldStep.dataHandling.shape
    assert len(domainSize) == 2 and domainSize[1] == 1, "Not a 1D scenario"

    dh = phaseFieldStep.dataHandling

    numPhases = phaseFieldStep.numOrderParameters

    plt.subplot(1, 3, 1)
    plt.title('φ')
    phiName = phaseFieldStep.phiFieldName
    for i in range(numPhases):
        plt.plot(dh.gatherArray(phiName, makeSlice[fromX:toX, 0, i]), marker='x', label='φ_%d' % (i,))
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("μ")
    muName = phaseFieldStep.muFieldName
    for i in range(numPhases):
        plt.plot(dh.gatherArray(muName, makeSlice[fromX:toX, 0, i]), marker='x', label='μ_%d' % (i,));
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Force and Velocity")
    plt.plot(dh.gatherArray(phaseFieldStep.forceFieldName, makeSlice[fromX:toX, 0, 0]), label='F', marker='x')
    plt.plot(dh.gatherArray(phaseFieldStep.velFieldName, makeSlice[fromX:toX, 0, 0]), label='u', marker='v')
    plt.legend()


def plotFreeEnergyBulkContours(freeEnergy, orderParameters, phase0=0, phase1=1, **kwargs):
    import lbmpy.plot2d as plt

    x = np.linspace(-.2, 1.2, 100)
    y = np.linspace(-.2, 1.2, 100)
    xg, yg = np.meshgrid(x, y)
    substitutions = {op: 0 for i, op in enumerate(orderParameters) if i not in (phase0, phase1)}
    substitutions.update({d: 0 for d in freeEnergy.atoms(Diff)})  # remove interface components of free energy
    freeEnergyLambda = sp.lambdify([orderParameters[phase0], orderParameters[phase1]],
                                   freeEnergy.subs(substitutions))
    if 'levels' not in kwargs:
        kwargs['levels'] = np.linspace(0, 1, 60)
    plt.contour(x, y, freeEnergyLambda(xg, yg), **kwargs)


def initSharpInterface(pfStep, phaseIdx, x1=None, x2=None, inverse=False):
    domainSize = pfStep.dataHandling.shape
    if x1 is None:
        x1 = domainSize[0] // 4
    if x2 is None:
        x2 = 3 * x1

    if phaseIdx >= pfStep.numOrderParameters:
        return

    for b in pfStep.dataHandling.iterate():
        x = b.cellIndexArrays[0]
        mid = np.logical_and(x1 < x, x < x2)

        phi = b[pfStep.phiFieldName]
        val1, val2 = (1, 0) if inverse else (0, 1)

        phi[..., phaseIdx].fill(val1)
        phi[mid, phaseIdx] = val2

    pfStep.setPdfFieldsFromMacroscopicValues()


def tanhTest(pfStep, phase0, phase1, expectedInterfaceWidth=1, timeSteps=10000):
    """
    Initializes a sharp interface and checks if tanh-shaped profile is developing
    :param pfStep: phase field scenario / step
    :param phase0: index of first phase to initialize
    :param phase1: index of second phase to initialize inversely
    :param expectedInterfaceWidth: interface width parameter alpha that is used in analytical form
    :param timeSteps: number of time steps run before evaluation
    :return: deviation of simulated profile from analytical solution as average(abs(simulation-analytic))
    """
    import lbmpy.plot2d as plt
    from lbmpy.phasefield.analytical import analyticInterfaceProfile

    domainSize = pfStep.dataHandling.shape
    pfStep.reset()
    pfStep.dataHandling.fill(pfStep.phiFieldName, 0)
    initSharpInterface(pfStep, phaseIdx=phase0, inverse=False)
    initSharpInterface(pfStep, phaseIdx=phase1, inverse=True)
    pfStep.setPdfFieldsFromMacroscopicValues()
    pfStep.run(timeSteps)

    visWidth = 20
    x = np.arange(visWidth) - (visWidth // 2)
    analytic = np.array([analyticInterfaceProfile(x_i - 0.5, expectedInterfaceWidth) for x_i in x], dtype=np.float64)

    stepLocation = domainSize[0] // 4
    simulated = pfStep.phi[stepLocation - visWidth//2:stepLocation + visWidth//2, 0, phase0]
    plt.plot(analytic, label='analytic', marker='o')
    plt.plot(simulated, label='simulated', marker='x')
    plt.legend()

    return np.average(np.abs(simulated - analytic))


def galileanInvarianceTest(pfStep, velocity=0.05, rounds=3, phase0=0, phase1=1,
                           expectedInterfaceWidth=1, initTimeSteps=5000):
    """
    Moves interface at constant speed through periodic domain - check if momentum is conserved
    :param pfStep: phase field scenario / step
    :param velocity: constant velocity to move interface
    :param rounds: how many times the interface should travel through the domain
    :param phase0: index of first phase to initialize
    :param phase1: index of second phase to initialize inversely
    :param expectedInterfaceWidth: interface width parameter alpha that is used in analytical form
    :param initTimeSteps: before velocity is set, this many time steps are run to let interface settle to tanh shape
    :return: change in velocity
    """
    import lbmpy.plot2d as plt
    from lbmpy.phasefield.analytical import analyticInterfaceProfile

    domainSize = pfStep.dataHandling.shape
    roundTimeSteps = int((domainSize[0]+0.25) / velocity)

    print("Velocity:", velocity, " Timesteps for round:", roundTimeSteps)

    pfStep.reset()
    pfStep.dataHandling.fill(pfStep.phiFieldName, 0)
    initSharpInterface(pfStep, phaseIdx=phase0, inverse=False)
    initSharpInterface(pfStep, phaseIdx=phase1, inverse=True)
    pfStep.setPdfFieldsFromMacroscopicValues()

    print("Running", initTimeSteps, "initial time steps")
    pfStep.run(initTimeSteps)
    pfStep.dataHandling.fill(pfStep.velFieldName, velocity, fValue=0)
    pfStep.setPdfFieldsFromMacroscopicValues()

    stepLocation = domainSize[0] // 4
    visWidth = 20

    simulatedProfiles = []

    def captureProfile():
        simulated = pfStep.phi[stepLocation - visWidth // 2:stepLocation + visWidth // 2, 0, phase0].copy()
        simulatedProfiles.append(simulated)

    captureProfile()
    for rt in range(rounds ):
        print("Running round %d/%d" % (rt+1, rounds))
        pfStep.run(roundTimeSteps)
        captureProfile()

    x = np.arange(visWidth) - (visWidth // 2)
    analytic = np.array([analyticInterfaceProfile(x_i - 0.5, expectedInterfaceWidth) for x_i in x], dtype=np.float64)

    plt.plot(x, analytic, label='analytic', marker='o')
    for i, profile in enumerate(simulatedProfiles):
        plt.plot(x, profile, label="After %d rounds" % (i,))

    plt.legend()

    return np.average(pfStep.velocity[:, 0, 0]) - velocity
