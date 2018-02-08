import sympy as sp
from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
from lbmpy.phasefield.analytical import freeEnergyFunction3Phases, freeEnergyFunctionalNPhases, symbolicOrderParameters


def createThreePhaseModel(alpha=1, kappa=(0.015, 0.015, 0.015), includeRho=True, **kwargs):
    parameters = {sp.Symbol("alpha"): alpha,
                  sp.Symbol("kappa_0"): kappa[0],
                  sp.Symbol("kappa_1"): kappa[1],
                  sp.Symbol("kappa_2"): kappa[2]}
    if includeRho:
        orderParameters = sp.symbols("rho phi psi")
        freeEnergy = freeEnergyFunction3Phases(orderParameters).subs(parameters)
        return PhaseFieldStep(freeEnergy, orderParameters, densityOrderParameter=orderParameters[0], **kwargs)
    else:
        orderParameters = sp.symbols("phi psi")
        freeEnergy = freeEnergyFunction3Phases((1,) + orderParameters).subs(parameters)
        return PhaseFieldStep(freeEnergy, orderParameters, densityOrderParameter=None, **kwargs)


def createNPhaseModel(alpha=1, numPhases=4, surfaceTensions=lambda i, j: 0.005 if i != j else 0, **kwargs):
    orderParameters = symbolicOrderParameters(numPhases-1)
    freeEnergy = freeEnergyFunctionalNPhases(numPhases, surfaceTensions, alpha, orderParameters)
    return PhaseFieldStep(freeEnergy, orderParameters, **kwargs)
