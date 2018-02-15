import sympy as sp
import numpy as np
from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
from lbmpy.phasefield.analytical import freeEnergyFunction3Phases, freeEnergyFunctionalNPhases, symbolicOrderParameters, \
    freeEnergyFunctionalNPhasesPenaltyTerm


def createThreePhaseModel(alpha=1, kappa=(0.015, 0.015, 0.015), includeRho=True, **kwargs):
    parameters = {sp.Symbol("alpha"): alpha,
                  sp.Symbol("kappa_0"): kappa[0],
                  sp.Symbol("kappa_1"): kappa[1],
                  sp.Symbol("kappa_2"): kappa[2]}
    if includeRho:
        orderParameters = sp.symbols("rho phi psi")
        freeEnergy, transformationMatrix = freeEnergyFunction3Phases(orderParameters)
        freeEnergy = freeEnergy.subs(parameters)
        M = np.array(transformationMatrix).astype(float)
        Minv = np.array(transformationMatrix.inv()).astype(float)

        def concentrationToOrderParameters(c):
            phi = np.tensordot(c, M, axes=([-1], [1]))
            return phi

        return PhaseFieldStep(freeEnergy, orderParameters, densityOrderParameter=orderParameters[0],
                              concentrationToOrderParameters=concentrationToOrderParameters,
                              orderParametersToConcentrations=lambda phi: np.tensordot(phi, Minv, axes=([-1], [1])),
                              **kwargs)
    else:
        orderParameters = sp.symbols("phi psi")
        freeEnergy, transformationMatrix = freeEnergyFunction3Phases((1,) + orderParameters)
        freeEnergy = freeEnergy.subs(parameters)
        M = transformationMatrix.copy()
        M.row_del(0)  # rho is assumed to be 1 - is not required
        M = np.array(M).astype(float)
        reverse = transformationMatrix.inv() * sp.Matrix(sp.symbols("rho phi psi"))
        MinvTrafo = sp.lambdify(sp.symbols("phi psi"), reverse.subs(sp.Symbol("rho"), 1))

        def orderParametersToConcentrations(phi):
            phi = np.array(phi)
            transformed = MinvTrafo(phi[..., 0], phi[..., 1])
            return np.moveaxis(transformed[:, 0], 0, -1)

        def concentrationToOrderParameters(c):
            phi = np.tensordot(c, M, axes=([-1], [1]))
            return phi

        return PhaseFieldStep(freeEnergy, orderParameters, densityOrderParameter=None,
                              concentrationToOrderParameters=concentrationToOrderParameters,
                              orderParametersToConcentrations=orderParametersToConcentrations,
                              **kwargs)


def createNPhaseModel(alpha=1, numPhases=4, surfaceTensions=lambda i, j: 0.005 if i != j else 0, **kwargs):
    orderParameters = symbolicOrderParameters(numPhases-1)
    freeEnergy = freeEnergyFunctionalNPhases(numPhases, surfaceTensions, alpha, orderParameters)
    return PhaseFieldStep(freeEnergy, orderParameters, **kwargs)


def createNPhaseModelPenaltyTerm(alpha=1, numPhases=4, kappa=0.015, penaltyTermFactor=0.01, **kwargs):
    orderParameters = symbolicOrderParameters(numPhases)
    freeEnergy = freeEnergyFunctionalNPhasesPenaltyTerm(orderParameters, alpha, kappa, penaltyTermFactor)
    return PhaseFieldStep(freeEnergy, orderParameters, **kwargs)
