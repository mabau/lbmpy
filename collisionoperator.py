import sympy as sp
from lbmpy.equilibria import getMaxwellBoltzmannEquilibriumMoments, standardDiscreteEquilibrium
import lbmpy.moments as m


# ----------------      From standard discrete equilibrium -------------------------------------------------------------


def makeSRT(stencil, order=2, compressible=False):
    momentSystem = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(discreteEquilibrium, mom, stencil) for mom in momentSystem.allMoments]
    relaxationRates = [sp.Symbol('omega')] * len(stencil)
    return LatticeModel(stencil, momentSystem.allMoments, equilibriumMoments, relaxationRates, compressible)


def makeTRT(stencil, order=2, compressible=False):
    momentSystem = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(discreteEquilibrium, mom, stencil) for mom in momentSystem.allMoments]

    lambda_e, lambda_o = sp.symbols("lambda_e lambda_o")
    relaxationRates = [lambda_e if m.isEven(moment) else lambda_o for moment in momentSystem.allMoments]
    return LatticeModel(stencil, momentSystem.allMoments, equilibriumMoments, relaxationRates, compressible)


def makeMRT(stencil, order=2, compressible=False):
    momentSystem = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(discreteEquilibrium, mom, stencil) for mom in momentSystem.allMoments]
    if not momentSystem.hasMomentGroups:
        raise NotImplementedError("No moment grouping available for this lattice model")

    relaxationRates = momentSystem.getSymbolicRelaxationRates()
    return LatticeModel(stencil, momentSystem.allMoments, equilibriumMoments, relaxationRates, compressible)


# ----------------      From Continuous Maxwell Boltzmann  -------------------------------------------------------------


def makeSRTFromMaxwellBoltzmann(stencil, order=2):
    Q = len(stencil)
    moments = m.getDefaultOrthogonalMoments(stencil)
    return LatticeModel(stencil, moments,
                        getMaxwellBoltzmannEquilibriumMoments(moments, order=order),
                        [sp.Symbol('omega')] * Q, True)


class LatticeModel:

    def __init__(self, stencil, moments, equilibriumMoments, relaxationRates, compressible):
        self._stencil = stencil
        self._moments = moments
        self._equilibriumMoments = sp.Matrix(equilibriumMoments)

        self._momentMatrix = m.momentMatrix(moments, stencil)
        self._relaxationMatrix = sp.diag(*relaxationRates)
        self.compressible = compressible

    @property
    def dim(self):
        return len(self._stencil[0])

    def getCollideTerms(self, pdfs=None):
        Q = len(self._stencil)
        if pdfs is None:
            pdfs = sp.Matrix(Q, 1, [sp.Symbol("f_%d" % (i,)) for i in range(Q)])
        elif type(pdfs) is list:
            pdfs = sp.Matrix(Q, 1, pdfs)

        M = self._momentMatrix
        return M.inv() * self._relaxationMatrix * (self._equilibriumMoments - M * pdfs)

    def getVelocityTerms(self, pdfs=None):
        Q = len(self._stencil)
        if pdfs is None:
            pdfs = sp.Matrix(Q, 1, [sp.Symbol("f_%d" % (i,)) for i in range(Q)])
        elif type(pdfs) is list:
            pdfs = sp.Matrix(Q, 1, pdfs)

        result = []
        for i in range(self.dim):
            result.append(sum([st[i] * f for st, f in zip(self._stencil, pdfs)]))
        return result

    def setCollisionDOFs(self, symbolToConstantMap):
        substitutions = [(sp.Symbol(key), value) for key, value in symbolToConstantMap.items()]
        self._relaxationMatrix = self._relaxationMatrix.subs(substitutions)

    @property
    def relaxationRates(self):
        return self._relaxationMatrix.atoms(sp.Symbol)

    @property
    def relaxationMatrix(self):
        return self._relaxationMatrix

    @property
    def momentMatrix(self):
        return self._momentMatrix

    @property
    def stencil(self):
        return self._stencil

    #def getWeights(self):
    #    result = []
    #    for f_q in feq:
    #        containingSymbols = f_q.atoms(sp.Symbol)
    #        for s in containingSymbols:
    #            if s.name.startswith('u_'):
    #                f_q = f_q.subs(s, 0)
    #            elif s.name == 'rho':
    #                f_q = f_q.subs(s, 1)
    #        result.append(f_q)
    #    return result




