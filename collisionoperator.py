import sympy as sp
from lbmpy.equilibria import getMaxwellBoltzmannEquilibriumMoments, standardDiscreteEquilibrium
import lbmpy.moments as m


def makeSRT(stencil, order=2, compressible=False):
    Q = len(stencil)
    moments = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(discreteEquilibrium, mom, stencil) for mom in moments]
    return LatticeModel(stencil, moments, equilibriumMoments, relaxationFactors=[sp.Symbol('omega')] * Q)


def makeTRT(stencil, order=2, compressible=False):
    Q = len(stencil)
    moments = m.getDefaultMoments(Q)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(discreteEquilibrium, mom, stencil) for mom in moments]

    lambda_e, lambda_o = sp.symbols("lambda_e lambda_o")
    relaxationFactors = [lambda_e if m.isEven(moment) else lambda_o for moment in moments]
    return LatticeModel(stencil, moments, equilibriumMoments, relaxationFactors=relaxationFactors)


def makeSRTFromMaxwellBoltzmann(stencil, order=2):
    Q = len(stencil)
    moments = m.getDefaultOrthogonalMoments(stencil)
    return LatticeModel(stencil, moments,
                        getMaxwellBoltzmannEquilibriumMoments(moments, order=order),
                        relaxationFactors=[sp.Symbol('omega')] * Q)


class LatticeModel:

    def __init__(self, stencil, moments, equilibriumMoments, relaxationFactors):
        self._stencil = stencil
        self._moments = moments
        self._equilibriumMoments = sp.Matrix(equilibriumMoments)

        self._momentMatrix = m.momentMatrix(moments, stencil)
        self._relaxationMatrix = sp.diag(*relaxationFactors)
        #self._pdfs = sp.Matrix(len(stencil), 1, [sp.Symbol("f_%d" % (i,)) for i in range(len(stencil))])

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

    @property
    def collisionDOFs(self):
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




