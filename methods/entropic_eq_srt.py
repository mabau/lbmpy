import sympy as sp
from pystencils import Assignment
from lbmpy.maxwellian_equilibrium import getWeights
from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation


class EntropicEquilibriumSRT(AbstractLbMethod):
    def __init__(self, stencil, relaxationRate, forceModel, conservedQuantityCalculation):
        super(EntropicEquilibriumSRT, self).__init__(stencil)

        self._cqc = conservedQuantityCalculation
        self._weights = getWeights(stencil, c_s_sq=sp.Rational(1, 3))
        self._relaxationRate = relaxationRate
        self._forceModel = forceModel
        self.shearRelaxationRate = relaxationRate

    @property
    def conservedQuantityComputation(self):
        return self._cqc

    @property
    def weights(self):
        return self._weights

    @property
    def zerothOrderEquilibriumMomentSymbol(self, ):
        return self._cqc.zerothOrderMomentSymbol

    @property
    def firstOrderEquilibriumMomentSymbols(self, ):
        return self._cqc.firstOrderMomentSymbols

    def getEquilibrium(self, conservedQuantityEquations=None, includeForceTerms=False):
        return self._getCollisionRuleWithRelaxationRate(1, conservedQuantityEquations=conservedQuantityEquations,
                                                        includeForceTerms=includeForceTerms)

    def _getCollisionRuleWithRelaxationRate(self, relaxationRate, includeForceTerms=True,
                                            conservedQuantityEquations=None):
        f = sp.Matrix(self.preCollisionPdfSymbols)
        rho = self._cqc.zerothOrderMomentSymbol
        u = self._cqc.firstOrderMomentSymbols

        if conservedQuantityEquations is None:
            conservedQuantityEquations = self._cqc.equilibriumInputEquationsFromPdfs(f)
        allSubexpressions = conservedQuantityEquations.allEquations

        eq = []
        for w_i, direction in zip(self.weights, self.stencil):
            f_i = rho * w_i
            for u_a, e_ia in zip(u, direction):
                B = sp.sqrt(1 + 3 * u_a ** 2)
                f_i *= (2 - B) * ((2 * u_a + B) / (1 - u_a)) ** e_ia
            eq.append(f_i)

        collisionEqs = [Assignment(lhs, (1 - relaxationRate) * f_i + relaxationRate * eq_i)
                        for lhs, f_i, eq_i in zip(self.postCollisionPdfSymbols, self.preCollisionPdfSymbols, eq)]

        if (self._forceModel is not None) and includeForceTerms:
            forceModelTerms = self._forceModel(self)
            forceTermSymbols = sp.symbols("forceTerm_:%d" % (len(forceModelTerms, )))
            forceSubexpressions = [Assignment(sym, forceModelTerm)
                                   for sym, forceModelTerm in zip(forceTermSymbols, forceModelTerms)]
            allSubexpressions += forceSubexpressions
            collisionEqs = [Assignment(eq.lhs, eq.rhs + forceTermSymbol)
                            for eq, forceTermSymbol in zip(collisionEqs, forceTermSymbols)]

        return LbmCollisionRule(self, collisionEqs, allSubexpressions)

    def getCollisionRule(self):
        return self._getCollisionRuleWithRelaxationRate(self._relaxationRate)


def createEntropicSRT(stencil, relaxationRate, forceModel, compressible):
    if not compressible:
        raise NotImplementedError("entropic-srt only implemented for compressible models")
    densityVelocityComputation = DensityVelocityComputation(stencil, compressible, forceModel)
    return EntropicEquilibriumSRT(stencil, relaxationRate, forceModel, densityVelocityComputation)
