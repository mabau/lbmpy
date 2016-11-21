import sympy as sp

from lbmpy.densityVelocityExpressions import getDensityVelocityExpressions
from lbmpy.equilibria import getMaxwellBoltzmannEquilibriumMoments, standardDiscreteEquilibrium, getWeights
import lbmpy.moments as m
import lbmpy.transformations as trafos
import lbmpy.util as util


# ----------------      From standard discrete equilibrium -------------------------------------------------------------


def makeSRT(stencil, order=2, compressible=False, forceModel=None):
    momentSystem = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(discreteEquilibrium, mom, stencil) for mom in momentSystem.allMoments]
    relaxationRates = [sp.Symbol('omega')] * len(stencil)
    return MomentRelaxationLatticeModel(stencil, momentSystem.allMoments, equilibriumMoments,
                                        relaxationRates, compressible, forceModel)


def makeTRT(stencil, order=2, compressible=False, forceModel=None):
    momentSystem = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(discreteEquilibrium, mom, stencil) for mom in momentSystem.allMoments]

    lambda_e, lambda_o = sp.symbols("lambda_e lambda_o")
    relaxationRates = [lambda_e if m.isEven(moment) else lambda_o for moment in momentSystem.allMoments]
    return MomentRelaxationLatticeModel(stencil, momentSystem.allMoments, equilibriumMoments,
                                        relaxationRates, compressible, forceModel)


def makeMRT(stencil, order=2, compressible=False, forceModel=None):
    momentSystem = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(discreteEquilibrium, mom, stencil) for mom in momentSystem.allMoments]
    if not momentSystem.hasMomentGroups:
        raise NotImplementedError("No moment grouping available for this lattice model")

    relaxationRates = momentSystem.getSymbolicRelaxationRates()
    return MomentRelaxationLatticeModel(stencil, momentSystem.allMoments, equilibriumMoments,
                                        relaxationRates, compressible, forceModel)


# ----------------      From Continuous Maxwell Boltzmann  -------------------------------------------------------------


def makeSRTFromMaxwellBoltzmann(stencil, order=2, forceModel=None):
    Q = len(stencil)
    moments = m.getDefaultOrthogonalMoments(stencil)
    return LatticeModel(stencil, moments,
                        getMaxwellBoltzmannEquilibriumMoments(moments, order=order),
                        [sp.Symbol('omega')] * Q, True, forceModel)


# ----------------      From Continuous Maxwell Boltzmann  -------------------------------------------------------------


class LbmCollisionRule:
    def __init__(self, updateEquations, subExpressions, latticeModel):
        self.subexpressions = subExpressions
        self.updateEquations = updateEquations
        self.latticeModel = latticeModel

    def newWithSubexpressions(self, newUpdateEquations, newSubexpressions):
        assert len(self.updateEquations) == len(newUpdateEquations)
        return LbmCollisionRule(newUpdateEquations, self.subexpressions+newSubexpressions, self.latticeModel)

    def newWithSubstitutions(self, substitutionDict):
        newSubexpressions = [e.subs(substitutionDict) for e in self.subexpressions]
        newUpdateEquations = [e.subs(substitutionDict) for e in self.updateEquations]
        return LbmCollisionRule(newUpdateEquations, newSubexpressions, self.latticeModel)

    def countNumberOfOperations(self):
        return trafos.countNumberOfOperations(self.subexpressions + self.updateEquations)

    @property
    def equations(self):
        return self.subexpressions + self.updateEquations


class LatticeModel:
    def __init__(self, stencil, relaxationRates, compressible, forceModel=None):
        self._stencil = stencil
        self._compressible = compressible
        self._forceModel = forceModel
        self._relaxationRates = relaxationRates

    @property
    def stencil(self):
        """Sequence of directions (discretization of velocity space)"""
        return self._stencil

    @property
    def compressible(self):
        """Determines how to calculate density/velocity and how pdfs are stored:
        True: pdfs are centered around 1 (normal)
        False: pdfs are centered around 0, density is sum(pdfs)+1"""
        return self._compressible

    @property
    def dim(self):
        """Spatial dimension of method"""
        return len(self._stencil[0])

    @property
    def forceModel(self):
        """Force model passed in constructor"""
        return self._forceModel

    @property
    def pdfSymbols(self):
        Q = len(self._stencil)
        return [sp.Symbol("f_%d" % (i,)) for i in range(Q)]

    @property
    def pdfDestinationSymbols(self):
        Q = len(self._stencil)
        return [sp.Symbol("d_%d" % (i,)) for i in range(Q)]

    @property
    def relaxationRates(self):
        """Sequence of len(stencil) relaxation rates (may be symbolic or constant)"""
        return self._relaxationRates

    @property
    def symbolicDensity(self):
        return util.getSymbolicDensity()

    @property
    def symbolicVelocity(self):
        return util.getSymbolicVelocityVector(self.dim)

    def setCollisionDOFs(self, replacementDict):
        """Replace relaxation rate symbols by passing a dictionary from symbol name to new value"""
        substitutions = [(sp.Symbol(key), value) for key, value in replacementDict.items()]
        self._relaxationRates = [rr.subs(substitutions) for rr in self._relaxationRates]

    def getCollisionRule(self):
        raise NotImplemented("This method has to be implemented in subclass")


class MomentRelaxationLatticeModel(LatticeModel):

    def __init__(self, stencil, moments, equilibriumMoments, relaxationRates, compressible, forceModel=None):
        super(MomentRelaxationLatticeModel, self).__init__(stencil, relaxationRates, compressible, forceModel)
        self._moments = moments
        self._equilibriumMoments = sp.Matrix(equilibriumMoments)

    @property
    def momentMatrix(self):
        return m.momentMatrix(self._moments, self.stencil)

    @property
    def weights(self):
        return getWeights(self._stencil)

    def getCollisionRule(self):
        relaxationMatrix = sp.diag(*self._relaxationRates)
        M = self.momentMatrix  # transform from physical to moment space
        pdfVector = sp.Matrix(self.pdfSymbols)
        collisionResult = M.inv() * relaxationMatrix * (self._equilibriumMoments - M * pdfVector)
        if self.forceModel:
            collisionResult += sp.Matrix(self.forceModel(latticeModel=self))
        collisionEqs = [sp.Eq(dst_i, s+t)
                        for s, dst_i, t in zip(self.pdfSymbols, self.pdfDestinationSymbols, collisionResult)]

        # get optimized calculation rules for density and velocity
        rhoSubexprs, rhoEq, uSubexprs, uEqs = getDensityVelocityExpressions(self.stencil, self.pdfSymbols,
                                                                            self.compressible)

        # for some force models the velocity has to be shifted
        if self.forceModel and hasattr(self.forceModel, "equilibriumVelocity"):
            uSymbols = [e.lhs for e in uEqs]
            uRhs = [e.rhs for e in uEqs]
            correctedVel = self.forceModel.equilibriumVelocity(self, uRhs, rhoEq.lhs)
            uEqs = [sp.Eq(u_i, correctedVel_i) for u_i, correctedVel_i in zip(uSymbols, correctedVel)]
        subExpressions = rhoSubexprs + [rhoEq] + uSubexprs + uEqs

        return LbmCollisionRule(collisionEqs, subExpressions, self)