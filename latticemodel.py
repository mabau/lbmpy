import sympy as sp

from pystencils.transformations import fastSubs
from lbmpy.densityVelocityExpressions import getDensityVelocityExpressions
from lbmpy.equilibria import getMaxwellBoltzmannEquilibriumMoments, standardDiscreteEquilibrium, getWeights
import lbmpy.moments as m
import lbmpy.transformations as trafos
import lbmpy.util as util


# ----------------      From standard discrete equilibrium -------------------------------------------------------------


def makeSRT(stencil, order=2, compressible=False, forceModel=None, velocityRelaxationRate=None):
    momentSystem = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(tuple(discreteEquilibrium), mom, stencil) for mom in momentSystem.allMoments]
    relaxationRates = [sp.Symbol('omega')] * len(stencil)

    if velocityRelaxationRate is not None:
        for id in momentSystem.conservedMomentIds[1:]:
            relaxationRates[id] = velocityRelaxationRate
    return MomentRelaxationLatticeModel(stencil, momentSystem.allMoments, equilibriumMoments,
                                        relaxationRates, compressible, forceModel)


def makeTRT(stencil, order=2, compressible=False, forceModel=None, velocityRelaxationRate=None):
    momentSystem = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(tuple(discreteEquilibrium), mom, stencil) for mom in momentSystem.allMoments]

    lambda_e, lambda_o = sp.symbols("lambda_e lambda_o")
    relaxationRates = [lambda_e if m.isEven(moment) else lambda_o for moment in momentSystem.allMoments]
    if velocityRelaxationRate is not None:
        for id in momentSystem.conservedMomentIds[1:]:
            relaxationRates[id] = velocityRelaxationRate
    return MomentRelaxationLatticeModel(stencil, momentSystem.allMoments, equilibriumMoments,
                                        relaxationRates, compressible, forceModel)


def makeMRT(stencil, order=2, compressible=False, forceModel=None, velocityRelaxationRate=None):
    momentSystem = m.getDefaultOrthogonalMoments(stencil)
    discreteEquilibrium = standardDiscreteEquilibrium(stencil, order=order,
                                                      compressible=compressible, c_s_sq=sp.Rational(1, 3))
    equilibriumMoments = [m.discreteMoment(tuple(discreteEquilibrium), mom, stencil) for mom in momentSystem.allMoments]
    if not momentSystem.hasMomentGroups:
        raise NotImplementedError("No moment grouping available for this lattice model")

    relaxationRates = momentSystem.getSymbolicRelaxationRates()
    if velocityRelaxationRate is not None:
        for id in momentSystem.conservedMomentIds[1:]:
            relaxationRates[id] = velocityRelaxationRate
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
    def __init__(self, updateEquations, subExpressions, latticeModel, updateEquationDirections=None):
        self.subexpressions = subExpressions
        self.updateEquations = updateEquations
        if updateEquationDirections is None:
            self.updateEquationDirections = latticeModel.stencil
        else:
            self.updateEquationDirections = updateEquationDirections
        self.latticeModel = latticeModel

    def newWithSubexpressions(self, newUpdateEquations, newSubexpressions, newOrder=None):
        assert len(self.updateEquations) == len(newUpdateEquations)
        ordering = self.updateEquationDirections if newOrder is None else newOrder
        return LbmCollisionRule(newUpdateEquations, self.subexpressions+newSubexpressions, self.latticeModel, ordering)

    def newWithSubstitutions(self, substitutionDict, newOrder=None):
        newSubexpressions = [fastSubs(e, substitutionDict) for e in self.subexpressions]
        newUpdateEquations = [fastSubs(e, substitutionDict) for e in self.updateEquations]
        ordering = self.updateEquationDirections if newOrder is not None else newOrder
        return LbmCollisionRule(newUpdateEquations, newSubexpressions, self.latticeModel, ordering)

    def countNumberOfOperations(self):
        return trafos.countNumberOfOperations(self.subexpressions + self.updateEquations)

    @property
    def equations(self):
        return self.subexpressions + self.updateEquations

    def display(self, printFunction=print):
        """
        Prints subexpressions and update rules of this collision rule
        :param printFunction: function that is used for printing, for IPython notebooks IPython.display can be useful
        """
        printFunction("Subexpressions:")
        for s in self.subexpressions:
            printFunction(s)
        printFunction("Update Rules")
        for s in self.updateEquations:  # [-1:]:
            printFunction(s)

    def displayRepresentative(self, printFunction=print, directions=None):
        """Prints the update rules for C, W, NW and for 3D models TNW
        :param printFunction: function that is used for printing, for IPython notebooks IPython.display can be useful
        :param directions: can be a list of directions, to print only these directions
                                         if None, the default directions are printed
        """
        if directions is None:
            if self.latticeModel.dim == 2:
                directions = [(0, 0), (1, 0), (1, 1)]
            elif self.latticeModel.dim == 3:
                directions = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
            else:
                raise NotImplementedError("Only 2D and 3D models supported")
        indices = [self.updateEquationDirections.index(i) for i in directions]
        for i in indices:
            printFunction(self.updateEquations[i])


class LatticeModel:
    def __init__(self, stencil, compressible, forceModel=None):
        self._stencil = stencil
        self._compressible = compressible
        self._forceModel = forceModel

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
    def symbolicDensity(self):
        return util.getSymbolicDensity()

    @property
    def symbolicVelocity(self):
        return util.getSymbolicVelocityVector(self.dim)

    def getCollisionRule(self):
        raise NotImplemented("This method has to be implemented in subclass")


class MomentRelaxationLatticeModel(LatticeModel):

    def __init__(self, stencil, moments, equilibriumMoments, relaxationRates, compressible, forceModel=None):
        super(MomentRelaxationLatticeModel, self).__init__(stencil, compressible, forceModel)
        self._moments = moments
        self._equilibriumMoments = sp.Matrix(equilibriumMoments)
        self._relaxationRates = relaxationRates

    @property
    def momentMatrix(self):
        return m.momentMatrix(self._moments, self.stencil)

    @property
    def weights(self):
        return getWeights(self._stencil)

    @property
    def relaxationRates(self):
        """Sequence of len(stencil) relaxation rates (may be symbolic or constant)"""
        return self._relaxationRates

    @property
    def allRelaxationRatesFixed(self):
        symbols = [rt for rt in self._relaxationRates if isinstance(rt, sp.Symbol)]
        return len(symbols) == 0

    def setCollisionDOFs(self, replacementDict):
        """Replace relaxation rate symbols by passing a dictionary from symbol name to new value"""
        substitutions = {sp.Symbol(key): value for key, value in replacementDict.items()}
        self._relaxationRates = [fastSubs(rr, substitutions) for rr in self._relaxationRates]

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

