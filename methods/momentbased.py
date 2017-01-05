import sympy as sp
import itertools
import collections
from collections import namedtuple
from lbmpy.methods.abstractlbmmethod import AbstractLbmMethod
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.moments import MOMENT_SYMBOLS, momentMatrix, exponentToPolynomialRepresentation, isShearMoment
from pystencils.equationcollection import EquationCollection

"""
Ways to create method:
    - moment (polynomial or tuple) mapped to relaxation rate
    - moment matrix & relaxation vector
    - createSRT, createTRT, createMRT
"""

RelaxationInfo = namedtuple('Relaxationinfo', ['equilibriumValue', 'relaxationRate'])


class MomentBasedLbmMethod(AbstractLbmMethod):

    def __init__(self, stencil, momentToRelaxationInfoDict, conservedQuantityComputation, forceModel=None):
        """

        :param stencil:
        :param momentToRelaxationInfoDict:
        :param conservedQuantityComputation: instance of :class:`lbmpy.methods.AbstractConservedQuantityComputation`
        :param forceModel:
        """
        super(MomentBasedLbmMethod, self).__init__(stencil)

        assert isinstance(conservedQuantityComputation, AbstractConservedQuantityComputation)

        moments = []
        relaxationRates = []
        equilibriumMoments = []
        for moment, relaxInfo in momentToRelaxationInfoDict.items():
            moments.append(moment)
            relaxationRates.append(relaxInfo.relaxationRate)
            equilibriumMoments.append(relaxInfo.equilibriumValue)

        self._forceModel = forceModel
        self._momentToRelaxationInfoDict = momentToRelaxationInfoDict
        self._momentMatrix = momentMatrix(moments, self.stencil)
        self._relaxationRates = sp.Matrix(relaxationRates)
        self._equilibriumMoments = sp.Matrix(equilibriumMoments)
        self._conservedQuantityComputation = conservedQuantityComputation

        symbolsInEquilibriumMoments = self._equilibriumMoments.atoms(sp.Symbol)
        conservedQuantities = set()
        for v in self._conservedQuantityComputation.definedSymbols().values():
            if isinstance(v, collections.Sequence):
                conservedQuantities.update(v)
            else:
                conservedQuantities.add(v)
        undefinedEquilibriumSymbols = symbolsInEquilibriumMoments - conservedQuantities
        assert len(undefinedEquilibriumSymbols) == 0, "Undefined symbol(s) in equilibrium moment: %s" % \
                                                      (undefinedEquilibriumSymbols, )

        self._weights = self._computeWeights()

    @property
    def zerothOrderEquilibriumMomentSymbol(self,):
        return self._conservedQuantityComputation.definedSymbols(order=0)[1]

    @property
    def firstOrderEquilibriumMomentSymbols(self,):
        return self._conservedQuantityComputation.definedSymbols(order=1)[1]

    @property
    def weights(self):
        return self._weights

    def _computeWeights(self):
        replacements = self._conservedQuantityComputation.defaultValues
        eqColl = self.getEquilibrium().newWithSubstitutionsApplied(replacements).insertSubexpressions()
        weights = []
        for eq in eqColl.mainEquations:
            value = eq.rhs.expand()
            assert len(value.atoms(sp.Symbol)) == 0, "Failed to compute weights"
            weights.append(value)
        return weights

    def getShearRelaxationRate(self):
        """
        Assumes that all shear moments are relaxed with same rate - returns this rate
        Shear moments in 3D are: x*y, x*z and y*z - in 2D its only x*y
        The shear relaxation rate determines the viscosity in hydrodynamic LBM schemes
        """
        relaxationRates = set()
        for moment, relaxInfo in self._momentToRelaxationInfoDict.items():
            if isShearMoment(moment):
                relaxationRates.add(relaxInfo.relaxationRate)
        if len(relaxationRates) == 1:
            return relaxationRates.pop()
        else:
            if len(relaxationRates) > 1:
                raise ValueError("Shear moments are relaxed with different relaxation times: %s" % (relaxationRates,))
            else:
                raise NotImplementedError("Shear moments seem to be not relaxed separately - "
                                          "Can not determine their relaxation rate automatically")

    def getEquilibrium(self):
        D = sp.eye(len(self._relaxationRates))
        return self._getCollisionRuleWithRelaxationMatrix(D)

    def getCollisionRule(self):
        D = sp.diag(*self._relaxationRates)
        eqColl = self._getCollisionRuleWithRelaxationMatrix(D)
        if self._forceModel is not None:
            forceModelTerms = self._forceModel(self)
            newEqs = [sp.Eq(eq.lhs, eq.rhs + fmt) for eq, fmt in zip(eqColl.mainEquations, forceModelTerms)]
            eqColl = eqColl.newWithAdditionalSubexpressions(newEqs, [])
        return eqColl

    @property
    def conservedQuantityComputation(self):
        return self._conservedQuantityComputation

    def _getCollisionRuleWithRelaxationMatrix(self, D):
        f = sp.Matrix(self.preCollisionPdfSymbols)
        M = self._momentMatrix
        m_eq = self._equilibriumMoments

        collisionRule = f + M.inv() * D * (m_eq - M * f)
        collisionEqs = [sp.Eq(lhs, rhs) for lhs, rhs in zip(self.postCollisionPdfSymbols, collisionRule)]

        eqValueEqs = self._conservedQuantityComputation.equilibriumInputEquationsFromPdfs(f)
        simplificationHints = eqValueEqs.simplificationHints
        # TODO add own simplification hints here
        return EquationCollection(collisionEqs, eqValueEqs.subexpressions + eqValueEqs.mainEquations,
                                  simplificationHints)


def createByMatchingMoments(stencil, moments, ):
    pass


def createSRT(stencil, relaxationRate):
    pass


def createTRT(stencil, relaxationRateEvenMoments, relaxationRateOddMoments):
    pass


def createMRT():
    pass


if __name__ == '__main__':
    from lbmpy.stencils import getStencil
    from lbmpy.methods import *
    from lbmpy.moments import *
    from lbmpy.maxwellian_equilibrium import *
    from lbmpy.forcemodels import *
    import sympy as sp

    stencil = getStencil('D2Q9')
    pdfs = sp.symbols("f_:9")
    force = sp.symbols("F_:2")
    forceModel = Luo(force)
    compressible = True

    dim = len(stencil[0])

    c = DensityVelocityComputation(stencil, compressible, forceModel)

    moments = momentsUpToComponentOrder(2, dim=dim)
    eqMoments = getMomentsOfDiscreteMaxwellianEquilibrium(stencil, moments, c_s_sq=sp.Rational(1, 3),
                                                          compressible=compressible)
    omega = sp.Symbol("omega")
    relaxInfoDict = {m: RelaxationInfo(eqMoment, omega) for m, eqMoment in zip(moments, eqMoments)}

    m = MomentBasedLbmMethod(stencil, relaxInfoDict, c, forceModel)
    m.getCollisionRule()
