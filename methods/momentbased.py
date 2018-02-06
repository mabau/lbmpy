import sympy as sp
from collections import OrderedDict

from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.moments import MOMENT_SYMBOLS, momentMatrix
from pystencils.sympyextensions import replaceAdditive


class MomentBasedLbMethod(AbstractLbMethod):
    def __init__(self, stencil, momentToRelaxationInfoDict, conservedQuantityComputation=None, forceModel=None):
        """
        Moment based LBM is a class to represent the single (SRT), two (TRT) and multi relaxation time (MRT) methods.
        These methods work by transforming the pdfs into moment space using a linear transformation. In the moment
        space each component (moment) is relaxed to an equilibrium moment by a certain relaxation rate. These
        equilibrium moments can e.g. be determined by taking the equilibrium moments of the continuous Maxwellian.

        :param stencil: see :func:`lbmpy.stencils.getStencil`
        :param momentToRelaxationInfoDict: a dictionary mapping moments in either tuple or polynomial formulation
                                           to a RelaxationInfo, which consists of the corresponding equilibrium moment
                                           and a relaxation rate
        :param conservedQuantityComputation: instance of :class:`lbmpy.methods.AbstractConservedQuantityComputation`.
                                             This determines how conserved quantities are computed, and defines
                                             the symbols used in the equilibrium moments like e.g. density and velocity
        :param forceModel: force model instance, or None if no forcing terms are required
        """
        assert isinstance(conservedQuantityComputation, AbstractConservedQuantityComputation)
        super(MomentBasedLbMethod, self).__init__(stencil)

        self._forceModel = forceModel
        self._momentToRelaxationInfoDict = OrderedDict(momentToRelaxationInfoDict.items())
        self._conservedQuantityComputation = conservedQuantityComputation
        self._weights = None

    @property
    def forceModel(self):
        return self._forceModel

    @property
    def relaxationInfoDict(self):
        return self._momentToRelaxationInfoDict

    @property
    def conservedQuantityComputation(self):
        return self._conservedQuantityComputation

    @property
    def moments(self):
        return tuple(self._momentToRelaxationInfoDict.keys())

    @property
    def momentEquilibriumValues(self):
        return tuple([e.equilibriumValue for e in self._momentToRelaxationInfoDict.values()])

    @property
    def relaxationRates(self):
        return tuple([e.relaxationRate for e in self._momentToRelaxationInfoDict.values()])

    @property
    def zerothOrderEquilibriumMomentSymbol(self, ):
        return self._conservedQuantityComputation.zerothOrderMomentSymbol

    @property
    def firstOrderEquilibriumMomentSymbols(self, ):
        return self._conservedQuantityComputation.firstOrderMomentSymbols

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self._computeWeights()
        return self._weights

    def overrideWeights(self, weights):
        assert len(weights) == len(self.stencil)
        self._weights = weights

    def getEquilibrium(self, conservedQuantityEquations=None, includeForceTerms=False):
        D = sp.eye(len(self.relaxationRates))
        return self._getCollisionRuleWithRelaxationMatrix(D, conservedQuantityEquations=conservedQuantityEquations,
                                                          includeForceTerms=includeForceTerms)

    def getEquilibriumTerms(self):
        equilibrium = self.getEquilibrium()
        return sp.Matrix([eq.rhs for eq in equilibrium.mainEquations])

    def getCollisionRule(self, conservedQuantityEquations=None):
        D = sp.diag(*self.relaxationRates)
        relaxationRateSubExpressions, D = self._generateRelaxationMatrix(D)
        eqColl = self._getCollisionRuleWithRelaxationMatrix(D, relaxationRateSubExpressions,
                                                            True, conservedQuantityEquations)
        return eqColl

    def setZerothMomentRelaxationRate(self, relaxationRate):
        e = sp.Rational(1, 1)
        prevEntry = self._momentToRelaxationInfoDict[e]
        newEntry = RelaxationInfo(prevEntry[0], relaxationRate)
        self._momentToRelaxationInfoDict[e] = newEntry

    def setFirstMomentRelaxationRate(self, relaxationRate):
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._momentToRelaxationInfoDict, "First moments are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            prevEntry = self._momentToRelaxationInfoDict[e]
            newEntry = RelaxationInfo(prevEntry[0], relaxationRate)
            self._momentToRelaxationInfoDict[e] = newEntry

    @property
    def collisionMatrix(self):
        M = self.momentMatrix
        D = sp.diag(*self.relaxationRates)
        return M.inv() * D * M

    @property
    def inverseCollisionMatrix(self):
        M = self.momentMatrix
        Dinv = sp.diag(*[1/e for e in self.relaxationRates])
        return M.inv() * Dinv * M

    @property
    def momentMatrix(self):
        return momentMatrix(self.moments, self.stencil)

    def __getstate__(self):
        # Workaround for a bug in joblib
        self._momentToRelaxationInfoDictToPickle = [i for i in self._momentToRelaxationInfoDict.items()]
        return self.__dict__

    def _repr_html_(self):
        table = """
        <table style="border:none; width: 100%">
            <tr {nb}>
                <th {nb} >Moment</th>
                <th {nb} >Eq. Value </th>
                <th {nb} >Relaxation Rate</th>
            </tr>
            {content}
        </table>
        """
        content = ""
        for moment, (eqValue, rr) in self._momentToRelaxationInfoDict.items():
            vals = {
                'rr': sp.latex(rr),
                'moment': sp.latex(moment),
                'eqValue': sp.latex(eqValue),
                'nb': 'style="border:none"',
            }
            content += """<tr {nb}>
                            <td {nb}>${moment}$</td>
                            <td {nb}>${eqValue}$</td>
                            <td {nb}>${rr}$</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')

    def _computeWeights(self):
        replacements = self._conservedQuantityComputation.defaultValues
        eqColl = self.getEquilibrium(includeForceTerms=False)
        eqColl = eqColl.copyWithSubstitutionsApplied(replacements, substituteOnLhs=False).insertSubexpressions()

        newMainEqs = [sp.Eq(e.lhs,
                            replaceAdditive(e.rhs, 1, sum(self.preCollisionPdfSymbols), requiredMatchReplacement=1.0))
                      for e in eqColl.mainEquations]
        eqColl = eqColl.copy(newMainEqs)

        weights = []
        for eq in eqColl.mainEquations:
            value = eq.rhs.expand()
            assert len(value.atoms(sp.Symbol)) == 0, "Failed to compute weights " + str(value)
            weights.append(value)
        return weights

    def _getCollisionRuleWithRelaxationMatrix(self, D, additionalSubexpressions=(), includeForceTerms=True,
                                              conservedQuantityEquations=None):
        f = sp.Matrix(self.preCollisionPdfSymbols)
        M = self.momentMatrix
        m_eq = sp.Matrix(self.momentEquilibriumValues)

        collisionRule = f + M.inv() * D * (m_eq - M * f)
        collisionEqs = [sp.Eq(lhs, rhs) for lhs, rhs in zip(self.postCollisionPdfSymbols, collisionRule)]

        if conservedQuantityEquations is None:
            conservedQuantityEquations = self._conservedQuantityComputation.equilibriumInputEquationsFromPdfs(f)

        simplificationHints = conservedQuantityEquations.simplificationHints.copy()
        simplificationHints.update(self._conservedQuantityComputation.definedSymbols())
        simplificationHints['relaxationRates'] = [D[i, i] for i in range(D.rows)]

        allSubexpressions = list(additionalSubexpressions) + conservedQuantityEquations.allEquations

        if self._forceModel is not None and includeForceTerms:
            forceModelTerms = self._forceModel(self)
            forceTermSymbols = sp.symbols("forceTerm_:%d" % (len(forceModelTerms,)))
            forceSubexpressions = [sp.Eq(sym, forceModelTerm)
                                   for sym, forceModelTerm in zip(forceTermSymbols, forceModelTerms)]
            allSubexpressions += forceSubexpressions
            collisionEqs = [sp.Eq(eq.lhs, eq.rhs + forceTermSymbol)
                            for eq, forceTermSymbol in zip(collisionEqs, forceTermSymbols)]
            simplificationHints['forceTerms'] = forceTermSymbols

        return LbmCollisionRule(self, collisionEqs, allSubexpressions,
                                simplificationHints)

    @staticmethod
    def _generateRelaxationMatrix(relaxationMatrix):
        """
        For SRT and TRT the equations can be easier simplified if the relaxation times are symbols, not numbers.
        This function replaces the numbers in the relaxation matrix with symbols in this case, and returns also
         the subexpressions, that assign the number to the newly introduced symbol
        """
        rr = [relaxationMatrix[i, i] for i in range(relaxationMatrix.rows)]
        uniqueRelaxationRates = set(rr)
        if len(uniqueRelaxationRates) <= 2:
            # special handling for SRT and TRT
            subexpressions = {}
            for rt in uniqueRelaxationRates:
                rt = sp.sympify(rt)
                if not isinstance(rt, sp.Symbol):
                    rtSymbol = sp.Symbol("rr_%d" % (len(subexpressions),))
                    subexpressions[rt] = rtSymbol

            newRR = [subexpressions[sp.sympify(e)] if sp.sympify(e) in subexpressions else e
                     for e in rr]
            substitutions = [sp.Eq(e[1], e[0]) for e in subexpressions.items()]
            return substitutions, sp.diag(*newRR)
        else:
            return [], relaxationMatrix




