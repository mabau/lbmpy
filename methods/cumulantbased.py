import sympy as sp
from collections import OrderedDict
from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.moments import MOMENT_SYMBOLS, extractMonomials, momentMatrix, monomialToPolynomialTransformationMatrix
from lbmpy.cumulants import cumulantAsFunctionOfRawMoments, rawMomentAsFunctionOfCumulants
from pystencils.sympyextensions import fastSubs, replaceAdditive


class CumulantBasedLbMethod(AbstractLbMethod):

    def __init__(self, stencil, cumulantToRelaxationInfoDict, conservedQuantityComputation, forceModel=None):
        assert isinstance(conservedQuantityComputation, AbstractConservedQuantityComputation)
        super(CumulantBasedLbMethod, self).__init__(stencil)

        self._forceModel = forceModel
        self._cumulantToRelaxationInfoDict = OrderedDict(cumulantToRelaxationInfoDict.items())
        self._conservedQuantityComputation = conservedQuantityComputation
        self._weights = None

    @property
    def cumulantToRelaxationInfoDict(self):
        return self._cumulantToRelaxationInfoDict

    def setFirstMomentRelaxationRate(self, relaxationRate):
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._cumulantToRelaxationInfoDict, "First cumulants are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            prevEntry = self._cumulantToRelaxationInfoDict[e]
            newEntry = RelaxationInfo(prevEntry[0], relaxationRate)
            self._cumulantToRelaxationInfoDict[e] = newEntry

    @property
    def cumulants(self):
        return tuple(self._cumulantToRelaxationInfoDict.keys())

    @property
    def cumulantEquilibriumValues(self):
        return tuple([e.equilibriumValue for e in self._cumulantToRelaxationInfoDict.values()])

    @property
    def relaxationRates(self):
        return tuple([e.relaxationRate for e in self._cumulantToRelaxationInfoDict.values()])

    @property
    def conservedQuantityComputation(self):
        return self._conservedQuantityComputation

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self._computeWeights()
        return self._weights

    def _repr_html_(self):
        table = """
        <table style="border:none; width: 100%">
            <tr {nb}>
                <th {nb} >Cumulant</th>
                <th {nb} >Eq. Value </th>
                <th {nb} >Relaxation Time</th>
            </tr>
            {content}
        </table>
        """
        content = ""
        for cumulant, (eqValue, rr) in self._cumulantToRelaxationInfoDict.items():
            vals = {
                'rr': sp.latex(rr),
                'cumulant': sp.latex(cumulant),
                'eqValue': sp.latex(eqValue),
                'nb': 'style="border:none"',
            }
            content += """<tr {nb}>
                            <td {nb}>${cumulant}$</td>
                            <td {nb}>${eqValue}$</td>
                            <td {nb}>${rr}$</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')

    def getEquilibrium(self, conservedQuantityEquations=None):
        D = sp.eye(len(self.relaxationRates))
        return self._getCollisionRuleWithRelaxationMatrix(D, conservedQuantityEquations, False, False, False)

    def getCollisionRule(self, conservedQuantityEquations=None, momentSubexpressions=False,
                         preCollisionSubexpressions=True, postCollisionSubexpressions=False):
        return self._getCollisionRuleWithRelaxationMatrix(sp.diag(*self.relaxationRates), conservedQuantityEquations,
                                                          momentSubexpressions, preCollisionSubexpressions,
                                                          postCollisionSubexpressions)

    def _computeWeights(self):
        replacements = self._conservedQuantityComputation.defaultValues
        eqColl = self.getEquilibrium().copyWithSubstitutionsApplied(replacements).insertSubexpressions()
        newMainEqs = [sp.Eq(e.lhs,
                            replaceAdditive(e.rhs, 1, sum(self.preCollisionPdfSymbols), requiredMatchReplacement=1.0))
                      for e in eqColl.mainEquations]
        eqColl = eqColl.copy(newMainEqs)

        weights = []
        for eq in eqColl.mainEquations:
            value = eq.rhs.expand()
            assert len(value.atoms(sp.Symbol)) == 0, "Failed to compute weights"
            weights.append(value)
        return weights

    def _getCollisionRuleWithRelaxationMatrix(self, relaxationMatrix, conservedQuantityEquations=None,
                                              momentSubexpressions=False, preCollisionSubexpressions=True,
                                              postCollisionSubexpressions=False):
        def tupleToSymbol(exp, prefix):
            dim = len(exp)
            formatString = prefix + "_" + "_".join(["%d"]*dim)
            return sp.Symbol(formatString % exp)

        def substituteConservedQuantities(expressions, cqe):
            cqe = cqe.insertSubexpressions()
            substitutionDict = {eq.rhs: eq.lhs for eq in cqe.mainEquations}
            density = cqe.mainEquations[0].lhs
            substitutionDict.update({density * eq.rhs: density * eq.lhs for eq in cqe.mainEquations[1:]})
            return [fastSubs(e, substitutionDict) for e in expressions]

        f = self.preCollisionPdfSymbols
        if conservedQuantityEquations is None:
            conservedQuantityEquations = self._conservedQuantityComputation.equilibriumInputEquationsFromPdfs(f)

        subexpressions = conservedQuantityEquations.allEquations

        # 1) Determine monomial indices, and arange them such that the zeroth and first order indices come first
        indices = list(extractMonomials(self.cumulants, dim=len(self.stencil[0])))
        zerothMomentExponent = (0,) * self.dim
        firstMomentExponents = [tuple([1 if i == j else 0 for i in range(self.dim)]) for j in range(self.dim)]
        lowerOrderIndices = [zerothMomentExponent] + firstMomentExponents
        numLowerOrderIndices = len(lowerOrderIndices)
        assert all(e in indices for e in lowerOrderIndices), \
            "Cumulant system does not contain relaxation rules for zeroth and first order cumulants"
        higherOrderIndices = [e for e in indices if e not in lowerOrderIndices]
        indices = lowerOrderIndices + higherOrderIndices  # reorder

        # 2) Transform pdfs to moments
        momentTransformationMatrix = momentMatrix(indices, self.stencil)
        moments = momentTransformationMatrix * sp.Matrix(f)
        moments = substituteConservedQuantities(moments, conservedQuantityEquations)
        if momentSubexpressions:
            symbols = [tupleToSymbol(t, "m") for t in higherOrderIndices]
            subexpressions += [sp.Eq(sym, moment) for sym, moment in zip(symbols, moments[numLowerOrderIndices:])]
            moments = moments[:numLowerOrderIndices] + symbols

        # 3) Transform moments to monomial cumulants
        momentsDict = {idx: m for idx, m in zip(indices, moments)}
        monomialCumulants = [cumulantAsFunctionOfRawMoments(idx, momentsDict) for idx in indices]

        if preCollisionSubexpressions:
            symbols = [tupleToSymbol(t, "preC") for t in higherOrderIndices]
            subexpressions += [sp.Eq(sym, c)
                               for sym, c in zip(symbols, monomialCumulants[numLowerOrderIndices:])]
            monomialCumulants = monomialCumulants[:numLowerOrderIndices] + symbols

        # 4) Transform monomial to polynomial cumulants which are then relaxed and transformed back
        monToPoly = monomialToPolynomialTransformationMatrix(indices, self.cumulants)
        polyValues = monToPoly * sp.Matrix(monomialCumulants)
        eqValues = sp.Matrix(self.cumulantEquilibriumValues)
        collidedPolyValues = polyValues + relaxationMatrix * (eqValues - polyValues)  # collision
        relaxedMonomialCumulants = monToPoly.inv() * collidedPolyValues

        if postCollisionSubexpressions:
            symbols = [tupleToSymbol(t, "postC") for t in higherOrderIndices]
            subexpressions += [sp.Eq(sym, c)
                               for sym, c in zip(symbols, relaxedMonomialCumulants[numLowerOrderIndices:])]
            relaxedMonomialCumulants = relaxedMonomialCumulants[:numLowerOrderIndices] + symbols

        # 5) Transform post-collision cumulant back to moments and from there to pdfs
        cumulantDict = {idx: value for idx, value in zip(indices, relaxedMonomialCumulants)}
        collidedMoments = [rawMomentAsFunctionOfCumulants(idx, cumulantDict) for idx in indices]
        result = momentTransformationMatrix.inv() * sp.Matrix(collidedMoments)
        mainEquations = [sp.Eq(sym, val) for sym, val in zip(self.postCollisionPdfSymbols, result)]

        return LbmCollisionRule(self, mainEquations, subexpressions, simplificationHints={})





