import sympy as sp
from collections import OrderedDict
from pystencils import Assignment
from pystencils.sympyextensions import fast_subs, subs_additive
from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.moments import MOMENT_SYMBOLS, extractMonomials, momentMatrix, monomialToPolynomialTransformationMatrix
from lbmpy.cumulants import cumulantAsFunctionOfRawMoments, rawMomentAsFunctionOfCumulants


class CumulantBasedLbMethod(AbstractLbMethod):

    def __init__(self, stencil, cumulantToRelaxationInfoDict, conservedQuantityComputation, forceModel=None):
        assert isinstance(conservedQuantityComputation, AbstractConservedQuantityComputation)
        super(CumulantBasedLbMethod, self).__init__(stencil)

        self._forceModel = forceModel
        self._cumulantToRelaxationInfoDict = OrderedDict(cumulantToRelaxationInfoDict.items())
        self._conservedQuantityComputation = conservedQuantityComputation
        self._weights = None

    @property
    def forceModel(self):
        return self._forceModel

    @property
    def relaxationInfoDict(self):
        return self._cumulantToRelaxationInfoDict

    @property
    def zerothOrderEquilibriumMomentSymbol(self, ):
        return self._conservedQuantityComputation.zerothOrderMomentSymbol

    @property
    def firstOrderEquilibriumMomentSymbols(self, ):
        return self._conservedQuantityComputation.firstOrderMomentSymbols

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

    def overrideWeights(self, weights):
        assert len(weights) == len(self.stencil)
        self._weights = weights

    def _repr_html_(self):
        table = """
        <table style="border:none; width: 100%">
            <tr {nb}>
                <th {nb} >Cumulant</th>
                <th {nb} >Eq. Value </th>
                <th {nb} >Relaxation Rate</th>
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
        return self._getCollisionRuleWithRelaxationMatrix(D, conservedQuantityEquations, False, False, False, False)

    def getEquilibriumTerms(self):
        equilibrium = self.getEquilibrium()
        return sp.Matrix([eq.rhs for eq in equilibrium.main_assignments])

    def getCollisionRule(self, conservedQuantityEquations=None, momentSubexpressions=False,
                         preCollisionSubexpressions=True, postCollisionSubexpressions=False):
        return self._getCollisionRuleWithRelaxationMatrix(sp.diag(*self.relaxationRates), conservedQuantityEquations,
                                                          momentSubexpressions, preCollisionSubexpressions,
                                                          postCollisionSubexpressions)

    def _computeWeights(self):
        replacements = self._conservedQuantityComputation.defaultValues
        eq = self.getEquilibrium()
        eqColl = eq.new_with_substitutions(replacements, substitute_on_lhs=False).new_without_subexpressions()
        newMainEqs = [Assignment(e.lhs,
                                 subs_additive(e.rhs, 1, sum(self.preCollisionPdfSymbols),
                                               required_match_replacement=1.0))
                      for e in eqColl.main_assignments]
        eqColl = eqColl.copy(newMainEqs)

        weights = []
        for eq in eqColl.main_assignments:
            value = eq.rhs.expand()
            assert len(value.atoms(sp.Symbol)) == 0, "Failed to compute weights"
            weights.append(value)
        return weights

    def _getCollisionRuleWithRelaxationMatrix(self, relaxationMatrix, conserved_quantity_equations=None,
                                              momentSubexpressions=False, preCollisionSubexpressions=True,
                                              postCollisionSubexpressions=False, includeForceTerms=True):
        def tuple_to_symbol(exp, prefix):
            dim = len(exp)
            format_string = prefix + "_" + "_".join(["%d"]*dim)
            return sp.Symbol(format_string % exp)

        def substitute_conserved_quantities(expressions, cqe):
            cqe = cqe.new_without_subexpressions()
            substitution_dict = {eq.rhs: eq.lhs for eq in cqe.main_assignments}
            density = cqe.main_assignments[0].lhs
            substitution_dict.update({density * eq.rhs: density * eq.lhs for eq in cqe.main_assignments[1:]})
            return [fast_subs(e, substitution_dict) for e in expressions]

        f = self.preCollisionPdfSymbols
        if conserved_quantity_equations is None:
            conserved_quantity_equations = self._conservedQuantityComputation.equilibriumInputEquationsFromPdfs(f)

        subexpressions = conserved_quantity_equations.all_assignments

        # 1) Determine monomial indices, and arrange them such that the zeroth and first order indices come first
        indices = list(extractMonomials(self.cumulants, dim=len(self.stencil[0])))
        zeroth_moment_exponent = (0,) * self.dim
        first_moment_exponents = [tuple([1 if i == j else 0 for i in range(self.dim)]) for j in range(self.dim)]
        lower_order_indices = [zeroth_moment_exponent] + first_moment_exponents
        num_lower_order_indices = len(lower_order_indices)
        assert all(e in indices for e in lower_order_indices), \
            "Cumulant system does not contain relaxation rules for zeroth and first order cumulants"
        higher_order_indices = [e for e in indices if e not in lower_order_indices]
        indices = lower_order_indices + higher_order_indices  # reorder

        # 2) Transform pdfs to moments
        moment_transformation_matrix = momentMatrix(indices, self.stencil)
        moments = moment_transformation_matrix * sp.Matrix(f)
        moments = substitute_conserved_quantities(moments, conserved_quantity_equations)
        if momentSubexpressions:
            symbols = [tuple_to_symbol(t, "m") for t in higher_order_indices]
            subexpressions += [Assignment(sym, moment) for sym, moment in zip(symbols, moments[num_lower_order_indices:])]
            moments = moments[:num_lower_order_indices] + symbols

        # 3) Transform moments to monomial cumulants
        moments_dict = {idx: m for idx, m in zip(indices, moments)}
        monomial_cumulants = [cumulantAsFunctionOfRawMoments(idx, moments_dict) for idx in indices]

        if preCollisionSubexpressions:
            symbols = [tuple_to_symbol(t, "preC") for t in higher_order_indices]
            subexpressions += [Assignment(sym, c)
                               for sym, c in zip(symbols, monomial_cumulants[num_lower_order_indices:])]
            monomial_cumulants = monomial_cumulants[:num_lower_order_indices] + symbols

        # 4) Transform monomial to polynomial cumulants which are then relaxed and transformed back
        mon_to_poly = monomialToPolynomialTransformationMatrix(indices, self.cumulants)
        poly_values = mon_to_poly * sp.Matrix(monomial_cumulants)
        eq_values = sp.Matrix(self.cumulantEquilibriumValues)
        collided_poly_values = poly_values + relaxationMatrix * (eq_values - poly_values)  # collision
        relaxed_monomial_cumulants = mon_to_poly.inv() * collided_poly_values

        if postCollisionSubexpressions:
            symbols = [tuple_to_symbol(t, "postC") for t in higher_order_indices]
            subexpressions += [Assignment(sym, c)
                               for sym, c in zip(symbols, relaxed_monomial_cumulants[num_lower_order_indices:])]
            relaxed_monomial_cumulants = relaxed_monomial_cumulants[:num_lower_order_indices] + symbols

        # 5) Transform post-collision cumulant back to moments and from there to pdfs
        cumulant_dict = {idx: value for idx, value in zip(indices, relaxed_monomial_cumulants)}
        collided_moments = [rawMomentAsFunctionOfCumulants(idx, cumulant_dict) for idx in indices]
        result = moment_transformation_matrix.inv() * sp.Matrix(collided_moments)
        main_assignments = [Assignment(sym, val) for sym, val in zip(self.postCollisionPdfSymbols, result)]

        # 6) Add forcing terms
        if self._forceModel is not None and includeForceTerms:
            force_model_terms = self._forceModel(self)
            force_term_symbols = sp.symbols("forceTerm_:%d" % (len(force_model_terms,)))
            force_subexpressions = [Assignment(sym, forceModelTerm)
                                    for sym, forceModelTerm in zip(force_term_symbols, force_model_terms)]
            subexpressions += force_subexpressions
            main_assignments = [Assignment(eq.lhs, eq.rhs + forceTermSymbol)
                                for eq, forceTermSymbol in zip(main_assignments, force_term_symbols)]

        sh = {'relaxationRates': list(self.relaxationRates)}
        return LbmCollisionRule(self, main_assignments, subexpressions, simplification_hints=sh)





