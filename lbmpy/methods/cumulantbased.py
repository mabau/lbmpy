from collections import OrderedDict

import sympy as sp

from lbmpy.cumulants import cumulant_as_function_of_raw_moments, raw_moment_as_function_of_cumulants
from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.moments import (
    MOMENT_SYMBOLS, extract_monomials, moment_matrix, monomial_to_polynomial_transformation_matrix)
from pystencils import Assignment
from pystencils.sympyextensions import fast_subs, subs_additive


class CumulantBasedLbMethod(AbstractLbMethod):

    def __init__(self, stencil, cumulant_to_relaxation_info_dict, conserved_quantity_computation, force_model=None):
        assert isinstance(conserved_quantity_computation, AbstractConservedQuantityComputation)
        super(CumulantBasedLbMethod, self).__init__(stencil)

        self._force_model = force_model
        self._cumulant_to_relaxation_info_dict = OrderedDict(cumulant_to_relaxation_info_dict.items())
        self._conserved_quantity_computation = conserved_quantity_computation
        self._weights = None

    @property
    def force_model(self):
        return self._force_model

    @property
    def relaxation_info_dict(self):
        return self._cumulant_to_relaxation_info_dict

    @property
    def zeroth_order_equilibrium_moment_symbol(self, ):
        return self._conserved_quantity_computation.zeroth_order_moment_symbol

    @property
    def first_order_equilibrium_moment_symbols(self, ):
        return self._conserved_quantity_computation.first_order_moment_symbols

    def set_first_moment_relaxation_rate(self, relaxation_rate):
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._cumulant_to_relaxation_info_dict, \
                "First cumulants are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            prev_entry = self._cumulant_to_relaxation_info_dict[e]
            new_entry = RelaxationInfo(prev_entry[0], relaxation_rate)
            self._cumulant_to_relaxation_info_dict[e] = new_entry

    @property
    def cumulants(self):
        return tuple(self._cumulant_to_relaxation_info_dict.keys())

    @property
    def cumulant_equilibrium_values(self):
        return tuple([e.equilibrium_value for e in self._cumulant_to_relaxation_info_dict.values()])

    @property
    def relaxation_rates(self):
        return tuple([e.relaxation_rate for e in self._cumulant_to_relaxation_info_dict.values()])

    @property
    def conserved_quantity_computation(self):
        return self._conserved_quantity_computation

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self._compute_weights()
        return self._weights

    def override_weights(self, weights):
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
        for cumulant, (eq_value, rr) in self._cumulant_to_relaxation_info_dict.items():
            vals = {
                'rr': sp.latex(rr),
                'cumulant': sp.latex(cumulant),
                'eq_value': sp.latex(eq_value),
                'nb': 'style="border:none"',
            }
            content += """<tr {nb}>
                            <td {nb}>${cumulant}$</td>
                            <td {nb}>${eq_value}$</td>
                            <td {nb}>${rr}$</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')

    def get_equilibrium(self, conserved_quantity_equations=None):
        d = sp.eye(len(self.relaxation_rates))
        return self._get_collision_rule_with_relaxation_matrix(d, conserved_quantity_equations,
                                                               False, False, False, False)

    def get_equilibrium_terms(self):
        equilibrium = self.get_equilibrium()
        return sp.Matrix([eq.rhs for eq in equilibrium.main_assignments])

    def get_collision_rule(self, conserved_quantity_equations=None, moment_subexpressions=False,
                           pre_collision_subexpressions=True, post_collision_subexpressions=False,
                           keep_rrs_symbolic=None):
        return self._get_collision_rule_with_relaxation_matrix(sp.diag(*self.relaxation_rates),
                                                               conserved_quantity_equations,
                                                               moment_subexpressions, pre_collision_subexpressions,
                                                               post_collision_subexpressions)

    def _compute_weights(self):
        replacements = self._conserved_quantity_computation.default_values
        eq = self.get_equilibrium()
        ac = eq.new_with_substitutions(replacements, substitute_on_lhs=False).new_without_subexpressions()
        new_main_eqs = [Assignment(e.lhs,
                                   subs_additive(e.rhs, sp.sympify(1), sum(self.pre_collision_pdf_symbols),
                                                 required_match_replacement=1.0))
                        for e in ac.main_assignments]
        ac = ac.copy(new_main_eqs)

        weights = []
        for eq in ac.main_assignments:
            value = eq.rhs.expand()
            assert len(value.atoms(sp.Symbol)) == 0, "Failed to compute weights"
            weights.append(value)
        return weights

    def _get_collision_rule_with_relaxation_matrix(self, relaxation_matrix, conserved_quantity_equations=None,
                                                   moment_subexpressions=False, pre_collision_subexpressions=True,
                                                   post_collision_subexpressions=False, include_force_terms=True):
        def tuple_to_symbol(exp, prefix):
            dim = len(exp)
            format_string = prefix + "_" + "_".join(["%d"] * dim)
            return sp.Symbol(format_string % exp)

        def substitute_conserved_quantities(expressions, cqe):
            cqe = cqe.new_without_subexpressions()
            substitution_dict = {eq.rhs: eq.lhs for eq in cqe.main_assignments}
            density = cqe.main_assignments[0].lhs
            substitution_dict.update({density * eq.rhs: density * eq.lhs for eq in cqe.main_assignments[1:]})
            return [fast_subs(e, substitution_dict) for e in expressions]

        f = self.pre_collision_pdf_symbols
        if conserved_quantity_equations is None:
            conserved_quantity_equations = self._conserved_quantity_computation.equilibrium_input_equations_from_pdfs(f)

        subexpressions = conserved_quantity_equations.all_assignments

        # 1) Determine monomial indices, and arrange them such that the zeroth and first order indices come first
        indices = list(extract_monomials(self.cumulants, dim=len(self.stencil[0])))
        zeroth_moment_exponent = (0,) * self.dim
        first_moment_exponents = [tuple([1 if i == j else 0 for i in range(self.dim)]) for j in range(self.dim)]
        lower_order_indices = [zeroth_moment_exponent] + first_moment_exponents
        num_lower_order_indices = len(lower_order_indices)
        assert all(e in indices for e in lower_order_indices), \
            "Cumulant system does not contain relaxation rules for zeroth and first order cumulants"
        higher_order_indices = [e for e in indices if e not in lower_order_indices]
        indices = lower_order_indices + higher_order_indices  # reorder

        # 2) Transform pdfs to moments
        moment_transformation_matrix = moment_matrix(indices, self.stencil)
        moments = moment_transformation_matrix * sp.Matrix(f)
        moments = substitute_conserved_quantities(moments, conserved_quantity_equations)
        if moment_subexpressions:
            symbols = [tuple_to_symbol(t, "m") for t in higher_order_indices]
            subexpressions += [Assignment(sym, moment)
                               for sym, moment in zip(symbols, moments[num_lower_order_indices:])]
            moments = moments[:num_lower_order_indices] + symbols

        # 3) Transform moments to monomial cumulants
        moments_dict = {idx: m for idx, m in zip(indices, moments)}
        monomial_cumulants = [cumulant_as_function_of_raw_moments(idx, moments_dict) for idx in indices]

        if pre_collision_subexpressions:
            symbols = [tuple_to_symbol(t, "pre_c") for t in higher_order_indices]
            subexpressions += [Assignment(sym, c)
                               for sym, c in zip(symbols, monomial_cumulants[num_lower_order_indices:])]
            monomial_cumulants = monomial_cumulants[:num_lower_order_indices] + symbols

        # 4) Transform monomial to polynomial cumulants which are then relaxed and transformed back
        mon_to_poly = monomial_to_polynomial_transformation_matrix(indices, self.cumulants)
        poly_values = mon_to_poly * sp.Matrix(monomial_cumulants)
        eq_values = sp.Matrix(self.cumulant_equilibrium_values)
        collided_poly_values = poly_values + relaxation_matrix * (eq_values - poly_values)  # collision
        relaxed_monomial_cumulants = mon_to_poly.inv() * collided_poly_values

        if post_collision_subexpressions:
            symbols = [tuple_to_symbol(t, "post_c") for t in higher_order_indices]
            subexpressions += [Assignment(sym, c)
                               for sym, c in zip(symbols, relaxed_monomial_cumulants[num_lower_order_indices:])]
            relaxed_monomial_cumulants = relaxed_monomial_cumulants[:num_lower_order_indices] + symbols

        # 5) Transform post-collision cumulant back to moments and from there to pdfs
        cumulant_dict = {idx: value for idx, value in zip(indices, relaxed_monomial_cumulants)}
        collided_moments = [raw_moment_as_function_of_cumulants(idx, cumulant_dict) for idx in indices]
        result = moment_transformation_matrix.inv() * sp.Matrix(collided_moments)
        main_assignments = [Assignment(sym, val) for sym, val in zip(self.post_collision_pdf_symbols, result)]

        # 6) Add forcing terms
        if self._force_model is not None and include_force_terms:
            force_model_terms = self._force_model(self)
            force_term_symbols = sp.symbols("forceTerm_:%d" % (len(force_model_terms,)))
            force_subexpressions = [Assignment(sym, force_model_term)
                                    for sym, force_model_term in zip(force_term_symbols, force_model_terms)]
            subexpressions += force_subexpressions
            main_assignments = [Assignment(eq.lhs, eq.rhs + force_term_symbol)
                                for eq, force_term_symbol in zip(main_assignments, force_term_symbols)]

        sh = {'relaxation_rates': list(self.relaxation_rates)}
        return LbmCollisionRule(self, main_assignments, subexpressions, simplification_hints=sh)
