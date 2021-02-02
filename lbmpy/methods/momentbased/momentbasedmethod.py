from collections import OrderedDict

import sympy as sp

from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.moments import MOMENT_SYMBOLS, moment_matrix
from pystencils import Assignment
from pystencils.sympyextensions import subs_additive


class MomentBasedLbMethod(AbstractLbMethod):
    def __init__(self, stencil, moment_to_relaxation_info_dict, conserved_quantity_computation=None, force_model=None):
        """
        Moment based LBM is a class to represent the single (SRT), two (TRT) and multi relaxation time (MRT) methods.
        These methods work by transforming the pdfs into moment space using a linear transformation. In the moment
        space each component (moment) is relaxed to an equilibrium moment by a certain relaxation rate. These
        equilibrium moments can e.g. be determined by taking the equilibrium moments of the continuous Maxwellian.

        Args:
            stencil: see :func:`lbmpy.stencils.get_stencil`
            moment_to_relaxation_info_dict: a dictionary mapping moments in either tuple or polynomial formulation
                                            to a RelaxationInfo, which consists of the corresponding equilibrium moment
                                            and a relaxation rate
            conserved_quantity_computation: instance of :class:`lbmpy.methods.AbstractConservedQuantityComputation`.
                                            This determines how conserved quantities are computed, and defines
                                            the symbols used in the equilibrium moments like e.g. density and velocity
            force_model: force model instance, or None if no forcing terms are required
        """
        assert isinstance(conserved_quantity_computation, AbstractConservedQuantityComputation)
        super(MomentBasedLbMethod, self).__init__(stencil)

        self._forceModel = force_model
        self._momentToRelaxationInfoDict = OrderedDict(moment_to_relaxation_info_dict.items())
        self._conservedQuantityComputation = conserved_quantity_computation
        self._weights = None

    @property
    def force_model(self):
        return self._forceModel

    @property
    def relaxation_info_dict(self):
        return self._momentToRelaxationInfoDict

    @property
    def conserved_quantity_computation(self):
        return self._conservedQuantityComputation

    @property
    def moments(self):
        return tuple(self._momentToRelaxationInfoDict.keys())

    @property
    def moment_equilibrium_values(self):
        return tuple([e.equilibrium_value for e in self._momentToRelaxationInfoDict.values()])

    @property
    def relaxation_rates(self):
        return tuple([e.relaxation_rate for e in self._momentToRelaxationInfoDict.values()])

    @property
    def zeroth_order_equilibrium_moment_symbol(self, ):
        return self._conservedQuantityComputation.zeroth_order_moment_symbol

    @property
    def first_order_equilibrium_moment_symbols(self, ):
        return self._conservedQuantityComputation.first_order_moment_symbols

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self._compute_weights()
        return self._weights

    def override_weights(self, weights):
        assert len(weights) == len(self.stencil)
        self._weights = weights

    def get_equilibrium(self, conserved_quantity_equations=None, include_force_terms=False):
        relaxation_matrix = sp.eye(len(self.relaxation_rates))
        return self._collision_rule_with_relaxation_matrix(relaxation_matrix,
                                                           conserved_quantity_equations=conserved_quantity_equations,
                                                           include_force_terms=include_force_terms)

    def get_equilibrium_terms(self):
        equilibrium = self.get_equilibrium()
        return sp.Matrix([eq.rhs for eq in equilibrium.main_assignments])

    def get_collision_rule(self, conserved_quantity_equations=None, pre_simplification=True):
        d = self.relaxation_matrix
        relaxation_rate_sub_expressions, d = self._generate_relaxation_matrix(d, pre_simplification)
        ac = self._collision_rule_with_relaxation_matrix(d, relaxation_rate_sub_expressions,
                                                         True, conserved_quantity_equations)
        return ac

    def set_zeroth_moment_relaxation_rate(self, relaxation_rate):
        e = sp.Rational(1, 1)
        prev_entry = self._momentToRelaxationInfoDict[e]
        new_entry = RelaxationInfo(prev_entry[0], relaxation_rate)
        self._momentToRelaxationInfoDict[e] = new_entry

    def set_first_moment_relaxation_rate(self, relaxation_rate):
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._momentToRelaxationInfoDict, "First moments are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            prev_entry = self._momentToRelaxationInfoDict[e]
            new_entry = RelaxationInfo(prev_entry[0], relaxation_rate)
            self._momentToRelaxationInfoDict[e] = new_entry

    def set_conserved_moments_relaxation_rate(self, relaxation_rate):
        self.set_zeroth_moment_relaxation_rate(relaxation_rate)
        self.set_first_moment_relaxation_rate(relaxation_rate)

    def set_force_model(self, force_model):
        self._forceModel = force_model

    @property
    def collision_matrix(self):
        pdfs_to_moments = self.moment_matrix
        d = self.relaxation_matrix
        return pdfs_to_moments.inv() * d * pdfs_to_moments

    @property
    def inverse_collision_matrix(self):
        pdfs_to_moments = self.moment_matrix
        inverse_relaxation_matrix = self.relaxation_matrix.inv()
        return pdfs_to_moments.inv() * inverse_relaxation_matrix * pdfs_to_moments

    @property
    def moment_matrix(self):
        return moment_matrix(self.moments, self.stencil)

    @property
    def relaxation_matrix(self):
        d = sp.zeros(len(self.relaxation_rates))
        for i in range(0, len(self.relaxation_rates)):
            d[i, i] = self.relaxation_rates[i]
        return d

    @property
    def is_orthogonal(self):
        return (self.moment_matrix * self.moment_matrix.T).is_diagonal()

    @property
    def is_weighted_orthogonal(self):
        w = get_weights(self.stencil, sp.Rational(1, 3))
        return (sp.matrix_multiply_elementwise(self.moment_matrix, sp.Matrix([w] * len(w))) * self.moment_matrix.T
                ).is_diagonal()

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
        for moment, (eq_value, rr) in self._momentToRelaxationInfoDict.items():
            vals = {
                'rr': sp.latex(rr),
                'moment': sp.latex(moment),
                'eq_value': sp.latex(eq_value),
                'nb': 'style="border:none"',
            }
            content += """<tr {nb}>
                            <td {nb}>${moment}$</td>
                            <td {nb}>${eq_value}$</td>
                            <td {nb}>${rr}$</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')

    def _compute_weights(self):
        replacements = self._conservedQuantityComputation.default_values
        ac = self.get_equilibrium(include_force_terms=False)
        ac = ac.new_with_substitutions(replacements, substitute_on_lhs=False).new_without_subexpressions()

        new_assignments = [Assignment(e.lhs,
                                      subs_additive(e.rhs, sp.sympify(1), sum(self.pre_collision_pdf_symbols),
                                                    required_match_replacement=1.0))
                           for e in ac.main_assignments]
        ac = ac.copy(new_assignments)

        weights = []
        for eq in ac.main_assignments:
            value = eq.rhs.expand()
            assert len(value.atoms(sp.Symbol)) == 0, "Failed to compute weights " + str(value)
            weights.append(value)
        return weights

    def _collision_rule_with_relaxation_matrix(self, d, additional_subexpressions=(), include_force_terms=True,
                                               conserved_quantity_equations=None):
        f = sp.Matrix(self.pre_collision_pdf_symbols)
        pdf_to_moment = self.moment_matrix
        m_eq = sp.Matrix(self.moment_equilibrium_values)

        collision_rule = f + pdf_to_moment.inv() * d * (m_eq - pdf_to_moment * f)
        collision_eqs = [Assignment(lhs, rhs) for lhs, rhs in zip(self.post_collision_pdf_symbols, collision_rule)]

        if conserved_quantity_equations is None:
            conserved_quantity_equations = self._conservedQuantityComputation.equilibrium_input_equations_from_pdfs(f)

        simplification_hints = conserved_quantity_equations.simplification_hints.copy()
        simplification_hints.update(self._conservedQuantityComputation.defined_symbols())
        simplification_hints['relaxation_rates'] = [d[i, i] for i in range(d.rows)]

        all_subexpressions = list(additional_subexpressions) + conserved_quantity_equations.all_assignments

        if self._forceModel is not None and include_force_terms:
            force_model_terms = self._forceModel(self)
            force_term_symbols = sp.symbols("forceTerm_:%d" % (len(force_model_terms,)))
            force_subexpressions = [Assignment(sym, force_model_term)
                                    for sym, force_model_term in zip(force_term_symbols, force_model_terms)]
            all_subexpressions += force_subexpressions
            collision_eqs = [Assignment(eq.lhs, eq.rhs + force_term_symbol)
                             for eq, force_term_symbol in zip(collision_eqs, force_term_symbols)]
            simplification_hints['force_terms'] = force_term_symbols

        return LbmCollisionRule(self, collision_eqs, all_subexpressions,
                                simplification_hints)

    @staticmethod
    def _generate_relaxation_matrix(relaxation_matrix, keep_rr_symbolic):
        """
        For SRT and TRT the equations can be easier simplified if the relaxation times are symbols, not numbers.
        This function replaces the numbers in the relaxation matrix with symbols in this case, and returns also
        the subexpressions, that assign the number to the newly introduced symbol
        """
        rr = [relaxation_matrix[i, i] for i in range(relaxation_matrix.rows)]
        if keep_rr_symbolic <= 2:
            unique_relaxation_rates = set(rr)
            subexpressions = {}
            for rt in unique_relaxation_rates:
                rt = sp.sympify(rt)
                if not isinstance(rt, sp.Symbol):
                    rt_symbol = sp.Symbol("rr_%d" % (len(subexpressions),))
                    subexpressions[rt] = rt_symbol

            new_rr = [subexpressions[sp.sympify(e)] if sp.sympify(e) in subexpressions else e
                      for e in rr]
            substitutions = [Assignment(e[1], e[0]) for e in subexpressions.items()]
            return substitutions, sp.diag(*new_rr)
        else:
            return [], relaxation_matrix
