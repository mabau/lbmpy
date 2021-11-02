from collections import OrderedDict

import sympy as sp
import numpy as np

from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.moments import MOMENT_SYMBOLS, moment_matrix
from pystencils.sympyextensions import subs_additive
from pystencils import Assignment, AssignmentCollection

from lbmpy.moment_transforms import PdfsToMomentsByChimeraTransform


class MomentBasedLbMethod(AbstractLbMethod):
    def __init__(self, stencil, moment_to_relaxation_info_dict, conserved_quantity_computation=None, force_model=None,
                 moment_transform_class=PdfsToMomentsByChimeraTransform):
        """
        Moment based LBM is a class to represent the single (SRT), two (TRT) and multi relaxation time (MRT) methods.
        These methods work by transforming the pdfs into moment space using a linear transformation. In the moment
        space each component (moment) is relaxed to an equilibrium moment by a certain relaxation rate. These
        equilibrium moments can e.g. be determined by taking the equilibrium moments of the continuous Maxwellian.

        Args:
            stencil: see :class:`lbmpy.stencils.LBStencil`
            moment_to_relaxation_info_dict: a dictionary mapping moments in either tuple or polynomial formulation
                                            to a RelaxationInfo, which consists of the corresponding equilibrium moment
                                            and a relaxation rate
            conserved_quantity_computation: instance of :class:`lbmpy.methods.AbstractConservedQuantityComputation`.
                                            This determines how conserved quantities are computed, and defines
                                            the symbols used in the equilibrium moments like e.g. density and velocity
            force_model: force model instance, or None if no forcing terms are required
            moment_transform_class: transformation class to transform PDFs to the moment space.
        """
        assert isinstance(conserved_quantity_computation, AbstractConservedQuantityComputation)
        super(MomentBasedLbMethod, self).__init__(stencil)

        self._forceModel = force_model
        self._momentToRelaxationInfoDict = OrderedDict(moment_to_relaxation_info_dict.items())
        self._conservedQuantityComputation = conserved_quantity_computation
        self._weights = None
        self._moment_transform_class = moment_transform_class

    @property
    def force_model(self):
        return self._forceModel

    @property
    def moment_space_collision(self):
        """Returns whether collision is derived in terms of moments or in terms of populations only."""
        return self._moment_transform_class is not None

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

    def get_equilibrium(self, conserved_quantity_equations=None, include_force_terms=False,
                        pre_simplification=False, subexpressions=False, keep_cqc_subexpressions=True):
        relaxation_matrix = sp.eye(len(self.relaxation_rates))
        ac = self._collision_rule_with_relaxation_matrix(relaxation_matrix,
                                                         conserved_quantity_equations=conserved_quantity_equations,
                                                         include_force_terms=include_force_terms,
                                                         pre_simplification=pre_simplification)
        if not subexpressions:
            if keep_cqc_subexpressions:
                bs = self._bound_symbols_cqc(conserved_quantity_equations)
                return ac.new_without_subexpressions(subexpressions_to_keep=bs)
            else:
                return ac.new_without_subexpressions()
        else:
            return ac

    def get_equilibrium_terms(self):
        equilibrium = self.get_equilibrium()
        return sp.Matrix([eq.rhs for eq in equilibrium.main_assignments])

    def get_collision_rule(self, conserved_quantity_equations=None, pre_simplification=True):
        relaxation_rate_sub_expressions, d = self._generate_symbolic_relaxation_matrix()
        ac = self._collision_rule_with_relaxation_matrix(d, relaxation_rate_sub_expressions,
                                                         True, conserved_quantity_equations,
                                                         pre_simplification=pre_simplification)
        return ac

    def set_zeroth_moment_relaxation_rate(self, relaxation_rate):
        one = sp.Rational(1, 1)
        prev_entry = self._momentToRelaxationInfoDict[one]
        new_entry = RelaxationInfo(prev_entry[0], relaxation_rate)
        self._momentToRelaxationInfoDict[one] = new_entry

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
    def is_orthogonal(self):
        return (self.moment_matrix * self.moment_matrix.T).is_diagonal()

    @property
    def is_weighted_orthogonal(self):
        weights_matrix = sp.Matrix([self.weights] * len(self.weights))
        moment_matrix_times_weights = sp.Matrix(np.multiply(self.moment_matrix, weights_matrix))

        return (moment_matrix_times_weights * self.moment_matrix.T).is_diagonal()

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

    def _bound_symbols_cqc(self, conserved_quantity_equations=None):
        f = self.pre_collision_pdf_symbols
        cqe = conserved_quantity_equations

        if cqe is None:
            cqe = self._conservedQuantityComputation.equilibrium_input_equations_from_pdfs(f, False)

        return cqe.bound_symbols

    def _collision_rule_with_relaxation_matrix(self, d, additional_subexpressions=(), include_force_terms=True,
                                               conserved_quantity_equations=None, pre_simplification=False):
        f = sp.Matrix(self.pre_collision_pdf_symbols)
        moment_polynomials = list(self.moments)

        cqe = conserved_quantity_equations
        if cqe is None:
            cqe = self._conservedQuantityComputation.equilibrium_input_equations_from_pdfs(f, False)

        if self._forceModel is None:
            include_force_terms = False

        moment_space_forcing = False

        if include_force_terms and self._moment_transform_class:
            if self._forceModel is not None:
                moment_space_forcing = self._forceModel.has_moment_space_forcing

        forcing_subexpressions = []
        if self._forceModel is not None:
            forcing_subexpressions = AssignmentCollection(self._forceModel.subs_dict_force).all_assignments

        rho = self.zeroth_order_equilibrium_moment_symbol
        u = self.first_order_equilibrium_moment_symbols
        m_eq = sp.Matrix(self.moment_equilibrium_values)

        if self._moment_transform_class:
            #   Derive equations in moment space if a transform is given
            pdf_to_m_transform = self._moment_transform_class(self.stencil, moment_polynomials, rho, u,
                                                              conserved_quantity_equations=cqe)

            m_pre = pdf_to_m_transform.pre_collision_symbols
            m_post = pdf_to_m_transform.post_collision_symbols

            pdf_to_m_eqs = pdf_to_m_transform.forward_transform(self.pre_collision_pdf_symbols,
                                                                simplification=pre_simplification)
            m_post_to_f_post_eqs = pdf_to_m_transform.backward_transform(self.post_collision_pdf_symbols,
                                                                         simplification=pre_simplification)

            m_pre_vec = sp.Matrix(m_pre)
            collision_rule = m_pre_vec + d * (m_eq - m_pre_vec)

            if include_force_terms and moment_space_forcing:
                collision_rule += self._forceModel.moment_space_forcing(self)

            collision_eqs = [Assignment(lhs, rhs) for lhs, rhs in zip(m_post, collision_rule)]
            collision_eqs = AssignmentCollection(collision_eqs)

            all_acs = [] if pdf_to_m_transform.absorbs_conserved_quantity_equations else [cqe]
            all_acs += [pdf_to_m_eqs, collision_eqs]
            subexpressions = list(additional_subexpressions) + forcing_subexpressions + [ac.all_assignments for ac in
                                                                                         all_acs]
            subexpressions += m_post_to_f_post_eqs.subexpressions
            main_assignments = m_post_to_f_post_eqs.main_assignments
        else:
            #   For SRT, TRT by default, and whenever customly required, derive equations entirely in
            #   population space
            pdf_to_moment = self.moment_matrix
            collision_rule = f + pdf_to_moment.inv() * d * (m_eq - pdf_to_moment * f)
            collision_eqs = [Assignment(lhs, rhs) for lhs, rhs in zip(self.post_collision_pdf_symbols, collision_rule)]
            subexpressions = list(additional_subexpressions) + forcing_subexpressions + cqe.all_assignments
            main_assignments = collision_eqs

        simplification_hints = cqe.simplification_hints.copy()
        simplification_hints.update(self._conservedQuantityComputation.defined_symbols())
        simplification_hints['relaxation_rates'] = [d[i, i] for i in range(d.rows)]

        if include_force_terms and not moment_space_forcing:
            force_model_terms = self._forceModel(self)
            force_term_symbols = sp.symbols(f"forceTerm_:{len(force_model_terms)}")
            force_subexpressions = [Assignment(sym, force_model_term)
                                    for sym, force_model_term in zip(force_term_symbols, force_model_terms)]
            subexpressions += force_subexpressions
            main_assignments = [Assignment(eq.lhs, eq.rhs + force_term_symbol)
                                for eq, force_term_symbol in zip(main_assignments, force_term_symbols)]
            simplification_hints['force_terms'] = force_term_symbols

        ac = LbmCollisionRule(self, main_assignments, subexpressions, simplification_hints)
        ac.topological_sort()
        return ac
