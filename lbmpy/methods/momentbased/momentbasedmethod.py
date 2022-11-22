from collections import OrderedDict
from typing import Iterable, Set

import sympy as sp
import numpy as np

from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation, DensityVelocityComputation
from lbmpy.moments import MOMENT_SYMBOLS, moment_matrix
from pystencils.sympyextensions import is_constant
from pystencils import Assignment, AssignmentCollection

from lbmpy.moment_transforms import PdfsToMomentsByChimeraTransform


class MomentBasedLbMethod(AbstractLbMethod):
    """
    Moment based LBM is a class to represent the single (SRT), two (TRT) and multi relaxation time (MRT) methods.
    These methods work by transforming the pdfs into moment space using a linear transformation. In the moment
    space each component (moment) is relaxed to an equilibrium moment by a certain relaxation rate. These
    equilibrium moments can e.g. be determined by taking the equilibrium moments of the continuous Maxwellian.

    Parameters:
        stencil: see :class:`lbmpy.stencils.LBStencil`
        equilibrium: Instance of :class:`lbmpy.equilibrium.AbstractEquilibrium`, defining the equilibrium distribution
                     used by this method.
        relaxation_dict: a dictionary mapping moments in either tuple or polynomial formulation
                         to their relaxation rate.
        conserved_quantity_computation: instance of :class:`lbmpy.methods.AbstractConservedQuantityComputation`.
                                        This determines how conserved quantities are computed, and defines
                                        the symbols used in the equilibrium moments like e.g. density and velocity.
        force_model: Instance of :class:`lbmpy.forcemodels.AbstractForceModel`, or None if no forcing terms are required
        zero_centered: Determines the PDF storage format, regular or centered around the equilibrium's
                       background distribution.
        moment_transform_class: transformation class to transform PDFs to the moment space (subclass of 
                                :class:`lbmpy.moment_transforms.AbstractRawMomentTransform`), or `None`
                                if equations are to be derived in population space. 
    """

    def __init__(self, stencil, equilibrium, relaxation_dict,
                 conserved_quantity_computation=None, force_model=None, zero_centered=False,
                 moment_transform_class=PdfsToMomentsByChimeraTransform):
        assert isinstance(conserved_quantity_computation, AbstractConservedQuantityComputation)
        super(MomentBasedLbMethod, self).__init__(stencil)

        self._equilibrium = equilibrium
        self._relaxation_dict = OrderedDict(relaxation_dict)
        self._cqc = conserved_quantity_computation
        self._force_model = force_model
        self._zero_centered = zero_centered
        self._weights = None
        self._moment_transform_class = moment_transform_class

    @property
    def force_model(self):
        """Force model employed by this method."""
        return self._force_model

    @property
    def relaxation_info_dict(self):
        """Dictionary mapping this method's moments to their relaxation rates and equilibrium values.
        Beware: Changes to this dictionary are not reflected in the method. For changing relaxation rates,
        use `relaxation_rate_dict` instead."""
        return OrderedDict({m: RelaxationInfo(v, rr)
                            for (m, rr), v in zip(self._relaxation_dict.items(), self.moment_equilibrium_values)})

    @property
    def conserved_quantity_computation(self):
        return self._cqc

    @property
    def equilibrium_distribution(self):
        """Returns this method's equilibrium distribution (see :class:`lbmpy.equilibrium.AbstractEquilibrium`"""
        return self._equilibrium

    @property
    def moment_transform_class(self):
        """The transform class (subclass of :class:`lbmpy.moment_transforms.AbstractRawMomentTransform` defining the
        transformation of populations to moment space."""
        return self._moment_transform_class

    @property
    def moment_space_collision(self):
        """Returns whether collision is derived in terms of moments or in terms of populations only."""
        return self._moment_transform_class is not None

    @property
    def moments(self):
        """Moments relaxed by this method."""
        return tuple(self._relaxation_dict.keys())

    @property
    def moment_equilibrium_values(self):
        """Equilibrium values of this method's :attr:`moments`."""
        return self._equilibrium.moments(self.moments)

    @property
    def relaxation_rates(self):
        """Relaxation rates for this method's :attr:`moments`."""
        return tuple(self._relaxation_dict.values())

    @property
    def relaxation_rate_dict(self):
        """Dictionary mapping moments to relaxation rates. Changes are reflected by the method."""
        return self._relaxation_dict

    @property
    def zeroth_order_equilibrium_moment_symbol(self, ):
        """Returns a symbol referring to the zeroth-order moment of this method's equilibrium distribution,
        which is the area under it's curve
        (see :attr:`lbmpy.equilibrium.AbstractEquilibrium.zeroth_order_moment_symbol`)."""
        return self._equilibrium.zeroth_order_moment_symbol

    @property
    def first_order_equilibrium_moment_symbols(self, ):
        """Returns a vector of symbols referring to the first-order moment of this method's equilibrium distribution,
        which is its mean value. (see :attr:`lbmpy.equilibrium.AbstractEquilibrium.first_order_moment_symbols`)."""
        return self._equilibrium.first_order_moment_symbols

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self._compute_weights()
        return self._weights

    def override_weights(self, weights):
        """Manually set this method's lattice weights."""
        assert len(weights) == len(self.stencil)
        self._weights = weights

    def get_equilibrium(self, conserved_quantity_equations: AssignmentCollection = None,
                        include_force_terms: bool = False, pre_simplification: bool = False,
                        subexpressions: bool = False, keep_cqc_subexpressions: bool = True) -> LbmCollisionRule:
        """Returns equation collection, to compute equilibrium values.
        The equations have the post collision symbols as left-hand sides and are
        functions of the conserved quantities

        Args:
            conserved_quantity_equations: equations to compute conserved quantities.
            include_force_terms: if set to True the equilibrium is shifted by forcing terms coming from the force model
                                 of the method.
            pre_simplification: with or without pre-simplifications for the calculation of the collision
            subexpressions: if set to false all subexpressions of the equilibrium assignments are plugged
                            into the main assignments
            keep_cqc_subexpressions: if equilibrium is returned without subexpressions keep_cqc_subexpressions
                                     determines if also subexpressions to calculate conserved quantities should be
                                     plugged into the main assignments

        """
        _, d = self._generate_symbolic_relaxation_matrix()
        rr_sub_expressions = set([Assignment(d[i, i], sp.Integer(1)) for i in range(len(self.relaxation_rates))])

        ac = self._collision_rule_with_relaxation_matrix(d=d,
                                                         additional_subexpressions=rr_sub_expressions,
                                                         include_force_terms=include_force_terms,
                                                         conserved_quantity_equations=conserved_quantity_equations,
                                                         pre_simplification=pre_simplification)

        if not subexpressions:
            if keep_cqc_subexpressions:
                bs = self._bound_symbols_cqc(conserved_quantity_equations)
                ac = ac.new_without_subexpressions(subexpressions_to_keep=bs)
                return ac.new_without_unused_subexpressions()
            else:
                ac = ac.new_without_subexpressions()
                return ac.new_without_unused_subexpressions()
        else:
            return ac.new_without_unused_subexpressions()

    def get_equilibrium_terms(self):
        """Returns this method's equilibrium populations as a vector."""
        equilibrium = self.get_equilibrium()
        return sp.Matrix([eq.rhs for eq in equilibrium.main_assignments])

    def get_collision_rule(self, conserved_quantity_equations: AssignmentCollection = None,
                           pre_simplification: bool = True) -> LbmCollisionRule:
        rr_sub_expressions, d = self._generate_symbolic_relaxation_matrix()
        ac = self._collision_rule_with_relaxation_matrix(d=d,
                                                         additional_subexpressions=rr_sub_expressions,
                                                         include_force_terms=True,
                                                         conserved_quantity_equations=conserved_quantity_equations,
                                                         pre_simplification=pre_simplification)
        return ac

    def set_zeroth_moment_relaxation_rate(self, relaxation_rate):
        """Alters the relaxation rate of the zeroth-order moment."""
        one = sp.Rational(1, 1)
        self._relaxation_dict[one] = relaxation_rate

    def set_first_moment_relaxation_rate(self, relaxation_rate):
        """Alters the relaxation rates of the first-order moments."""
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._relaxation_dict, "First moments are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            self._relaxation_dict[e] = relaxation_rate

    def set_conserved_moments_relaxation_rate(self, relaxation_rate):
        """Alters the relaxation rates of the zeroth- and first-order moments."""
        self.set_zeroth_moment_relaxation_rate(relaxation_rate)
        self.set_first_moment_relaxation_rate(relaxation_rate)

    def set_force_model(self, force_model):
        """Updates this method's force model."""
        self._force_model = force_model
        if isinstance(self._cqc, DensityVelocityComputation):
            self._cqc.set_force_model(force_model)

    @property
    def collision_matrix(self) -> sp.Matrix:
        pdfs_to_moments = self.moment_matrix
        d = self.relaxation_matrix
        return pdfs_to_moments.inv() * d * pdfs_to_moments

    @property
    def inverse_collision_matrix(self) -> sp.Matrix:
        pdfs_to_moments = self.moment_matrix
        inverse_relaxation_matrix = self.relaxation_matrix.inv()
        return pdfs_to_moments.inv() * inverse_relaxation_matrix * pdfs_to_moments

    @property
    def moment_matrix(self) -> sp.Matrix:
        return moment_matrix(self.moments, self.stencil)

    @property
    def is_orthogonal(self) -> bool:
        return (self.moment_matrix * self.moment_matrix.T).is_diagonal()

    @property
    def is_weighted_orthogonal(self) -> bool:
        weights_matrix = sp.Matrix([self.weights] * len(self.weights))
        moment_matrix_times_weights = sp.Matrix(np.multiply(self.moment_matrix, weights_matrix))

        return (moment_matrix_times_weights * self.moment_matrix.T).is_diagonal()

    def __getstate__(self):
        # Workaround for a bug in joblib
        self._momentToRelaxationInfoDictToPickle = [i for i in self._relaxation_dict.items()]
        return self.__dict__

    def _repr_html_(self):

        def stylized_bool(b):
            return "&#10003;" if b else "&#10007;"

        html = f"""
        <table style="border:none; width: 100%">
            <tr>
                <th colspan="3" style="text-align: left">
                    Moment-Based Method
                </th>
                <td>Stencil: {self.stencil.name}</td>
                <td>Zero-Centered Storage: {stylized_bool(self._zero_centered)}</td>
                <td>Force Model: {"None" if self._force_model is None else type(self._force_model).__name__}</td>
            </tr>
        </table>
        """

        html += self._equilibrium._repr_html_()

        html += """
        <table style="border:none; width: 100%">
            <tr> <th colspan="3" style="text-align: left"> Relaxation Info </th> </tr>
            <tr>
                <th>Moment</th>
                <th>Eq. Value </th>
                <th>Relaxation Rate</th>
            </tr>
        """

        for moment, (eq_value, rr) in self.relaxation_info_dict.items():
            vals = {
                'rr': sp.latex(rr),
                'moment': sp.latex(moment),
                'eq_value': sp.latex(eq_value),
                'nb': 'style="border:none"',
            }
            html += """<tr {nb}>
                            <td {nb}>${moment}$</td>
                            <td {nb}>${eq_value}$</td>
                            <td {nb}>${rr}$</td>
                         </tr>\n""".format(**vals)

        html += "</table>"
        return html

    def _compute_weights(self):
        bg = self.equilibrium_distribution.background_distribution
        assert bg is not None, "Could not compute weights, since no background distribution is given."
        if bg.discrete_populations is not None:
            #   Compute lattice weights as the discrete populations of the background distribution ...
            weights = bg.discrete_populations
        else:
            #   or, if those are not available, by moment matching.
            mm_inv = self.moment_matrix.inv()
            bg_moments = bg.moments(self.moments)
            weights = (mm_inv * sp.Matrix(bg_moments)).expand()

        for w in weights:
            assert is_constant(w)

        return [w for w in weights]

    def _bound_symbols_cqc(self, conserved_quantity_equations: AssignmentCollection = None) -> Set[sp.Symbol]:
        f = self.pre_collision_pdf_symbols
        cqe = conserved_quantity_equations

        if cqe is None:
            cqe = self._cqc.equilibrium_input_equations_from_pdfs(f, False)

        return cqe.bound_symbols

    def _collision_rule_with_relaxation_matrix(self, d: sp.Matrix,
                                               additional_subexpressions: Iterable[Assignment] = None,
                                               include_force_terms: bool = True,
                                               conserved_quantity_equations: AssignmentCollection = None,
                                               pre_simplification: bool = False) -> LbmCollisionRule:
        if additional_subexpressions is None:
            additional_subexpressions = list()
        f = sp.Matrix(self.pre_collision_pdf_symbols)
        moment_polynomials = list(self.moments)

        cqe = conserved_quantity_equations
        if cqe is None:
            cqe = self._cqc.equilibrium_input_equations_from_pdfs(f, False)

        if self._force_model is None:
            include_force_terms = False

        moment_space_forcing = False

        if include_force_terms and self._moment_transform_class:
            if self._force_model is not None:
                moment_space_forcing = self._force_model.has_moment_space_forcing

        forcing_subexpressions = []
        if self._force_model is not None:
            forcing_subexpressions = AssignmentCollection(self._force_model.subs_dict_force).all_assignments

        rho = self.zeroth_order_equilibrium_moment_symbol
        u = self.first_order_equilibrium_moment_symbols

        #   See if a background shift is necessary
        if self._zero_centered and not self._equilibrium.deviation_only:
            background_distribution = self._equilibrium.background_distribution
            assert background_distribution is not None
        else:
            background_distribution = None

        m_eq = sp.Matrix(self.moment_equilibrium_values)

        if self._moment_transform_class:
            # Derive equations in moment space if a transform is given
            pdf_to_m_transform = self._moment_transform_class(self.stencil, moment_polynomials, rho, u,
                                                              conserved_quantity_equations=cqe,
                                                              background_distribution=background_distribution)

            m_pre = pdf_to_m_transform.pre_collision_symbols
            m_post = pdf_to_m_transform.post_collision_symbols

            pdf_to_m_eqs = pdf_to_m_transform.forward_transform(self.pre_collision_pdf_symbols,
                                                                simplification=pre_simplification)
            m_post_to_f_post_eqs = pdf_to_m_transform.backward_transform(self.post_collision_pdf_symbols,
                                                                         simplification=pre_simplification)

            m_pre_vec = sp.Matrix(m_pre)
            collision_rule = m_pre_vec + d * (m_eq - m_pre_vec)

            if include_force_terms and moment_space_forcing:
                collision_rule += self._force_model.moment_space_forcing(self)

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
            if self._zero_centered and not self._equilibrium.deviation_only:
                raise Exception("Can only derive population-space equations for zero-centered storage"
                                " if delta equilibrium is used.")
            pdf_to_moment = self.moment_matrix
            collision_rule = f + pdf_to_moment.inv() * d * (m_eq - pdf_to_moment * f)
            collision_eqs = [Assignment(lhs, rhs) for lhs, rhs in zip(self.post_collision_pdf_symbols, collision_rule)]
            subexpressions = list(additional_subexpressions) + forcing_subexpressions + cqe.all_assignments
            main_assignments = collision_eqs

        simplification_hints = cqe.simplification_hints.copy()
        simplification_hints.update(self._cqc.defined_symbols())
        simplification_hints['relaxation_rates'] = [d[i, i] for i in range(d.rows)]

        if include_force_terms and not moment_space_forcing:
            force_model_terms = self._force_model(self)
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
