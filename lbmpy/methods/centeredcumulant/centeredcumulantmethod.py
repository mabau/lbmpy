from pystencils.simp.simplifications import sympy_cse
import sympy as sp
from collections import OrderedDict
from warnings import warn, filterwarnings

from pystencils import Assignment, AssignmentCollection
from pystencils.simp.assignment_collection import SymbolGen
from pystencils.stencil import have_same_entries
from pystencils.cache import disk_cache
from pystencils.sympyextensions import is_constant

from lbmpy.enums import Stencil
from lbmpy.stencils import LBStencil
from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation

from lbmpy.moments import (moments_up_to_order, get_order, moment_matrix,
                           monomial_to_polynomial_transformation_matrix,
                           moment_sort_key, exponent_tuple_sort_key,
                           exponent_to_polynomial_representation, extract_monomials, MOMENT_SYMBOLS,
                           statistical_quantity_symbol)

from lbmpy.forcemodels import Luo, Simple

#   Local Imports

from .cumulant_transform import (
    PRE_COLLISION_CUMULANT, POST_COLLISION_CUMULANT,
    CentralMomentsToCumulantsByGeneratingFunc)

from lbmpy.moment_transforms import (
    PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT, POST_COLLISION_MONOMIAL_CENTRAL_MOMENT,
    PdfsToCentralMomentsByShiftMatrix)

from lbmpy.methods.centeredcumulant.force_model import CenteredCumulantForceModel
from lbmpy.methods.centeredcumulant.galilean_correction import (
    contains_corrected_polynomials,
    add_galilean_correction,
    get_galilean_correction_terms)


#   ============================ Cached Transformations ================================================================

@disk_cache
def cached_forward_transform(transform_obj, *args, **kwargs):
    return transform_obj.forward_transform(*args, **kwargs)


@disk_cache
def cached_backward_transform(transform_obj, *args, **kwargs):
    return transform_obj.backward_transform(*args, **kwargs)


#   ============================ Lower Order Central Moment Collision ==================================================


@disk_cache
def relax_lower_order_central_moments(moment_indices, pre_collision_values,
                                      relaxation_rates, equilibrium_values,
                                      post_collision_base=POST_COLLISION_MONOMIAL_CENTRAL_MOMENT):
    post_collision_symbols = [statistical_quantity_symbol(post_collision_base, i) for i in moment_indices]
    equilibrium_vec = sp.Matrix(equilibrium_values)
    moment_vec = sp.Matrix(pre_collision_values)
    relaxation_matrix = sp.diag(*relaxation_rates)
    moment_vec = moment_vec + relaxation_matrix * (equilibrium_vec - moment_vec)
    main_assignments = [Assignment(s, eq) for s, eq in zip(post_collision_symbols, moment_vec)]

    return AssignmentCollection(main_assignments)


#   ============================ Polynomial Cumulant Collision =========================================================

@disk_cache
def relax_polynomial_cumulants(monomial_exponents, polynomials, relaxation_rates, equilibrium_values,
                               pre_simplification,
                               galilean_correction_terms=None,
                               pre_collision_base=PRE_COLLISION_CUMULANT,
                               post_collision_base=POST_COLLISION_CUMULANT,
                               subexpression_base='sub_col'):
    mon_to_poly_matrix = monomial_to_polynomial_transformation_matrix(monomial_exponents, polynomials)
    mon_vec = sp.Matrix([statistical_quantity_symbol(pre_collision_base, exp) for exp in monomial_exponents])
    equilibrium_vec = sp.Matrix(equilibrium_values)
    relaxation_matrix = sp.diag(*relaxation_rates)

    subexpressions = []

    poly_vec = mon_to_poly_matrix * mon_vec
    relaxed_polys = poly_vec + relaxation_matrix * (equilibrium_vec - poly_vec)

    if galilean_correction_terms is not None:
        relaxed_polys = add_galilean_correction(relaxed_polys, polynomials, galilean_correction_terms)
        subexpressions = galilean_correction_terms.all_assignments

    relaxed_monos = mon_to_poly_matrix.inv() * relaxed_polys

    main_assignments = [Assignment(statistical_quantity_symbol(post_collision_base, exp), v)
                        for exp, v in zip(monomial_exponents, relaxed_monos)]

    symbol_gen = SymbolGen(subexpression_base)
    ac = AssignmentCollection(
        main_assignments, subexpressions=subexpressions, subexpression_symbol_generator=symbol_gen)
    if pre_simplification == 'default_with_cse':
        ac = sympy_cse(ac)
    return ac


#   =============================== LB Method Implementation ===========================================================

class CenteredCumulantBasedLbMethod(AbstractLbMethod):
    """
    This class implements cumulant-based lattice boltzmann methods which relax all the non-conserved quantities
    as either monomial or polynomial cumulants. It is mostly inspired by the work presented in :cite:`geier2015`.

    Conserved quantities are relaxed in central moment space. This method supports an implicit forcing scheme
    through :class:`lbmpy.methods.centeredcumulant.CenteredCumulantForceModel` where forces are applied by
    shifting the central-moment frame of reference by :math:`F/2` and then relaxing the first-order central
    moments with a relaxation rate of two. This corresponds to the change-of-sign described in the paper.
    Classical forcing schemes can still be applied.

    The galilean correction described in :cite:`geier2015` is also available for the D3Q27 lattice.

    This method is implemented modularily as the transformation from populations to central moments to cumulants
    is governed by subclasses of :class:`lbmpy.moment_transforms.AbstractMomentTransform`
    which can be specified by constructor argument. This allows the selection of the most efficient transformation
    for a given setup.

    Parameters:
        stencil: see :class:`lbmpy.stencils.LBStencil`
        equilibrium: Instance of :class:`lbmpy.equilibrium.AbstractEquilibrium`, defining the equilibrium distribution
                     used by this method.
        relaxation_dict: a dictionary mapping cumulants in either tuple or polynomial formulation
                         to their relaxation rate.
        conserved_quantity_computation: instance of :class:`lbmpy.methods.AbstractConservedQuantityComputation`.
                                        This determines how conserved quantities are computed, and defines
                                        the symbols used in the equilibrium moments like e.g. density and velocity.
        force_model: Instance of :class:`lbmpy.forcemodels.AbstractForceModel`, or None if no forcing terms are required
        zero_centered: Determines the PDF storage format, regular or centered around the equilibrium's
                       background distribution.
        galilean_correction: if set to True the galilean_correction is applied to a D3Q27 cumulant method
        central_moment_transform_class: transformation class to transform PDFs to central moment space (subclass of 
                                        :class:`lbmpy.moment_transforms.AbstractCentralMomentTransform`)
        cumulant_transform_class: transform class to get from the central moment space to the cumulant space
    """

    def __init__(self, stencil, equilibrium, relaxation_dict,
                 conserved_quantity_computation=None,
                 force_model=None, zero_centered=False,
                 galilean_correction=False,
                 central_moment_transform_class=PdfsToCentralMomentsByShiftMatrix,
                 cumulant_transform_class=CentralMomentsToCumulantsByGeneratingFunc):
        assert isinstance(conserved_quantity_computation,
                          AbstractConservedQuantityComputation)
        super(CenteredCumulantBasedLbMethod, self).__init__(stencil)

        if force_model is not None:
            assert (isinstance(force_model, CenteredCumulantForceModel)
                    or isinstance(force_model, Simple)
                    or isinstance(force_model, Luo)), "Given force model currently not supported."

        for m in moments_up_to_order(1, dim=self.dim):
            if exponent_to_polynomial_representation(m) not in relaxation_dict.keys():
                raise ValueError(f'No relaxation info given for conserved cumulant {m}!')

        self._equilibrium = equilibrium
        self._relaxation_dict = OrderedDict(relaxation_dict)
        self._cqc = conserved_quantity_computation
        self._force_model = force_model
        self._zero_centered = zero_centered
        self._weights = None
        self._galilean_correction = galilean_correction

        if galilean_correction:
            if not have_same_entries(stencil, LBStencil(Stencil.D3Q27)):
                raise ValueError("Galilean Correction only available for D3Q27 stencil")

            if not contains_corrected_polynomials(relaxation_dict):
                raise ValueError("For the galilean correction, all three polynomial cumulants"
                                 "(x^2 - y^2), (x^2 - z^2) and (x^2 + y^2 + z^2) must be present!")

        self._cumulant_transform_class = cumulant_transform_class
        self._central_moment_transform_class = central_moment_transform_class

        self.force_model_rr_override = False
        if isinstance(self._force_model, CenteredCumulantForceModel) and \
                self._force_model.override_momentum_relaxation_rate is not None:
            self.set_first_moment_relaxation_rate(self._force_model.override_momentum_relaxation_rate)
            self.force_model_rr_override = True

    @property
    def force_model(self):
        """Force model employed by this method."""
        return self._force_model

    @property
    def relaxation_info_dict(self):
        """Dictionary mapping this method's cumulants to their relaxation rates and equilibrium values.
        Beware: Changes to this dictionary are not reflected in the method. For changing relaxation rates,
        use `relaxation_rate_dict` instead."""
        return OrderedDict({m: RelaxationInfo(v, rr)
                            for (m, rr), v in zip(self._relaxation_dict.items(), self.cumulant_equilibrium_values)})

    @property
    def conserved_quantity_computation(self):
        """Returns an instance of class :class:`lbmpy.methods.AbstractConservedQuantityComputation`"""
        return self._cqc

    @property
    def equilibrium_distribution(self):
        """Returns this method's equilibrium distribution (see :class:`lbmpy.equilibrium.AbstractEquilibrium`"""
        return self._equilibrium

    @property
    def central_moment_transform_class(self):
        """The transform class (subclass of :class:`lbmpy.moment_transforms.AbstractCentralMomentTransform` defining the
        transformation of populations to central moment space."""
        return self._central_moment_transform_class

    @property
    def cumulant_transform_class(self):
        """The transform class defining the transform from central moment to cumulant space."""
        return self._cumulant_transform_class

    @property
    def cumulants(self):
        """Cumulants relaxed by this method."""
        return tuple(self._relaxation_dict.keys())

    @property
    def cumulant_equilibrium_values(self):
        """Equilibrium values of this method's :attr:`cumulants`."""
        cumulants = self.cumulants
        equilibria = list(self._equilibrium.cumulants(cumulants, rescale=True))

        for i, c in enumerate(cumulants):
            if get_order(c) == 0:
                equilibria[i] = self.zeroth_order_equilibrium_moment_symbol
            elif get_order(c) == 1:
                equilibria[i] = sp.Integer(0)
        return tuple(equilibria)

    @property
    def relaxation_rates(self):
        """Relaxation rates for this method's :attr:`cumulants`."""
        return tuple(self._relaxation_dict.values())

    @property
    def relaxation_rate_dict(self):
        """Dictionary mapping cumulants to relaxation rates. Changes are reflected by the method."""
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
    def galilean_correction(self):
        """Whether or not the gallilean correction is included in the collision equations."""
        return self._galilean_correction

    def set_zeroth_moment_relaxation_rate(self, relaxation_rate):
        e = sp.Rational(1, 1)
        self._relaxation_dict[e] = relaxation_rate

    def set_first_moment_relaxation_rate(self, relaxation_rate):
        if self.force_model_rr_override:
            warn("Overwriting first-order relaxation rates governed by CenteredCumulantForceModel "
                 "might break your forcing scheme.")
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._relaxation_dict, \
                "First cumulants are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            self._relaxation_dict[e] = relaxation_rate

    def set_conserved_moments_relaxation_rate(self, relaxation_rate):
        self.set_zeroth_moment_relaxation_rate(relaxation_rate)
        self.set_first_moment_relaxation_rate(relaxation_rate)

    def set_force_model(self, force_model):
        assert (isinstance(force_model, CenteredCumulantForceModel)
                or isinstance(force_model, Simple)
                or isinstance(force_model, Luo)), "Given force model currently not supported."
        self._force_model = force_model

    def _repr_html_(self):
        def stylized_bool(b):
            return "&#10003;" if b else "&#10007;"

        html = f"""
        <table style="border:none; width: 100%">
            <tr>
                <th colspan="3" style="text-align: left">
                    Cumulant-Based Method
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
                <th>Cumulant</th>
                <th>Eq. Value </th>
                <th>Relaxation Rate</th>
            </tr>
        """

        for cumulant, (eq_value, rr) in self.relaxation_info_dict.items():
            vals = {
                'rr': sp.latex(rr),
                'cumulant': sp.latex(cumulant),
                'eq_value': sp.latex(eq_value),
                'nb': 'style="border:none"',
            }
            order = get_order(cumulant)
            if order <= 1:
                vals['cumulant'] += ' (central moment)'
                if order == 1 and self.force_model_rr_override:
                    vals['rr'] += ' (overridden by force model)'
            html += """<tr {nb}>
                            <td {nb}>${cumulant}$</td>
                            <td {nb}>${eq_value}$</td>
                            <td {nb}>${rr}$</td>
                         </tr>\n""".format(**vals)

        html += "</table>"
        return html

    #   ----------------------- Overridden Abstract Members --------------------------

    @property
    def weights(self):
        """Returns a sequence of weights, one for each lattice direction"""
        if self._weights is None:
            self._weights = self._compute_weights()
        return self._weights

    def override_weights(self, weights):
        assert len(weights) == len(self.stencil)
        self._weights = weights

    def get_equilibrium(self, conserved_quantity_equations=None, subexpressions=False, pre_simplification=False,
                        keep_cqc_subexpressions=True):
        """Returns equation collection, to compute equilibrium values.
        The equations have the post collision symbols as left hand sides and are
        functions of the conserved quantities

        Args:
            conserved_quantity_equations: equations to compute conserved quantities.
            subexpressions: if set to false all subexpressions of the equilibrium assignments are plugged
                            into the main assignments
            pre_simplification: with or without pre_simplifications for the calculation of the collision
            keep_cqc_subexpressions: if equilibrium is returned without subexpressions keep_cqc_subexpressions
                                     determines if also subexpressions to calculate conserved quantities should be
                                     plugged into the main assignments
        """
        r_info_dict = {c: RelaxationInfo(info.equilibrium_value, sp.Integer(1))
                       for c, info in self.relaxation_info_dict.items()}
        ac = self._centered_cumulant_collision_rule(
            r_info_dict, conserved_quantity_equations, pre_simplification, include_galilean_correction=False)
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

    def get_collision_rule(self, conserved_quantity_equations=None, pre_simplification=False):
        """Returns an LbmCollisionRule i.e. an equation collection with a reference to the method.
        This collision rule defines the collision operator."""
        return self._centered_cumulant_collision_rule(
            self.relaxation_info_dict, conserved_quantity_equations, pre_simplification, True,
            symbolic_relaxation_rates=True)

    #   ------------------------------- Internals --------------------------------------------

    def _bound_symbols_cqc(self, conserved_quantity_equations=None):
        f = self.pre_collision_pdf_symbols
        cqe = conserved_quantity_equations

        if cqe is None:
            cqe = self._cqc.equilibrium_input_equations_from_pdfs(f, False)

        return cqe.bound_symbols

    def _compute_weights(self):
        bg = self.equilibrium_distribution.background_distribution
        assert bg is not None, "Could not compute weights, since no background distribution is given."
        if bg.discrete_populations is not None:
            #   Compute lattice weights as the discrete populations of the background distribution ...
            weights = bg.discrete_populations
        else:
            #   or, if those are not available, by moment matching.
            moments = self.cumulants
            mm_inv = moment_matrix(moments, self.stencil).inv()
            bg_moments = bg.moments(moments)
            weights = (mm_inv * sp.Matrix(bg_moments)).expand()

        for w in weights:
            assert is_constant(w)

        return [w for w in weights]

    def _centered_cumulant_collision_rule(self, cumulant_to_relaxation_info_dict,
                                          conserved_quantity_equations=None,
                                          pre_simplification=False,
                                          include_force_terms=False,
                                          include_galilean_correction=True,
                                          symbolic_relaxation_rates=False):

        # Filter out JobLib warnings. They are not usefull for use:
        # https://github.com/joblib/joblib/issues/683
        filterwarnings("ignore", message="Persisting input arguments took")

        stencil = self.stencil
        f = self.pre_collision_pdf_symbols
        density = self.zeroth_order_equilibrium_moment_symbol
        velocity = self.first_order_equilibrium_moment_symbols
        cqe = conserved_quantity_equations

        relaxation_info_dict = dict()
        subexpressions_relaxation_rates = []
        if symbolic_relaxation_rates:
            subexpressions_relaxation_rates, sd = self._generate_symbolic_relaxation_matrix()
            for i, cumulant in enumerate(cumulant_to_relaxation_info_dict):
                relaxation_info_dict[cumulant] = RelaxationInfo(cumulant_to_relaxation_info_dict[cumulant][0],
                                                                sd[i, i])
        else:
            relaxation_info_dict = cumulant_to_relaxation_info_dict

        if cqe is None:
            cqe = self._cqc.equilibrium_input_equations_from_pdfs(f, False)

        forcing_subexpressions = AssignmentCollection([])
        if self._force_model is not None:
            forcing_subexpressions = AssignmentCollection(self._force_model.subs_dict_force)

        #   See if a background shift is necessary
        if self._zero_centered:
            assert not self._equilibrium.deviation_only
            background_distribution = self._equilibrium.background_distribution
            assert background_distribution is not None
        else:
            background_distribution = None

        #   1) Extract Monomial Cumulants for the higher-order polynomials
        polynomial_cumulants = relaxation_info_dict.keys()
        polynomial_cumulants = sorted(list(polynomial_cumulants), key=moment_sort_key)
        higher_order_polynomials = [p for p in polynomial_cumulants if get_order(p) > 1]
        monomial_cumulants = sorted(list(extract_monomials(
            higher_order_polynomials, dim=self.dim)), key=exponent_tuple_sort_key)

        #   2) Get Forward and Backward Transformations between central moment and cumulant space,
        #      and find required central moments
        k_to_c_transform = self._cumulant_transform_class(stencil, monomial_cumulants, density, velocity)
        k_to_c_eqs = cached_forward_transform(k_to_c_transform, simplification=pre_simplification)
        c_post_to_k_post_eqs = cached_backward_transform(
            k_to_c_transform, simplification=pre_simplification, omit_conserved_moments=True)
        central_moments = k_to_c_transform.required_central_moments
        assert len(central_moments) == stencil.Q, 'Number of required central moments must match stencil size.'

        #   3) Get Forward Transformation from PDFs to central moments
        pdfs_to_k_transform = self._central_moment_transform_class(
            stencil, None, density, velocity, moment_exponents=central_moments, conserved_quantity_equations=cqe,
            background_distribution=background_distribution)
        pdfs_to_k_eqs = cached_forward_transform(
            pdfs_to_k_transform, f, simplification=pre_simplification, return_monomials=True)

        #   4) Add relaxation rules for lower order moments
        lower_order_moments = moments_up_to_order(1, dim=self.dim)
        lower_order_moment_symbols = [statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT, exp)
                                      for exp in lower_order_moments]

        lower_order_relaxation_infos = [relaxation_info_dict[exponent_to_polynomial_representation(e)]
                                        for e in lower_order_moments]
        lower_order_relaxation_rates = [info.relaxation_rate for info in lower_order_relaxation_infos]
        lower_order_equilibrium = [info.equilibrium_value for info in lower_order_relaxation_infos]

        lower_order_moment_collision_eqs = relax_lower_order_central_moments(
            lower_order_moments, tuple(lower_order_moment_symbols),
            tuple(lower_order_relaxation_rates), tuple(lower_order_equilibrium))

        #   5) Add relaxation rules for higher-order, polynomial cumulants
        poly_relaxation_infos = [relaxation_info_dict[c] for c in higher_order_polynomials]
        poly_relaxation_rates = [info.relaxation_rate for info in poly_relaxation_infos]
        poly_equilibrium = [info.equilibrium_value for info in poly_relaxation_infos]

        if self._galilean_correction and include_galilean_correction:
            galilean_correction_terms = get_galilean_correction_terms(
                relaxation_info_dict, density, velocity)
        else:
            galilean_correction_terms = None

        cumulant_collision_eqs = relax_polynomial_cumulants(
            tuple(monomial_cumulants), tuple(higher_order_polynomials),
            tuple(poly_relaxation_rates), tuple(poly_equilibrium),
            pre_simplification,
            galilean_correction_terms=galilean_correction_terms)

        #   6) Get backward transformation from central moments to PDFs
        d = self.post_collision_pdf_symbols
        k_post_to_pdfs_eqs = cached_backward_transform(
            pdfs_to_k_transform, d, simplification=pre_simplification, start_from_monomials=True)

        #   7) That's all. Now, put it all together.
        all_acs = [] if pdfs_to_k_transform.absorbs_conserved_quantity_equations else [cqe]
        subexpressions_relaxation_rates = AssignmentCollection(subexpressions_relaxation_rates)
        all_acs += [subexpressions_relaxation_rates, forcing_subexpressions, pdfs_to_k_eqs, k_to_c_eqs,
                    lower_order_moment_collision_eqs, cumulant_collision_eqs, c_post_to_k_post_eqs]
        subexpressions = [ac.all_assignments for ac in all_acs]
        subexpressions += k_post_to_pdfs_eqs.subexpressions
        main_assignments = k_post_to_pdfs_eqs.main_assignments

        #   8) Maybe add forcing terms if CenteredCumulantForceModel was not used
        if self._force_model is not None and \
                not isinstance(self._force_model, CenteredCumulantForceModel) and include_force_terms:
            force_model_terms = self._force_model(self)
            force_term_symbols = sp.symbols(f"forceTerm_:{len(force_model_terms)}")
            force_subexpressions = [Assignment(sym, force_model_term)
                                    for sym, force_model_term in zip(force_term_symbols, force_model_terms)]
            subexpressions += force_subexpressions
            main_assignments = [Assignment(eq.lhs, eq.rhs + force_term_symbol)
                                for eq, force_term_symbol in zip(main_assignments, force_term_symbols)]

        #   Aaaaaand we're done.
        return LbmCollisionRule(self, main_assignments, subexpressions)
