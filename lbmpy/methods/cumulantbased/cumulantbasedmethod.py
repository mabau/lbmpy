import sympy as sp
from collections import OrderedDict
from typing import Set
from warnings import filterwarnings

from pystencils import Assignment, AssignmentCollection
from pystencils.sympyextensions import is_constant
from pystencils.simp import apply_to_all_assignments

from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation

from lbmpy.moments import moment_matrix, MOMENT_SYMBOLS, statistical_quantity_symbol

from lbmpy.moment_transforms import (
    PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT, POST_COLLISION_MONOMIAL_CENTRAL_MOMENT,
    CentralMomentsToCumulantsByGeneratingFunc,
    BinomialChimeraTransform)


class CumulantBasedLbMethod(AbstractLbMethod):
    """
    This class implements cumulant-based lattice boltzmann methods which relax all the non-conserved quantities
    as either monomial or polynomial cumulants. It is mostly inspired by the work presented in :cite:`geier2015`.

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
        central_moment_transform_class: transformation class to transform PDFs to central moment space (subclass of 
                                        :class:`lbmpy.moment_transforms.AbstractCentralMomentTransform`)
        cumulant_transform_class: transform class to get from the central moment space to the cumulant space
    """

    def __init__(self, stencil, equilibrium, relaxation_dict,
                 conserved_quantity_computation=None,
                 force_model=None, zero_centered=False,
                 central_moment_transform_class=BinomialChimeraTransform,
                 cumulant_transform_class=CentralMomentsToCumulantsByGeneratingFunc):
        assert isinstance(conserved_quantity_computation,
                          AbstractConservedQuantityComputation)
        super(CumulantBasedLbMethod, self).__init__(stencil)

        if force_model is not None:
            if not force_model.has_symmetric_central_moment_forcing:
                raise ValueError(f"Force model {force_model} does not offer symmetric central moment forcing.")

        self._equilibrium = equilibrium
        self._relaxation_dict = OrderedDict(relaxation_dict)
        self._cqc = conserved_quantity_computation
        self._force_model = force_model
        self._zero_centered = zero_centered
        self._weights = None
        self._cumulant_transform_class = cumulant_transform_class
        self._central_moment_transform_class = central_moment_transform_class

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
        return self._equilibrium.cumulants(self.cumulants, rescale=True)

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

    def set_zeroth_moment_relaxation_rate(self, relaxation_rate):
        e = sp.Rational(1, 1)
        self._relaxation_dict[e] = relaxation_rate

    def set_first_moment_relaxation_rate(self, relaxation_rate):
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._relaxation_dict, \
                "First cumulants are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            self._relaxation_dict[e] = relaxation_rate

    def set_conserved_moments_relaxation_rate(self, relaxation_rate):
        self.set_zeroth_moment_relaxation_rate(relaxation_rate)
        self.set_first_moment_relaxation_rate(relaxation_rate)

    def set_force_model(self, force_model):
        if not force_model.has_symmetric_central_moment_forcing:
            raise ValueError("Given force model does not support symmetric central moment forcing.")
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

    def get_equilibrium(self, conserved_quantity_equations: AssignmentCollection = None, subexpressions: bool = False,
                        pre_simplification: bool = False, keep_cqc_subexpressions: bool = True,
                        include_force_terms: bool = False) -> LbmCollisionRule:
        """Returns equation collection, to compute equilibrium values.
        The equations have the post collision symbols as left-hand sides and are
        functions of the conserved quantities

        Args:
            conserved_quantity_equations: equations to compute conserved quantities.
            subexpressions: if set to false all subexpressions of the equilibrium assignments are plugged
                            into the main assignments
            pre_simplification: with or without pre_simplifications for the calculation of the collision
            keep_cqc_subexpressions: if equilibrium is returned without subexpressions keep_cqc_subexpressions
                                     determines if also subexpressions to calculate conserved quantities should be
                                     plugged into the main assignments
            include_force_terms: if set to True the equilibrium is shifted by forcing terms coming from the force model
                                 of the method.
        """
        r_info_dict = OrderedDict({c: RelaxationInfo(info.equilibrium_value, sp.Integer(1))
                                   for c, info in self.relaxation_info_dict.items()})
        ac = self._centered_cumulant_collision_rule(cumulant_to_relaxation_info_dict=r_info_dict,
                                                    conserved_quantity_equations=conserved_quantity_equations,
                                                    pre_simplification=pre_simplification,
                                                    include_force_terms=include_force_terms,
                                                    symbolic_relaxation_rates=False)

        expand_all_assignments = apply_to_all_assignments(sp.expand)

        if not subexpressions:
            if keep_cqc_subexpressions:
                bs = self._bound_symbols_cqc(conserved_quantity_equations)
                ac = expand_all_assignments(ac.new_without_subexpressions(subexpressions_to_keep=bs))
                return ac.new_without_unused_subexpressions()
            else:
                ac = expand_all_assignments(ac.new_without_subexpressions())
                return ac.new_without_unused_subexpressions()
        else:
            return ac.new_without_unused_subexpressions()

    def get_equilibrium_terms(self) -> sp.Matrix:
        equilibrium = self.get_equilibrium()
        return sp.Matrix([eq.rhs for eq in equilibrium.main_assignments])

    def get_collision_rule(self, conserved_quantity_equations: AssignmentCollection = None,
                           pre_simplification: bool = False) -> AssignmentCollection:
        """Returns an LbmCollisionRule i.e. an equation collection with a reference to the method.
        This collision rule defines the collision operator."""
        return self._centered_cumulant_collision_rule(cumulant_to_relaxation_info_dict=self.relaxation_info_dict,
                                                      conserved_quantity_equations=conserved_quantity_equations,
                                                      pre_simplification=pre_simplification,
                                                      include_force_terms=True, symbolic_relaxation_rates=True)

    #   ------------------------------- Internals --------------------------------------------

    def _bound_symbols_cqc(self, conserved_quantity_equations: AssignmentCollection = None) -> Set[sp.Symbol]:
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

    def _centered_cumulant_collision_rule(self, cumulant_to_relaxation_info_dict: OrderedDict,
                                          conserved_quantity_equations: AssignmentCollection = None,
                                          pre_simplification: bool = False,
                                          include_force_terms: bool = False,
                                          symbolic_relaxation_rates: bool = False) -> LbmCollisionRule:

        # Filter out JobLib warnings. They are not usefull for use:
        # https://github.com/joblib/joblib/issues/683
        filterwarnings("ignore", message="Persisting input arguments took")

        stencil = self.stencil
        f = self.pre_collision_pdf_symbols
        density = self.zeroth_order_equilibrium_moment_symbol
        velocity = self.first_order_equilibrium_moment_symbols
        cqe = conserved_quantity_equations

        polynomial_cumulants = self.cumulants

        rrs = [cumulant_to_relaxation_info_dict[c].relaxation_rate for c in polynomial_cumulants]
        if symbolic_relaxation_rates:
            subexpressions_relaxation_rates, d = self._generate_symbolic_relaxation_matrix(relaxation_rates=rrs)
        else:
            subexpressions_relaxation_rates = []
            d = sp.zeros(len(rrs))
            for i, w in enumerate(rrs):
                # note that 0.0 is converted to sp.Zero here. It is not possible to prevent this.
                d[i, i] = w

        if cqe is None:
            cqe = self._cqc.equilibrium_input_equations_from_pdfs(f, False)

        forcing_subexpressions = AssignmentCollection([])
        if self._force_model is not None:
            forcing_subexpressions = AssignmentCollection(self._force_model.subs_dict_force)
        else:
            include_force_terms = False

        #   See if a background shift is necessary
        if self._zero_centered:
            assert not self._equilibrium.deviation_only
            background_distribution = self._equilibrium.background_distribution
            assert background_distribution is not None
        else:
            background_distribution = None

        #   1) Get Forward and Backward Transformations between central moment and cumulant space,
        #      and find required central moments
        k_to_c_transform = self._cumulant_transform_class(stencil, polynomial_cumulants, density, velocity)
        k_to_c_eqs = k_to_c_transform.forward_transform(simplification=pre_simplification)
        c_post_to_k_post_eqs = k_to_c_transform.backward_transform(simplification=pre_simplification)

        C_pre = k_to_c_transform.pre_collision_symbols
        C_post = k_to_c_transform.post_collision_symbols
        central_moments = k_to_c_transform.required_central_moments

        #   2) Get Forward Transformation from PDFs to central moments
        pdfs_to_k_transform = self._central_moment_transform_class(
            stencil, None, density, velocity, moment_exponents=central_moments, conserved_quantity_equations=cqe,
            background_distribution=background_distribution)
        pdfs_to_k_eqs = pdfs_to_k_transform.forward_transform(
            f, simplification=pre_simplification, return_monomials=True)

        #   3) Symmetric forcing
        if include_force_terms:
            force_before, force_after = self._force_model.symmetric_central_moment_forcing(self, central_moments)
            k_asms_dict = pdfs_to_k_eqs.main_assignments_dict
            for cm_exp, kappa_f in zip(central_moments, force_before):
                cm_symb = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT, cm_exp)
                k_asms_dict[cm_symb] += kappa_f
            pdfs_to_k_eqs.set_main_assignments_from_dict(k_asms_dict)

            k_post_asms_dict = c_post_to_k_post_eqs.main_assignments_dict
            for cm_exp, kappa_f in zip(central_moments, force_after):
                cm_symb = statistical_quantity_symbol(POST_COLLISION_MONOMIAL_CENTRAL_MOMENT, cm_exp)
                k_post_asms_dict[cm_symb] += kappa_f
            c_post_to_k_post_eqs.set_main_assignments_from_dict(k_post_asms_dict)

        #   4) Add relaxation rules for polynomial cumulants
        C_eq = sp.Matrix(self.cumulant_equilibrium_values)

        C_pre_vec = sp.Matrix(C_pre)
        collision_rule = C_pre_vec + d @ (C_eq - C_pre_vec)
        cumulant_collision_eqs = [Assignment(lhs, rhs) for lhs, rhs in zip(C_post, collision_rule)]
        cumulant_collision_eqs = AssignmentCollection(cumulant_collision_eqs)

        #   5) Get backward transformation from central moments to PDFs
        d = self.post_collision_pdf_symbols
        k_post_to_pdfs_eqs = pdfs_to_k_transform.backward_transform(
            d, simplification=pre_simplification, start_from_monomials=True)

        #   6) That's all. Now, put it all together.
        all_acs = [] if pdfs_to_k_transform.absorbs_conserved_quantity_equations else [cqe]
        subexpressions_relaxation_rates = AssignmentCollection(subexpressions_relaxation_rates)
        all_acs += [subexpressions_relaxation_rates, forcing_subexpressions, pdfs_to_k_eqs, k_to_c_eqs,
                    cumulant_collision_eqs, c_post_to_k_post_eqs]
        subexpressions = [ac.all_assignments for ac in all_acs]
        subexpressions += k_post_to_pdfs_eqs.subexpressions
        main_assignments = k_post_to_pdfs_eqs.main_assignments

        simplification_hints = cqe.simplification_hints.copy()
        simplification_hints.update(self._cqc.defined_symbols())
        simplification_hints['relaxation_rates'] = [rr for rr in self.relaxation_rates]
        simplification_hints['post_collision_monomial_central_moments'] = \
            pdfs_to_k_transform.post_collision_monomial_symbols

        #   Aaaaaand we're done.
        return LbmCollisionRule(self, main_assignments, subexpressions, simplification_hints)
