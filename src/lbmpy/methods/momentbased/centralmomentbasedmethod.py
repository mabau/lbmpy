import sympy as sp
from collections import OrderedDict
from typing import Set

from pystencils import Assignment, AssignmentCollection
from pystencils.sympyextensions import is_constant
from pystencils.simp import apply_to_all_assignments

from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.moment_transforms import BinomialChimeraTransform

from lbmpy.moments import MOMENT_SYMBOLS, moment_matrix, set_up_shift_matrix


def relax_central_moments(pre_collision_symbols, post_collision_symbols,
                          relaxation_rates, equilibrium_values,
                          force_terms):
    equilibrium_vec = sp.Matrix(equilibrium_values)
    moment_vec = sp.Matrix(pre_collision_symbols)
    relaxation_matrix = sp.diag(*relaxation_rates)
    moment_vec = moment_vec + relaxation_matrix * (equilibrium_vec - moment_vec) + force_terms
    main_assignments = [Assignment(s, eq) for s, eq in zip(post_collision_symbols, moment_vec)]

    return AssignmentCollection(main_assignments)


#   =============================== LB Method Implementation ===========================================================


class CentralMomentBasedLbMethod(AbstractLbMethod):
    """
    Central Moment based LBM is a class to represent the single (SRT), two (TRT) and multi relaxation time (MRT)
    methods, where the collision is performed in the central moment space.
    These methods work by transforming the pdfs into moment space using a linear transformation and then shiftig
    them into the central moment space. In the central moment space each component (moment) is relaxed to an
    equilibrium moment by a certain relaxation rate. These equilibrium moments can e.g. be determined by taking the
    equilibrium moments of the continuous Maxwellian.

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
        central_moment_transform_class: transformation class to transform PDFs to central moment space (subclass of 
                                        :class:`lbmpy.moment_transforms.AbstractCentralMomentTransform`)
    """

    def __init__(self, stencil, equilibrium, relaxation_dict,
                 conserved_quantity_computation=None,
                 force_model=None, zero_centered=False,
                 central_moment_transform_class=BinomialChimeraTransform):
        assert isinstance(conserved_quantity_computation, AbstractConservedQuantityComputation)
        super(CentralMomentBasedLbMethod, self).__init__(stencil)

        self._equilibrium = equilibrium
        self._relaxation_dict = OrderedDict(relaxation_dict)
        self._cqc = conserved_quantity_computation
        self._force_model = force_model
        self._zero_centered = zero_centered
        self._weights = None
        self._central_moment_transform_class = central_moment_transform_class

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
    def central_moment_transform_class(self):
        """The transform class (subclass of :class:`lbmpy.moment_transforms.AbstractCentralMomentTransform` defining the
        transformation of populations to central moment space."""
        return self._central_moment_transform_class

    @property
    def moments(self):
        """Central moments relaxed by this method."""
        return tuple(self._relaxation_dict.keys())

    @property
    def moment_equilibrium_values(self):
        """Equilibrium values of this method's :attr:`moments`."""
        return self._equilibrium.central_moments(self.moments, self.first_order_equilibrium_moment_symbols)

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
        """Returns a sequence of weights, one for each lattice direction"""
        if self._weights is None:
            self._weights = self._compute_weights()
        return self._weights

    def set_zeroth_moment_relaxation_rate(self, relaxation_rate):
        e = sp.Rational(1, 1)
        self._relaxation_dict[e] = relaxation_rate

    def set_first_moment_relaxation_rate(self, relaxation_rate):
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._relaxation_dict, "First moments are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            self._relaxation_dict[e] = relaxation_rate

    def set_conserved_moments_relaxation_rate(self, relaxation_rate):
        self.set_zeroth_moment_relaxation_rate(relaxation_rate)
        self.set_first_moment_relaxation_rate(relaxation_rate)

    def set_force_model(self, force_model):
        self._force_model = force_model

    @property
    def moment_matrix(self) -> sp.Matrix:
        return moment_matrix(self.moments, self.stencil)

    @property
    def shift_matrix(self) -> sp.Matrix:
        return set_up_shift_matrix(self.moments, self.stencil)

    @property
    def relaxation_matrix(self) -> sp.Matrix:
        d = sp.zeros(len(self.relaxation_rates))
        for i in range(0, len(self.relaxation_rates)):
            d[i, i] = self.relaxation_rates[i]
        return d

    def __getstate__(self):
        # Workaround for a bug in joblib
        self._moment_to_relaxation_info_dict_to_pickle = [i for i in self._relaxation_dict.items()]
        return self.__dict__

    def _repr_html_(self):
        def stylized_bool(b):
            return "&#10003;" if b else "&#10007;"

        html = f"""
        <table style="border:none; width: 100%">
            <tr>
                <th colspan="3" style="text-align: left">
                    Central-Moment-Based Method
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
                <th>Central Moment</th>
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

    #   ----------------------- Overridden Abstract Members --------------------------

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
        _, d = self._generate_symbolic_relaxation_matrix()
        rr_sub_expressions = set([Assignment(d[i, i], sp.Integer(1)) for i in range(len(self.relaxation_rates))])
        r_info_dict = OrderedDict({c: RelaxationInfo(info.equilibrium_value, sp.Integer(1))
                                   for c, info in self.relaxation_info_dict.items()})
        ac = self._central_moment_collision_rule(moment_to_relaxation_info_dict=r_info_dict,
                                                 conserved_quantity_equations=conserved_quantity_equations,
                                                 pre_simplification=pre_simplification,
                                                 include_force_terms=include_force_terms,
                                                 symbolic_relaxation_rates=False)
        ac.subexpressions = list(rr_sub_expressions) + ac.subexpressions
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
                           pre_simplification: bool = False) -> LbmCollisionRule:
        """Returns an LbmCollisionRule i.e. an equation collection with a reference to the method.
        This collision rule defines the collision operator."""
        return self._central_moment_collision_rule(moment_to_relaxation_info_dict=self.relaxation_info_dict,
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
            mm_inv = self.moment_matrix.inv()
            bg_moments = bg.moments(self.moments)
            weights = (mm_inv * sp.Matrix(bg_moments)).expand()
        
        for w in weights:
            assert is_constant(w)

        return [w for w in weights]

    def _central_moment_collision_rule(self, moment_to_relaxation_info_dict: OrderedDict,
                                       conserved_quantity_equations: AssignmentCollection = None,
                                       pre_simplification: bool = False,
                                       include_force_terms: bool = False,
                                       symbolic_relaxation_rates: bool = False) -> LbmCollisionRule:
        stencil = self.stencil
        f = self.pre_collision_pdf_symbols
        density = self.zeroth_order_equilibrium_moment_symbol
        velocity = self.first_order_equilibrium_moment_symbols
        cqe = conserved_quantity_equations

        relaxation_info_dict = dict()
        subexpressions_relaxation_rates = []
        if symbolic_relaxation_rates:
            subexpressions_relaxation_rates, sd = self._generate_symbolic_relaxation_matrix()
            for i, moment in enumerate(moment_to_relaxation_info_dict):
                relaxation_info_dict[moment] = RelaxationInfo(moment_to_relaxation_info_dict[moment][0], sd[i, i])
        else:
            relaxation_info_dict = moment_to_relaxation_info_dict

        if cqe is None:
            cqe = self._cqc.equilibrium_input_equations_from_pdfs(f, False)

        forcing_subexpressions = AssignmentCollection([])
        moment_space_forcing = False
        if self._force_model is not None:
            if include_force_terms:
                moment_space_forcing = self.force_model.has_central_moment_space_forcing
            forcing_subexpressions = AssignmentCollection(self._force_model.subs_dict_force)
        else:
            include_force_terms = False

        #   See if a background shift is necessary
        if self._zero_centered and not self._equilibrium.deviation_only:
            background_distribution = self._equilibrium.background_distribution
            assert background_distribution is not None
        else:
            background_distribution = None

        #   1) Get Forward Transformation from PDFs to central moments
        pdfs_to_c_transform = self.central_moment_transform_class(
            stencil, self.moments, density, velocity, conserved_quantity_equations=cqe,
            background_distribution=background_distribution)
        pdfs_to_c_eqs = pdfs_to_c_transform.forward_transform(f, simplification=pre_simplification)

        #   2) Collision
        k_pre = pdfs_to_c_transform.pre_collision_symbols
        k_post = pdfs_to_c_transform.post_collision_symbols

        relaxation_infos = [relaxation_info_dict[m] for m in self.moments]
        relaxation_rates = [info.relaxation_rate for info in relaxation_infos]
        equilibrium_value = [info.equilibrium_value for info in relaxation_infos]

        if moment_space_forcing:
            force_model_terms = self._force_model.central_moment_space_forcing(self)
        else:
            force_model_terms = sp.Matrix([0] * stencil.Q)

        collision_eqs = relax_central_moments(k_pre, k_post, tuple(relaxation_rates),
                                              tuple(equilibrium_value), force_terms=force_model_terms)

        #   3) Get backward transformation from central moments to PDFs
        post_collision_values = self.post_collision_pdf_symbols
        c_post_to_pdfs_eqs = pdfs_to_c_transform.backward_transform(post_collision_values,
                                                                    simplification=pre_simplification)

        #   4) Now, put it all together.
        all_acs = [] if pdfs_to_c_transform.absorbs_conserved_quantity_equations else [cqe]
        subexpressions_relaxation_rates = AssignmentCollection(subexpressions_relaxation_rates)
        all_acs += [subexpressions_relaxation_rates, forcing_subexpressions, pdfs_to_c_eqs, collision_eqs]
        subexpressions = [ac.all_assignments for ac in all_acs]
        subexpressions += c_post_to_pdfs_eqs.subexpressions
        main_assignments = c_post_to_pdfs_eqs.main_assignments

        simplification_hints = cqe.simplification_hints.copy()
        simplification_hints.update(self._cqc.defined_symbols())
        simplification_hints['relaxation_rates'] = [rr for rr in self.relaxation_rates]

        #   5) Maybe add forcing terms.
        if include_force_terms and not moment_space_forcing:
            force_model_terms = self._force_model(self)
            force_term_symbols = sp.symbols(f"forceTerm_:{len(force_model_terms)}")
            force_subexpressions = [Assignment(sym, force_model_term)
                                    for sym, force_model_term in zip(force_term_symbols, force_model_terms)]
            subexpressions += force_subexpressions
            main_assignments = [Assignment(eq.lhs, eq.rhs + force_term_symbol)
                                for eq, force_term_symbol in zip(main_assignments, force_term_symbols)]
            simplification_hints['force_terms'] = force_term_symbols

        return LbmCollisionRule(self, main_assignments, subexpressions)
