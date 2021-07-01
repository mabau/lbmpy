import sympy as sp
from collections import OrderedDict

from pystencils import Assignment, AssignmentCollection

from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.methods.momentbased.moment_transforms import (FastCentralMomentTransform,
                                                         PRE_COLLISION_CENTRAL_MOMENT, POST_COLLISION_CENTRAL_MOMENT)

from lbmpy.moments import (polynomial_to_exponent_representation, MOMENT_SYMBOLS, moment_matrix, set_up_shift_matrix,
                           statistical_quantity_symbol)


def relax_central_moments(moment_indices, pre_collision_values,
                          relaxation_rates, equilibrium_values,
                          post_collision_base=POST_COLLISION_CENTRAL_MOMENT):

    post_collision_symbols = [sp.Symbol(f'{post_collision_base}_{"".join(str(i) for i in m)}') for m in moment_indices]
    equilibrium_vec = sp.Matrix(equilibrium_values)
    moment_vec = sp.Matrix(pre_collision_values)
    relaxation_matrix = sp.diag(*relaxation_rates)
    moment_vec = moment_vec + relaxation_matrix * (equilibrium_vec - moment_vec)
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

    Args:
        stencil: see :func:`lbmpy.stencils.get_stencil`
        moment_to_relaxation_info_dict: a dictionary mapping moments in either tuple or polynomial formulation
                                        to a RelaxationInfo, which consists of the corresponding equilibrium moment
                                        and a relaxation rate
        conserved_quantity_computation: instance of :class:`lbmpy.methods.AbstractConservedQuantityComputation`.
                                        This determines how conserved quantities are computed, and defines
                                        the symbols used in the equilibrium moments like e.g. density and velocity
        force_model: force model instance, or None if no forcing terms are required
        central_moment_transform_class: class to transform PDFs to the central moment space.
    """

    def __init__(self, stencil, moment_to_relaxation_info_dict, conserved_quantity_computation=None, force_model=None,
                 central_moment_transform_class=FastCentralMomentTransform):
        assert isinstance(conserved_quantity_computation, AbstractConservedQuantityComputation)
        super(CentralMomentBasedLbMethod, self).__init__(stencil)

        self._force_model = force_model
        self._moment_to_relaxation_info_dict = OrderedDict(moment_to_relaxation_info_dict.items())
        self._conserved_quantity_computation = conserved_quantity_computation
        self._weights = None
        self._central_moment_transform_class = central_moment_transform_class

    @property
    def central_moment_transform_class(self):
        return self._central_moment_transform_class

    @property
    def moments(self):
        return tuple(self._moment_to_relaxation_info_dict.keys())

    @property
    def moment_equilibrium_values(self):
        return tuple([e.equilibrium_value for e in self._moment_to_relaxation_info_dict.values()])

    @property
    def first_order_equilibrium_moment_symbols(self, ):
        return self._conserved_quantity_computation.first_order_moment_symbols

    @property
    def force_model(self):
        return self._force_model

    @property
    def relaxation_info_dict(self):
        return self._moment_to_relaxation_info_dict

    @property
    def relaxation_rates(self):
        return tuple([e.relaxation_rate for e in self._moment_to_relaxation_info_dict.values()])

    @property
    def zeroth_order_equilibrium_moment_symbol(self, ):
        return self._conserved_quantity_computation.zeroth_order_moment_symbol

    def set_zeroth_moment_relaxation_rate(self, relaxation_rate):
        e = sp.Rational(1, 1)
        prev_entry = self._moment_to_relaxation_info_dict[e]
        new_entry = RelaxationInfo(prev_entry[0], relaxation_rate)
        self._moment_to_relaxation_info_dict[e] = new_entry

    def set_first_moment_relaxation_rate(self, relaxation_rate):
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._moment_to_relaxation_info_dict, "First moments are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            prev_entry = self._moment_to_relaxation_info_dict[e]
            new_entry = RelaxationInfo(prev_entry[0], relaxation_rate)
            self._moment_to_relaxation_info_dict[e] = new_entry

    def set_conserved_moments_relaxation_rate(self, relaxation_rate):
        self.set_zeroth_moment_relaxation_rate(relaxation_rate)
        self.set_first_moment_relaxation_rate(relaxation_rate)

    def set_force_model(self, force_model):
        self._force_model = force_model

    @property
    def moment_matrix(self):
        return moment_matrix(self.moments, self.stencil)

    @property
    def shift_matrix(self):
        return set_up_shift_matrix(self.moments, self.stencil)

    @property
    def relaxation_matrix(self):
        d = sp.zeros(len(self.relaxation_rates))
        for i in range(0, len(self.relaxation_rates)):
            d[i, i] = self.relaxation_rates[i]
        return d

    def __getstate__(self):
        # Workaround for a bug in joblib
        self._moment_to_relaxation_info_dict_to_pickle = [i for i in self._moment_to_relaxation_info_dict.items()]
        return self.__dict__

    def _repr_html_(self):
        table = """
        <table style="border:none; width: 100%">
            <tr {nb}>
                <th {nb} >Central Moment</th>
                <th {nb} >Eq. Value </th>
                <th {nb} >Relaxation Rate</th>
            </tr>
            {content}
        </table>
        """
        content = ""
        for moment, (eq_value, rr) in self._moment_to_relaxation_info_dict.items():
            vals = {
                'rr': f"${sp.latex(rr)}$",
                'cumulant': f"${sp.latex(moment)}$",
                'eq_value': f"${sp.latex(eq_value)}$",
                'nb': 'style="border:none"',
            }
            content += """<tr {nb}>
                            <td {nb}>{cumulant}</td>
                            <td {nb}>{eq_value}</td>
                            <td {nb}>{rr}</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')

    #   ----------------------- Overridden Abstract Members --------------------------

    @property
    def conserved_quantity_computation(self):
        """Returns an instance of class :class:`lbmpy.methods.AbstractConservedQuantityComputation`"""
        return self._conserved_quantity_computation

    @property
    def weights(self):
        """Returns a sequence of weights, one for each lattice direction"""
        if self._weights is None:
            self._weights = self._compute_weights()
        return self._weights

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
        r_info_dict = {c: RelaxationInfo(info.equilibrium_value, 1)
                       for c, info in self._moment_to_relaxation_info_dict.items()}
        ac = self._central_moment_collision_rule(
            r_info_dict, conserved_quantity_equations, pre_simplification)
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
        return self._central_moment_collision_rule(
            self._moment_to_relaxation_info_dict, conserved_quantity_equations, pre_simplification, True)

    #   ------------------------------- Internals --------------------------------------------

    def _bound_symbols_cqc(self, conserved_quantity_equations=None):
        f = self.pre_collision_pdf_symbols
        cqe = conserved_quantity_equations

        if cqe is None:
            cqe = self._conserved_quantity_computation.equilibrium_input_equations_from_pdfs(f)

        return cqe.bound_symbols

    def _compute_weights(self):
        defaults = self._conserved_quantity_computation.default_values
        cqe = AssignmentCollection([Assignment(s, e) for s, e in defaults.items()])
        eq_ac = self.get_equilibrium(cqe, subexpressions=False, keep_cqc_subexpressions=False)

        weights = []
        for eq in eq_ac.main_assignments:
            value = eq.rhs.expand()
            assert len(value.atoms(sp.Symbol)) == 0, "Failed to compute weights"
            weights.append(value)
        return weights

    def _central_moment_collision_rule(self, moment_to_relaxation_info_dict,
                                       conserved_quantity_equations=None,
                                       pre_simplification=False,
                                       include_force_terms=False):
        stencil = self.stencil
        dim = len(self.stencil[0])
        f = self.pre_collision_pdf_symbols
        density = self.zeroth_order_equilibrium_moment_symbol
        velocity = self.first_order_equilibrium_moment_symbols
        cqe = conserved_quantity_equations

        if cqe is None:
            cqe = self._conserved_quantity_computation.equilibrium_input_equations_from_pdfs(f)

        moments_as_exponents = list()
        for moment in self.moments:
            _, exponent_tuple = polynomial_to_exponent_representation(moment)[0]
            moments_as_exponents.append(exponent_tuple[:dim])

        #   1) Get Forward Transformation from PDFs to central moments
        pdfs_to_k_transform = self._central_moment_transform_class(
            stencil, moments_as_exponents, density, velocity, conserved_quantity_equations=cqe)
        pdfs_to_k_eqs = pdfs_to_k_transform.forward_transform(f, simplification=pre_simplification)

        #   2) Collision
        moment_symbols = [statistical_quantity_symbol(PRE_COLLISION_CENTRAL_MOMENT, exp)
                          for exp in moments_as_exponents]

        relaxation_infos = [moment_to_relaxation_info_dict[m] for m in self.moments]
        relaxation_rates = [info.relaxation_rate for info in relaxation_infos]
        equilibrium_value = [info.equilibrium_value for info in relaxation_infos]

        collision_eqs = relax_central_moments(
            moments_as_exponents, tuple(moment_symbols),
            tuple(relaxation_rates), tuple(equilibrium_value))

        #   3) Get backward transformation from central moments to PDFs
        d = self.post_collision_pdf_symbols
        k_post_to_pdfs_eqs = pdfs_to_k_transform.backward_transform(d, simplification=pre_simplification)

        #   4) Now, put it all together.
        all_acs = [] if pdfs_to_k_transform.absorbs_conserved_quantity_equations else [cqe]
        all_acs += [pdfs_to_k_eqs, collision_eqs]
        subexpressions = [ac.all_assignments for ac in all_acs]
        subexpressions += k_post_to_pdfs_eqs.subexpressions
        main_assignments = k_post_to_pdfs_eqs.main_assignments

        #   5) Maybe add forcing terms.
        if self._force_model is not None and include_force_terms:
            force_model_terms = self._force_model(self)
            force_term_symbols = sp.symbols(f"forceTerm_:{len(force_model_terms)}")
            force_subexpressions = [Assignment(sym, force_model_term)
                                    for sym, force_model_term in zip(force_term_symbols, force_model_terms)]
            subexpressions += force_subexpressions
            main_assignments = [Assignment(eq.lhs, eq.rhs + force_term_symbol)
                                for eq, force_term_symbol in zip(main_assignments, force_term_symbols)]

        return LbmCollisionRule(self, main_assignments, subexpressions)
