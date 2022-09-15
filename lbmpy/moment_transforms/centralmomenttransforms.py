from functools import partial
import sympy as sp

from pystencils import Assignment, AssignmentCollection
from pystencils.simp import (
    SimplificationStrategy, add_subexpressions_for_constants)
from pystencils.simp.assignment_collection import SymbolGen
from pystencils.sympyextensions import subs_additive, fast_subs

from lbmpy.moments import (
    moment_matrix, monomial_to_polynomial_transformation_matrix,
    set_up_shift_matrix, contained_moments, moments_up_to_order,
    moments_of_order,
    central_moment_reduced_monomial_to_polynomial_matrix)

from lbmpy.moments import statistical_quantity_symbol as sq_sym

from .abstractmomenttransform import (
    AbstractMomentTransform,
    PRE_COLLISION_CENTRAL_MOMENT, POST_COLLISION_CENTRAL_MOMENT,
    PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT, POST_COLLISION_MONOMIAL_CENTRAL_MOMENT
)

from .rawmomenttransforms import PdfsToMomentsByChimeraTransform


class AbstractCentralMomentTransform(AbstractMomentTransform):
    """Abstract base class for all transformations between population space
    and central-moment space."""

    def __init__(self, stencil, moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 pre_collision_symbol_base=PRE_COLLISION_CENTRAL_MOMENT,
                 post_collision_symbol_base=POST_COLLISION_CENTRAL_MOMENT,
                 pre_collision_monomial_symbol_base=PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT,
                 post_collision_monomial_symbol_base=POST_COLLISION_MONOMIAL_CENTRAL_MOMENT,
                 **kwargs):
        super(AbstractCentralMomentTransform, self).__init__(
            stencil, equilibrium_density, equilibrium_velocity,
            moment_polynomials=moment_polynomials,
            pre_collision_symbol_base=pre_collision_symbol_base,
            post_collision_symbol_base=post_collision_symbol_base,
            pre_collision_monomial_symbol_base=pre_collision_monomial_symbol_base,
            post_collision_monomial_symbol_base=post_collision_monomial_symbol_base,
            **kwargs
        )

        assert len(self.moment_polynomials) == self.q, 'Number of moments must match stencil'

    def _cm_background_shift(self, central_moments):
        if self.background_distribution is not None:
            shift = self.background_distribution.central_moments(central_moments, self.equilibrium_velocity)
        else:
            shift = (0,) * self.q
        return sp.Matrix(shift)
# end class AbstractRawMomentTransform


class PdfsToCentralMomentsByMatrix(AbstractCentralMomentTransform):
    """Transform from populations to central moment space by matrix-vector multiplication."""

    def __init__(self, stencil, moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 **kwargs):
        super(PdfsToCentralMomentsByMatrix, self).__init__(
            stencil, moment_polynomials, equilibrium_density, equilibrium_velocity, **kwargs)

        moment_matrix_without_shift = moment_matrix(self.moment_polynomials, self.stencil)
        shift_matrix = set_up_shift_matrix(self.moment_polynomials, self.stencil, equilibrium_velocity)

        self.forward_matrix = shift_matrix * moment_matrix_without_shift
        self.backward_matrix = moment_matrix_without_shift.inv() * shift_matrix.inv()

    def forward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_f_to_k',
                          return_monomials=False):
        r"""Returns an assignment collection containing equations for pre-collision polynomial
        central moments, expressed in terms of the pre-collision populations by matrix-multiplication.

        The central moment transformation matrix :math:`K` provided by :func:`lbmpy.moments.moment_matrix` 
        is used to compute the pre-collision moments as :math:`\mathbf{K} = K \cdot \mathbf{f}`,
        which are returned element-wise.

        Args:
            pdf_symbols: List of symbols that represent the pre-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            return_monomials: Return equations for monomial moments. Use only when specifying 
                              ``moment_exponents`` in constructor!
        """
        simplification = self._get_simp_strategy(simplification)

        if return_monomials:
            assert len(self.moment_exponents) == self.q, "Could not derive invertible monomial transform." \
                f"Expected {self.q} monomials, but got {len(self.moment_exponents)}."
            km = moment_matrix(self.moment_exponents, self.stencil, shift_velocity=self.equilibrium_velocity)
            background_shift = self._cm_background_shift(self.moment_exponents)
            pre_collision_moments = self.pre_collision_monomial_symbols
        else:
            km = self.forward_matrix
            background_shift = self._cm_background_shift(self.moment_polynomials)
            pre_collision_moments = self.pre_collision_symbols

        f_to_k_vec = km * sp.Matrix(pdf_symbols) + background_shift
        main_assignments = [Assignment(k, eq) for k, eq in zip(pre_collision_moments, f_to_k_vec)]

        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpression_symbol_generator=symbol_gen)

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_k_to_f',
                           start_from_monomials=False):
        r"""Returns an assignment collection containing equations for post-collision populations, 
        expressed in terms of the post-collision polynomial central moments by matrix-multiplication.

        The moment transformation matrix :math:`K` provided by :func:`lbmpy.moments.moment_matrix` is
        inverted and used to compute the pre-collision moments as 
        :math:`\mathbf{f}^{\ast} = K^{-1} \cdot \mathbf{K}_{\mathrm{post}}`, which is returned element-wise.

        Args:
            pdf_symbols: List of symbols that represent the post-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            start_from_monomials: Return equations for monomial moments. Use only when specifying 
                                  ``moment_exponents`` in constructor!
        """
        simplification = self._get_simp_strategy(simplification)

        if start_from_monomials:
            assert len(self.moment_exponents) == self.q, "Could not derive invertible monomial transform." \
                f"Expected {self.q} monomials, but got {len(self.moment_exponents)}."
            mm_inv = moment_matrix(self.moment_exponents, self.stencil).inv()
            shift_inv = set_up_shift_matrix(self.moment_exponents, self.stencil, self.equilibrium_velocity).inv()
            km_inv = mm_inv * shift_inv
            background_shift = self._cm_background_shift(self.moment_exponents)
            post_collision_moments = self.post_collision_monomial_symbols
        else:
            km_inv = self.backward_matrix
            background_shift = self._cm_background_shift(self.moment_polynomials)
            post_collision_moments = self.post_collision_symbols

        symbol_gen = SymbolGen(subexpression_base)

        subexpressions = [Assignment(xi, m - s)
                          for xi, m, s in zip(symbol_gen, post_collision_moments, background_shift)]

        m_to_f_vec = km_inv * sp.Matrix([s.lhs for s in subexpressions])
        main_assignments = [Assignment(f, eq) for f, eq in zip(pdf_symbols, m_to_f_vec)]

        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)

        if simplification:
            ac = simplification.apply(ac)
        return ac

    @property
    def _default_simplification(self):
        simplification = SimplificationStrategy()
        return simplification
# end class PdfsToCentralMomentsByMatrix


class BinomialChimeraTransform(AbstractCentralMomentTransform):
    """Transform from populations to central moments using a chimera transform implementing the binomial expansion."""

    def __init__(self, stencil, moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 **kwargs):
        super(BinomialChimeraTransform, self).__init__(
            stencil, moment_polynomials, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations, **kwargs)

        #   Potentially, de-aliasing is required
        if len(self.moment_exponents) != self.q:
            P, m_reduced = central_moment_reduced_monomial_to_polynomial_matrix(self.moment_polynomials,
                                                                                self.stencil,
                                                                                velocity_symbols=equilibrium_velocity)
            self.mono_to_poly_matrix = P
            self.moment_exponents = m_reduced
        else:
            self.mono_to_poly_matrix = monomial_to_polynomial_transformation_matrix(self.moment_exponents,
                                                                                    self.moment_polynomials)

        if 'moment_exponents' in kwargs:
            del kwargs['moment_exponents']

        self.raw_moment_transform = PdfsToMomentsByChimeraTransform(
            stencil, None, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations,
            moment_exponents=self.moment_exponents,
            **kwargs)

        self.poly_to_mono_matrix = self.mono_to_poly_matrix.inv()

    @property
    def absorbs_conserved_quantity_equations(self):
        return True

    def forward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_f_to_k',
                          return_monomials=False):
        r"""Returns equations for polynomial central moments, computed from pre-collision populations
        through a cascade of three steps.

        First, the monomial raw moment vector :math:`\mathbf{m}` is computed using the raw-moment
        chimera transform (see `lbmpy.moment_transforms.PdfsToMomentsByChimeraTransform`).

        Second, we obtain monomial central moments from monomial raw moments using the binomial
        chimera transform:

        .. math::

            \kappa_{ab|\gamma} &:= \sum_{c = 0}^{\gamma} \binom{\gamma}{c} v_z^{\gamma - c} m_{abc} \\
            \kappa_{a|\beta\gamma} &:= \sum_{b = 0}^{\beta} \binom{\beta}{b} v_z^{\beta - b} \kappa_{ab|\gamma} \\
            \kappa_{\alpha\beta\gamma} &:=
                \sum_{a = 0}^{\alpha} \binom{\alpha}{a} v_z^{\alpha - a} \kappa_{a|\beta\gamma} \\

        Lastly, the polynomial central moments are computed using the polynomialization matrix
        as :math:`\mathbf{K} = P \mathbf{\kappa}`.

        **Conserved Quantity Equations**

        If given, this transform absorbs the conserved quantity equations and simplifies them
        using the raw moment equations, if simplification is enabled.

        **Simplification**

        If simplification is enabled, the absorbed conserved quantity equations are - if possible - 
        rewritten using the monomial symbols. If the conserved quantities originate somewhere else
        than in the lower-order moments (like from an external field), they are not affected by this
        simplification.

        The raw moment chimera transform is simplified by propagation of aliases.

        The equations of the binomial chimera transform are simplified by expressing conserved raw moments
        in terms of the conserved quantities, and subsequent propagation of aliases, constants, and any
        expressions that are purely products of conserved quantities.

        **De-Aliasing**

        If more than :math:`q` monomial moments are extracted from the polynomial set, they
        are de-aliased and reduced to a set of only :math:`q` moments using the same rules
        as for raw moments. For polynomialization, a special reduced matrix :math:`\tilde{P}`
        is used, which is computed using `lbmpy.moments.central_moment_reduced_monomial_to_polynomial_matrix`.


        Args:
            pdf_symbols: List of symbols that represent the pre-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            return_monomials: Return equations for monomial moments. Use only when specifying 
                              ``moment_exponents`` in constructor!

        """
        simplification = self._get_simp_strategy(simplification, 'forward')

        mono_raw_moment_base = self.raw_moment_transform.mono_base_pre
        mono_central_moment_base = self.mono_base_pre

        mono_cm_symbols = self.pre_collision_monomial_symbols

        rm_ac = self.raw_moment_transform.forward_transform(pdf_symbols, simplification=False, return_monomials=True)
        cq_symbols_to_moments = self.raw_moment_transform.get_cq_to_moment_symbols_dict(mono_raw_moment_base)

        chim = self.BinomialChimera(tuple(-u for u in self.equilibrium_velocity),
                                    mono_raw_moment_base, mono_central_moment_base)
        chim_ac = chim(self.moment_exponents)

        cq_subs = dict()
        if simplification:
            from lbmpy.methods.momentbased.momentbasedsimplifications import (
                substitute_moments_in_conserved_quantity_equations)
            rm_ac = substitute_moments_in_conserved_quantity_equations(rm_ac)

            #   Compute replacements for conserved moments in terms of the CQE
            rm_asm_dict = rm_ac.main_assignments_dict
            for cq_sym, moment_sym in cq_symbols_to_moments.items():
                cq_eq = rm_asm_dict[cq_sym]
                solutions = sp.solve(cq_eq - cq_sym, moment_sym)
                if len(solutions) > 0:
                    cq_subs[moment_sym] = solutions[0]

            chim_ac = chim_ac.new_with_substitutions(cq_subs, substitute_on_lhs=False)

            fo_kappas = [sq_sym(mono_central_moment_base, es) for es in moments_of_order(1, dim=self.stencil.D)]
            ac_filtered = chim_ac.new_filtered(fo_kappas).new_without_subexpressions()
            chim_asm_dict = chim_ac.main_assignments_dict
            for asm in ac_filtered.main_assignments:
                chim_asm_dict[asm.lhs] = asm.rhs
            chim_ac.set_main_assignments_from_dict(chim_asm_dict)

        subexpressions = rm_ac.all_assignments + chim_ac.subexpressions

        if return_monomials:
            main_assignments = chim_ac.main_assignments
        else:
            subexpressions += chim_ac.main_assignments
            poly_eqs = self.mono_to_poly_matrix * sp.Matrix(mono_cm_symbols)
            main_assignments = [Assignment(m, v) for m, v in zip(self.pre_collision_symbols, poly_eqs)]

        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments=main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_k_to_f',
                           start_from_monomials=False):
        r"""Returns an assignment collection containing equations for post-collision populations, 
        expressed in terms of the post-collision polynomial central moments by three steps.

        The post-collision monomial central moments :math:`\mathbf{\kappa}_{\mathrm{post}}` are first 
        obtained from the polynomials through multiplication with :math:`P^{-1}`.

        Afterward, monomial post-collision raw moments are obtained from monomial central moments using the binomial
        chimera transform:

        .. math::

            m^{\ast}_{ab|\gamma} &:= \sum_{c = 0}^{\gamma} \binom{\gamma}{c} v_z^{\gamma - c} \kappa^{\ast}_{abc} \\
            m^{\ast}_{a|\beta\gamma} &:= \sum_{b = 0}^{\beta} \binom{\beta}{b} v_z^{\beta - b} m^{\ast}_{ab|\gamma} \\
            m^{\ast}_{\alpha\beta\gamma} &:=
                \sum_{a = 0}^{\alpha} \binom{\alpha}{a} v_z^{\alpha - a} m^{\ast}_{a|\beta\gamma} \\

        Finally, the monomial raw moment transformation 
        matrix :math:`M_r` provided by :func:`lbmpy.moments.moment_matrix` 
        is inverted and used to compute the pre-collision moments as 
        :math:`\mathbf{f}_{\mathrm{post}} = M_r^{-1} \cdot \mathbf{m}_{\mathrm{post}}`.

        **De-Aliasing**: 

        See `PdfsToCentralMomentsByShiftMatrix.forward_transform`.

        **Simplifications**

        If simplification is enabled, the inverse shift matrix equations are simplified by recursively 
        inserting lower-order moments into equations for higher-order moments. To this end, these equations 
        are factored recursively by the velocity symbols.

        The equations of the binomial chimera transform are simplified by propagation of aliases.

        Further, the equations for populations :math:`f_i` and :math:`f_{\bar{i}}` 
        of opposite stencil directions :math:`\mathbf{c}_i` and :math:`\mathbf{c}_{\bar{i}} = - \mathbf{c}_i`
        are split into their symmetric and antisymmetric parts :math:`f_i^{\mathrm{sym}}, f_i^{\mathrm{anti}}`, such
        that

        .. math::

            f_i = f_i^{\mathrm{sym}} + f_i^{\mathrm{anti}}

            f_{\bar{i}} = f_i^{\mathrm{sym}} - f_i^{\mathrm{anti}}


        Args:
            pdf_symbols: List of symbols that represent the post-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            start_from_monomials: Return equations for monomial moments. Use only when specifying 
                                  ``moment_exponents`` in constructor!
        """
        simplification = self._get_simp_strategy(simplification, 'backward')

        mono_cm_symbols = self.post_collision_monomial_symbols

        subexpressions = []
        if not start_from_monomials:
            mono_eqs = self.poly_to_mono_matrix * sp.Matrix(self.post_collision_symbols)
            subexpressions += [Assignment(cm, v) for cm, v in zip(mono_cm_symbols, mono_eqs)]

        mono_raw_moment_base = self.raw_moment_transform.mono_base_post
        mono_central_moment_base = self.mono_base_post

        chim = self.BinomialChimera(self.equilibrium_velocity, mono_central_moment_base, mono_raw_moment_base)
        chim_ac = chim(self.moment_exponents)

        if simplification:
            from pystencils.simp import insert_aliases
            chim_ac = insert_aliases(chim_ac)

        subexpressions += chim_ac.all_assignments

        rm_ac = self.raw_moment_transform.backward_transform(
            pdf_symbols, simplification=False, start_from_monomials=True)
        subexpressions += rm_ac.subexpressions

        ac = rm_ac.copy(subexpressions=subexpressions)
        if simplification:
            ac = simplification.apply(ac)

        return ac

    #   ----------------------------- Private Members -----------------------------

    class BinomialChimera:
        def __init__(self, v, from_base, to_base):
            self._v = v
            self._from_base = from_base
            self._to_base = to_base
            self._chim_dict = None

        def _chimera_symbol(self, fixed_directions, remaining_exponents):
            if not fixed_directions:
                return None

            fixed_str = '_'.join(str(direction) for direction in fixed_directions)
            exp_str = '_'.join(str(exp) for exp in remaining_exponents)
            return sp.Symbol(f"chimera_{self._to_base}_{fixed_str}_e_{exp_str}")

        @property
        def chimera_assignments(self):
            assert self._chim_dict is not None
            return [Assignment(lhs, rhs) for lhs, rhs in self._chim_dict.items()]

        def _derive(self, exponents, depth):
            if depth == len(exponents):
                return sq_sym(self._from_base, exponents)

            v = self._v

            fixed = exponents[:depth]
            remaining = exponents[depth:]
            chim_symb = self._chimera_symbol(fixed, remaining)
            if chim_symb in self._chim_dict:
                return chim_symb

            choose = sp.binomial

            alpha = exponents[depth]
            s = sp.Integer(0)
            for a in range(alpha + 1):
                rec_exps = list(exponents)
                rec_exps[depth] = a
                s += choose(alpha, a) * v[depth] ** (alpha - a) * self._derive(rec_exps, depth + 1)

            if chim_symb is not None:
                self._chim_dict[chim_symb] = s
                return chim_symb
            else:
                return Assignment(sq_sym(self._to_base, exponents), s)

        def __call__(self, monos):
            self._chim_dict = dict()
            ac = []
            for m in monos:
                ac.append(self._derive(m, 0))
            return AssignmentCollection(ac, self._chim_dict)

    @property
    def _default_simplification(self):
        from pystencils.simp import insert_aliases, insert_constants
        from lbmpy.methods.momentbased.momentbasedsimplifications import insert_pure_products

        cq = (self.equilibrium_density,) + self.equilibrium_velocity
        fw_skip = cq + self.raw_moment_transform.pre_collision_monomial_symbols + self.pre_collision_monomial_symbols

        forward_simp = SimplificationStrategy()
        forward_simp.add(partial(insert_pure_products, symbols=cq, skip=fw_skip))
        forward_simp.add(partial(insert_aliases, skip=fw_skip))
        forward_simp.add(partial(insert_constants, skip=fw_skip))

        from lbmpy.methods.momentbased.momentbasedsimplifications import split_pdf_main_assignments_by_symmetry

        bw_skip = self.raw_moment_transform.post_collision_monomial_symbols + self.post_collision_monomial_symbols

        backward_simp = SimplificationStrategy()
        backward_simp.add(partial(insert_aliases, skip=bw_skip))
        backward_simp.add(split_pdf_main_assignments_by_symmetry)
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }

# end class PdfsToCentralMomentsByShiftMatrix


class FastCentralMomentTransform(AbstractCentralMomentTransform):
    """Transform from populations to central moments, using the fast central-moment
    transform equations introduced by :cite:`geier2015`.

    **Attention:** The fast central moment transform has originally been designed for the
    D3Q27 stencil, and is also tested and safely usable with D2Q9 and D3Q19. While the forward-
    transform does not pose any problems, the backward equations may be inefficient, or
    even not cleanly derivable for other stencils. Use with care!"""

    def __init__(self, stencil,
                 moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 **kwargs):
        super(FastCentralMomentTransform, self).__init__(
            stencil, moment_polynomials, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations, **kwargs)

        #   Potentially, de-aliasing is required
        if len(self.moment_exponents) != self.q:
            P, m_reduced = central_moment_reduced_monomial_to_polynomial_matrix(self.moment_polynomials,
                                                                                self.stencil,
                                                                                velocity_symbols=equilibrium_velocity)
            self.mono_to_poly_matrix = P
            self.moment_exponents = m_reduced
        else:
            self.mono_to_poly_matrix = monomial_to_polynomial_transformation_matrix(self.moment_exponents,
                                                                                    self.moment_polynomials)

        self.poly_to_mono_matrix = self.mono_to_poly_matrix.inv()

        moment_matrix_without_shift = moment_matrix(self.moment_exponents, self.stencil)
        shift_matrix = set_up_shift_matrix(self.moment_exponents, self.stencil, equilibrium_velocity)
        self.inv_monomial_matrix = moment_matrix_without_shift.inv() * shift_matrix.inv()

    def forward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_f_to_k',
                          return_monomials=False):
        r"""Returns an assignment collection containing equations for pre-collision polynomial
        central moments, expressed in terms of the pre-collision populations.

        The monomial central moments are computed from populations through the central-moment
        chimera transform:

        .. math::

            f_{xyz} &:= f_i \text{ such that } c_i = (x,y,z)^T \\
            \kappa_{xy|\gamma} &:= \sum_{z \in \{-1, 0, 1\} } f_{xyz} \cdot (z - u_z)^{\gamma} \\
            \kappa_{x|\beta \gamma} &:= \sum_{y \in \{-1, 0, 1\}} \kappa_{xy|\gamma} \cdot (y - u_y)^{\beta} \\
            \kappa_{\alpha \beta \gamma} &:= \sum_{x \in \{-1, 0, 1\}} \kappa_{x|\beta \gamma} \cdot (x - u_x)^{\alpha}

        The polynomial moments are afterward computed from the monomials by matrix-multiplication 
        using the polynomialization matrix :math:`P`.

        **De-Aliasing**

        If more than :math:`q` monomial moments are extracted from the polynomial set, they
        are de-aliased and reduced to a set of only :math:`q` moments using the same rules
        as for raw moments. For polynomialization, a special reduced matrix :math:`\tilde{P}`
        is used, which is computed using `lbmpy.moments.central_moment_reduced_monomial_to_polynomial_matrix`.


        Args:
            pdf_symbols: List of symbols that represent the pre-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            return_monomials: Return equations for monomial moments. Use only when specifying 
                              ``moment_exponents`` in constructor!
        """
        simplification = self._get_simp_strategy(simplification, 'forward')
        monomial_symbol_base = self.mono_base_pre

        def _partial_kappa_symbol(fixed_directions, remaining_exponents):
            fixed_str = '_'.join(str(direction) for direction in fixed_directions).replace('-', 'm')
            exp_str = '_'.join(str(exp) for exp in remaining_exponents).replace('-', 'm')
            return sp.Symbol(f"partial_{monomial_symbol_base}_{fixed_str}_e_{exp_str}")

        partial_sums_dict = dict()
        monomial_eqs = []

        def collect_partial_sums(exponents, dimension=0, fixed_directions=tuple()):
            if dimension == self.dim:
                #   Base Case
                if fixed_directions in self.stencil:
                    return pdf_symbols[self.stencil.index(fixed_directions)]
                else:
                    return 0
            else:
                #   Recursive Case
                summation = sp.sympify(0)
                for d in [-1, 0, 1]:
                    next_partial = collect_partial_sums(
                        exponents, dimension=dimension + 1, fixed_directions=fixed_directions + (d,))
                    summation += next_partial * (d - self.equilibrium_velocity[dimension])**exponents[dimension]

                if dimension == 0:
                    lhs_symbol = sq_sym(monomial_symbol_base, exponents)
                    monomial_eqs.append(Assignment(lhs_symbol, summation))
                else:
                    lhs_symbol = _partial_kappa_symbol(fixed_directions, exponents[dimension:])
                    partial_sums_dict[lhs_symbol] = summation
                return lhs_symbol

        for e in self.moment_exponents:
            collect_partial_sums(e)

        subexpressions = [Assignment(lhs, rhs) for lhs, rhs in partial_sums_dict.items()]

        if return_monomials:
            shift = self._cm_background_shift(self.moment_exponents)
            main_assignments = [Assignment(a.lhs, a.rhs + s) for a, s in zip(monomial_eqs, shift)]
        else:
            subexpressions += monomial_eqs
            moment_eqs = self.mono_to_poly_matrix * sp.Matrix(self.pre_collision_monomial_symbols)
            moment_eqs += self._cm_background_shift(self.moment_polynomials)
            main_assignments = [Assignment(m, v) for m, v in zip(self.pre_collision_symbols, moment_eqs)]

        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)
        if simplification:
            ac = self._simplify_lower_order_moments(ac, monomial_symbol_base, return_monomials)
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_k_to_f',
                           start_from_monomials=False):
        r"""Returns an assignment collection containing equations for post-collision populations, 
        expressed in terms of the post-collision polynomial central moments using the backward
        fast central moment transform.

        First, monomial central moments are obtained from the polynomial moments by multiplication
        with :math:`P^{-1}`. Then, the elementwise equations of the matrix 
        multiplication :math:`K^{-1} \cdot \mathbf{K}` with the monomial central moment matrix 
        (see `PdfsToCentralMomentsByMatrix`) are recursively simplified by extracting certain linear 
        combinations of velocities, to obtain equations similar to the ones given in :cite:`geier2015`.

        The backward transform is designed for D3Q27, inherently generalizes to D2Q9, and is tested
        for D3Q19. It also returns correct equations for D3Q15, whose efficiency is however questionable.

        **De-Aliasing**: 

        See `FastCentralMomentTransform.forward_transform`.

        Args:
            pdf_symbols: List of symbols that represent the post-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            start_from_monomials: Return equations for monomial moments. Use only when specifying 
                                  ``moment_exponents`` in constructor!
        """
        simplification = self._get_simp_strategy(simplification, 'backward')

        post_collision_moments = self.post_collision_symbols
        post_collision_monomial_moments = self.post_collision_monomial_symbols

        symbol_gen = SymbolGen(subexpression_base)

        subexpressions = []
        if not start_from_monomials:
            background_shift = self._cm_background_shift(self.moment_polynomials)
            shift_equations = [Assignment(xi, m - s)
                               for xi, m, s in zip(symbol_gen, post_collision_moments, background_shift)]
            monomial_eqs = self.poly_to_mono_matrix * sp.Matrix([s.lhs for s in shift_equations])
            subexpressions += shift_equations
            subexpressions += [Assignment(m, v) for m, v in zip(post_collision_monomial_moments, monomial_eqs)]
        else:
            background_shift = self._cm_background_shift(self.moment_exponents)
            shift_equations = [Assignment(xi, m - s)
                               for xi, m, s in zip(symbol_gen, post_collision_monomial_moments, background_shift)]
            subexpressions += shift_equations
            post_collision_monomial_moments = [s.lhs for s in shift_equations]

        raw_equations = self.inv_monomial_matrix * sp.Matrix(post_collision_monomial_moments)
        raw_equations = [Assignment(f, eq) for f, eq in zip(pdf_symbols, raw_equations)]

        ac = self._split_backward_equations(raw_equations, symbol_gen)
        ac.subexpressions = subexpressions + ac.subexpressions
        if simplification:
            ac = simplification.apply(ac)
        return ac

    #   ----------------------------- Private Members -----------------------------

    @property
    def _default_simplification(self):
        forward_simp = SimplificationStrategy()

        backward_simp = SimplificationStrategy()
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }

    def _simplify_lower_order_moments(self, ac, moment_base, search_in_main_assignments):
        if self.cqe is None:
            return ac

        moment_symbols = [sq_sym(moment_base, e) for e in moments_up_to_order(1, dim=self.dim)]

        if search_in_main_assignments:
            f_to_cm_dict = ac.main_assignments_dict
            f_to_cm_dict_reduced = ac.new_without_subexpressions().main_assignments_dict
        else:
            f_to_cm_dict = ac.subexpressions_dict
            f_to_cm_dict_reduced = ac.new_without_subexpressions(moment_symbols).subexpressions_dict

        cqe_subs = self.cqe.new_without_subexpressions().main_assignments_dict
        for m in moment_symbols:
            m_eq = fast_subs(fast_subs(f_to_cm_dict_reduced[m], cqe_subs), cqe_subs)
            m_eq = m_eq.expand().cancel()
            for cqe_sym, cqe_exp in cqe_subs.items():
                m_eq = subs_additive(m_eq, cqe_sym, cqe_exp)
            f_to_cm_dict[m] = m_eq

        if search_in_main_assignments:
            main_assignments = [Assignment(lhs, rhs) for lhs, rhs in f_to_cm_dict.items()]
            return ac.copy(main_assignments=main_assignments)
        else:
            subexpressions = [Assignment(lhs, rhs) for lhs, rhs in f_to_cm_dict.items()]
            return ac.copy(subexpressions=subexpressions)

    def _split_backward_equations_recursive(self, assignment, all_subexpressions,
                                            stencil_direction, subexp_symgen, known_coeffs_dict,
                                            step=0):
        #   Base Cases
        # if step == self.dim:
        #     return assignment

        #   Base Case
        #   If there are no more velocity symbols in the subexpression,
        #   don't split it up further
        if assignment.rhs.atoms(sp.Symbol).isdisjoint(set(self.equilibrium_velocity)):
            return assignment

        #   Recursive Case

        u = self.equilibrium_velocity[-1 - step]
        d = stencil_direction[-1 - step]
        one = sp.Integer(1)
        two = sp.Integer(2)

        #   Factors to group terms by
        grouping_factors = {
            -1: [one, 2 * u - 1, u**2 - u],
            0: [-one, -2 * u, 1 - u**2],
            1: [one, 2 * u + 1, u**2 + u]
        }
        factors = grouping_factors[d]

        #   Common Integer factor to extract from all groups
        common_factor = one if d == 0 else two

        #   Proxy for factor grouping
        v = sp.Symbol('v')
        square_factor_eq = (factors[2] - v**2)
        lin_factor_eq = (factors[1] - v)
        sub_for_u_sq = sp.solve(square_factor_eq, u**2)[0]
        sub_for_u = sp.solve(lin_factor_eq, u)[0]
        subs = {u**2: sub_for_u_sq, u: sub_for_u}
        rhs_grouped_by_v = assignment.rhs.subs(subs).expand().collect(v)

        new_rhs = sp.Integer(0)
        for k in range(3):
            coeff = rhs_grouped_by_v.coeff(v, k)
            coeff_subexp = common_factor * coeff

            #   Explicitly divide out the constant factor in the zero case
            if k == 0:
                coeff_subexp = coeff_subexp / factors[0]

            #   MEMOISATION:
            #   The subexpression just generated might already have been found
            #   If so, reuse the existing symbol and skip forward.
            #   Otherwise, create it anew and continue recursively
            coeff_symb = known_coeffs_dict.get(coeff_subexp, None)
            if coeff_symb is None:
                coeff_symb = next(subexp_symgen)
                known_coeffs_dict[coeff_subexp] = coeff_symb
                coeff_assignment = Assignment(coeff_symb, coeff_subexp)

                #   Recursively split the coefficient term
                coeff_assignment = self._split_backward_equations_recursive(
                    coeff_assignment, all_subexpressions, stencil_direction, subexp_symgen,
                    known_coeffs_dict, step=step + 1)
                all_subexpressions.append(coeff_assignment)

            new_rhs += factors[k] * coeff_symb

        new_rhs = sp.Rational(1, common_factor) * new_rhs

        return Assignment(assignment.lhs, new_rhs)

    def _split_backward_equations(self, backward_assignments, subexp_symgen):
        all_subexpressions = []
        split_main_assignments = []
        known_coeffs_dict = dict()
        for asm, stencil_dir in zip(backward_assignments, self.stencil):
            split_asm = self._split_backward_equations_recursive(
                asm, all_subexpressions, stencil_dir, subexp_symgen, known_coeffs_dict)
            split_main_assignments.append(split_asm)
        ac = AssignmentCollection(split_main_assignments, subexpressions=all_subexpressions,
                                  subexpression_symbol_generator=subexp_symgen)
        ac.topological_sort(sort_main_assignments=False)
        return ac

# end class FastCentralMomentTransform


class PdfsToCentralMomentsByShiftMatrix(AbstractCentralMomentTransform):
    """Transform from populations to central moments using a shift matrix."""

    def __init__(self, stencil, moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 **kwargs):
        super(PdfsToCentralMomentsByShiftMatrix, self).__init__(
            stencil, moment_polynomials, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations, **kwargs)

        #   Potentially, de-aliasing is required
        if len(self.moment_exponents) != self.q:
            P, m_reduced = central_moment_reduced_monomial_to_polynomial_matrix(self.moment_polynomials,
                                                                                self.stencil,
                                                                                velocity_symbols=equilibrium_velocity)
            self.mono_to_poly_matrix = P
            self.moment_exponents = m_reduced
        else:
            self.mono_to_poly_matrix = monomial_to_polynomial_transformation_matrix(self.moment_exponents,
                                                                                    self.moment_polynomials)

        if 'moment_exponents' in kwargs:
            del kwargs['moment_exponents']

        self.raw_moment_transform = PdfsToMomentsByChimeraTransform(
            stencil, None, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations,
            moment_exponents=self.moment_exponents,
            **kwargs)

        self.shift_matrix = set_up_shift_matrix(self.moment_exponents, self.stencil, self.equilibrium_velocity)
        self.inv_shift_matrix = self.shift_matrix.inv()
        self.poly_to_mono_matrix = self.mono_to_poly_matrix.inv()

    @property
    def absorbs_conserved_quantity_equations(self):
        return True

    def forward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_f_to_k',
                          return_monomials=False):
        r"""Returns equations for polynomial central moments, computed from pre-collision populations
        through a cascade of three steps.

        First, the monomial raw moment vector :math:`\mathbf{m}` is computed using the raw-moment
        chimera transform (see `lbmpy.moment_transforms.PdfsToMomentsByChimeraTransform`). Then, the
        monomial shift matrix :math:`N` provided by `lbmpy.moments.set_up_shift_matrix` is used to compute
        the monomial central moment vector as :math:`\mathbf{\kappa} = N \mathbf{m}`. Lastly, the polynomial
        central moments are computed using the polynomialization matrix as :math:`\mathbf{K} = P \mathbf{\kappa}`.

        **Conserved Quantity Equations**

        If given, this transform absorbs the conserved quantity equations and simplifies them
        using the raw moment equations, if simplification is enabled.


        **Simplification**

        If simplification is enabled, the absorbed conserved quantity equations are - if possible - 
        rewritten using the monomial symbols. If the conserved quantities originate somewhere else
        than in the lower-order moments (like from an external field), they are not affected by this
        simplification.

        The relations between conserved quantities and raw moments are used to simplify the equations
        obtained from the shift matrix. Further, these equations are simplified by recursively inserting
        lower-order moments into equations for higher-order moments.

         **De-Aliasing**

        If more than :math:`q` monomial moments are extracted from the polynomial set, they
        are de-aliased and reduced to a set of only :math:`q` moments using the same rules
        as for raw moments. For polynomialization, a special reduced matrix :math:`\tilde{P}`
        is used, which is computed using `lbmpy.moments.central_moment_reduced_monomial_to_polynomial_matrix`.


        Args:
            pdf_symbols: List of symbols that represent the pre-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            return_monomials: Return equations for monomial moments. Use only when specifying 
                              ``moment_exponents`` in constructor!

        """
        simplification = self._get_simp_strategy(simplification, 'forward')

        raw_moment_base = self.raw_moment_transform.mono_base_pre
        central_moment_base = self.mono_base_pre

        mono_rm_symbols = self.raw_moment_transform.pre_collision_monomial_symbols
        mono_cm_symbols = self.pre_collision_monomial_symbols

        rm_ac = self.raw_moment_transform.forward_transform(pdf_symbols, simplification=False, return_monomials=True)
        cq_symbols_to_moments = self.raw_moment_transform.get_cq_to_moment_symbols_dict(raw_moment_base)
        rm_to_cm_vec = self.shift_matrix * sp.Matrix(mono_rm_symbols)

        cq_subs = dict()
        if simplification:
            from lbmpy.methods.momentbased.momentbasedsimplifications import (
                substitute_moments_in_conserved_quantity_equations)
            rm_ac = substitute_moments_in_conserved_quantity_equations(rm_ac)

            #   Compute replacements for conserved moments in terms of the CQE
            rm_asm_dict = rm_ac.main_assignments_dict
            for cq_sym, moment_sym in cq_symbols_to_moments.items():
                cq_eq = rm_asm_dict[cq_sym]
                solutions = sp.solve(cq_eq - cq_sym, moment_sym)
                if len(solutions) > 0:
                    cq_subs[moment_sym] = solutions[0]

            rm_to_cm_vec = fast_subs(rm_to_cm_vec, cq_subs)

        rm_to_cm_dict = {cm: rm for cm, rm in zip(mono_cm_symbols, rm_to_cm_vec)}

        if simplification:
            rm_to_cm_dict = self._simplify_raw_to_central_moments(
                rm_to_cm_dict, self.moment_exponents, raw_moment_base, central_moment_base)
            rm_to_cm_dict = self._undo_remaining_cq_subexpressions(rm_to_cm_dict, cq_subs)

        subexpressions = rm_ac.all_assignments

        if return_monomials:
            main_assignments = [Assignment(lhs, rhs) for lhs, rhs in rm_to_cm_dict.items()]
        else:
            subexpressions += [Assignment(lhs, rhs) for lhs, rhs in rm_to_cm_dict.items()]
            poly_eqs = self.mono_to_poly_matrix * sp.Matrix(mono_cm_symbols)
            main_assignments = [Assignment(m, v) for m, v in zip(self.pre_collision_symbols, poly_eqs)]

        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments=main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_k_to_f',
                           start_from_monomials=False):
        r"""Returns an assignment collection containing equations for post-collision populations, 
        expressed in terms of the post-collision polynomial central moments by matrix-multiplication
        including the shift matrix.

        The post-collision monomial central moments :math:`\mathbf{\kappa}_{\mathrm{post}}` are first 
        obtained from the polynomials through multiplication with :math:`P^{-1}`.
        The shift-matrix is inverted as well, to obtain the monomial raw moments as 
        :math:`\mathbf{m}_{post} = N^{-1} \mathbf{\kappa}_{post}`. Finally, the monomial raw moment transformation 
        matrix :math:`M_r` provided by :func:`lbmpy.moments.moment_matrix` 
        is inverted and used to compute the pre-collision moments as 
        :math:`\mathbf{f}_{\mathrm{post}} = M_r^{-1} \cdot \mathbf{m}_{\mathrm{post}}`.

         **De-Aliasing**: 

        See `PdfsToCentralMomentsByShiftMatrix.forward_transform`.

        **Simplifications**

        If simplification is enabled, the inverse shift matrix equations are simplified by recursively 
        inserting lower-order moments into equations for higher-order moments. To this end, these equations 
        are factored recursively by the velocity symbols.

        Further, the equations for populations :math:`f_i` and :math:`f_{\bar{i}}` 
        of opposite stencil directions :math:`\mathbf{c}_i` and :math:`\mathbf{c}_{\bar{i}} = - \mathbf{c}_i`
        are split into their symmetric and antisymmetric parts :math:`f_i^{\mathrm{sym}}, f_i^{\mathrm{anti}}`, such
        that

        .. math::

            f_i = f_i^{\mathrm{sym}} + f_i^{\mathrm{anti}}

            f_{\bar{i}} = f_i^{\mathrm{sym}} - f_i^{\mathrm{anti}}


        Args:
            pdf_symbols: List of symbols that represent the post-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            start_from_monomials: Return equations for monomial moments. Use only when specifying 
                                  ``moment_exponents`` in constructor!
        """
        simplification = self._get_simp_strategy(simplification, 'backward')

        mono_rm_symbols = self.raw_moment_transform.post_collision_monomial_symbols
        mono_cm_symbols = self.post_collision_monomial_symbols

        subexpressions = []
        if not start_from_monomials:
            mono_eqs = self.poly_to_mono_matrix * sp.Matrix(self.post_collision_symbols)
            subexpressions += [Assignment(cm, v) for cm, v in zip(mono_cm_symbols, mono_eqs)]

        cm_to_rm_vec = self.inv_shift_matrix * sp.Matrix(mono_cm_symbols)
        cm_to_rm_dict = {rm: eq for rm, eq in zip(mono_rm_symbols, cm_to_rm_vec)}

        if simplification:
            cm_to_rm_dict = self._factor_backward_eqs_by_velocities(mono_rm_symbols, cm_to_rm_dict)

        rm_ac = self.raw_moment_transform.backward_transform(
            pdf_symbols, simplification=False, start_from_monomials=True)
        subexpressions += [Assignment(lhs, rhs) for lhs, rhs in cm_to_rm_dict.items()]
        subexpressions += rm_ac.subexpressions
        ac = rm_ac.copy(subexpressions=subexpressions)
        if simplification:
            ac = simplification.apply(ac)

        return ac

    #   ----------------------------- Private Members -----------------------------

    def _simplify_raw_to_central_moments(self, rm_to_cm_dict, moment_exponents, raw_moment_base, central_moment_base):
        for cm in moment_exponents:
            if sum(cm) < 2:
                continue
            cm_symb = sq_sym(central_moment_base, cm)
            cm_asm_rhs = rm_to_cm_dict[cm_symb]
            for m in contained_moments(cm, min_order=2)[::-1]:
                contained_rm_symb = sq_sym(raw_moment_base, m)
                contained_cm_symb = sq_sym(central_moment_base, m)
                contained_cm_eq = rm_to_cm_dict[contained_cm_symb]
                rm_in_terms_of_cm = sp.solve(contained_cm_eq - contained_cm_symb, contained_rm_symb)[0]
                cm_asm_rhs = cm_asm_rhs.subs({contained_rm_symb: rm_in_terms_of_cm}).expand()
            rm_to_cm_dict[cm_symb] = cm_asm_rhs
        return rm_to_cm_dict

    def _undo_remaining_cq_subexpressions(self, rm_to_cm_dict, cq_subs):
        for cm, cm_eq in rm_to_cm_dict.items():
            for rm, rm_subexp in cq_subs.items():
                cm_eq = subs_additive(cm_eq, rm, rm_subexp)
            rm_to_cm_dict[cm] = cm_eq
        return rm_to_cm_dict

    def _factor_backward_eqs_by_velocities(self, symbolic_rms, cm_to_rm_dict, required_match_replacement=0.75):
        velocity_by_occurences = dict()
        for rm, rm_eq in cm_to_rm_dict.items():
            velocity_by_occurences[rm] = sorted(self.equilibrium_velocity, key=rm_eq.count, reverse=True)
        for d in range(self.dim):
            for rm, rm_eq in cm_to_rm_dict.items():
                u_sorted = velocity_by_occurences[rm]
                cm_to_rm_dict[rm] = rm_eq.expand().collect(u_sorted[d])

            for i, rm1 in enumerate(symbolic_rms):
                for _, rm2 in enumerate(symbolic_rms[i + 1:]):
                    cm_to_rm_dict[rm2] = subs_additive(
                        cm_to_rm_dict[rm2], rm1, cm_to_rm_dict[rm1],
                        required_match_replacement=required_match_replacement)
        return cm_to_rm_dict

    @property
    def _default_simplification(self):
        forward_simp = SimplificationStrategy()

        from lbmpy.methods.momentbased.momentbasedsimplifications import split_pdf_main_assignments_by_symmetry

        backward_simp = SimplificationStrategy()
        backward_simp.add(split_pdf_main_assignments_by_symmetry)
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }

# end class PdfsToCentralMomentsByShiftMatrix
