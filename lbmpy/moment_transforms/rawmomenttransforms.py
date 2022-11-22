from functools import partial
import sympy as sp

from pystencils import Assignment, AssignmentCollection
from pystencils.simp import (
    SimplificationStrategy, add_subexpressions_for_divisions, add_subexpressions_for_constants,
    insert_aliases, insert_constants)
from pystencils.simp.assignment_collection import SymbolGen

from lbmpy.moments import (
    moment_matrix, monomial_to_polynomial_transformation_matrix, non_aliased_polynomial_raw_moments)
from lbmpy.moments import statistical_quantity_symbol as sq_sym

from .abstractmomenttransform import (
    AbstractMomentTransform,
    PRE_COLLISION_RAW_MOMENT, POST_COLLISION_RAW_MOMENT,
    PRE_COLLISION_MONOMIAL_RAW_MOMENT, POST_COLLISION_MONOMIAL_RAW_MOMENT
)


class AbstractRawMomentTransform(AbstractMomentTransform):
    """Abstract base class for all transformations between population space
    and raw-moment space."""

    def __init__(self, stencil, moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 pre_collision_symbol_base=PRE_COLLISION_RAW_MOMENT,
                 post_collision_symbol_base=POST_COLLISION_RAW_MOMENT,
                 pre_collision_monomial_symbol_base=PRE_COLLISION_MONOMIAL_RAW_MOMENT,
                 post_collision_monomial_symbol_base=POST_COLLISION_MONOMIAL_RAW_MOMENT,
                 **kwargs):
        super(AbstractRawMomentTransform, self).__init__(
            stencil, equilibrium_density, equilibrium_velocity,
            moment_polynomials=moment_polynomials,
            pre_collision_symbol_base=pre_collision_symbol_base,
            post_collision_symbol_base=post_collision_symbol_base,
            pre_collision_monomial_symbol_base=pre_collision_monomial_symbol_base,
            post_collision_monomial_symbol_base=post_collision_monomial_symbol_base,
            **kwargs
        )

        assert len(self.moment_polynomials) == self.q, 'Number of moments must match stencil'

    def _rm_background_shift(self, raw_moments):
        if self.background_distribution is not None:
            shift = self.background_distribution.moments(raw_moments)
        else:
            shift = (0,) * self.q
        return sp.Matrix(shift)
# end class AbstractRawMomentTransform


class PdfsToMomentsByMatrixTransform(AbstractRawMomentTransform):
    """Transform between populations and moment space spanned by a polynomial
    basis, using matrix-vector multiplication."""

    def __init__(self, stencil, moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 **kwargs):
        super(PdfsToMomentsByMatrixTransform, self).__init__(
            stencil, moment_polynomials, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations, **kwargs)

        self.moment_matrix = moment_matrix(self.moment_polynomials, stencil)
        self.inv_moment_matrix = self.moment_matrix.inv()

    @property
    def absorbs_conserved_quantity_equations(self):
        return False

    def forward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_f_to_M',
                          return_monomials=False):
        r"""Returns an assignment collection containing equations for pre-collision polynomial
        moments, expressed in terms of the pre-collision populations by matrix-multiplication.

        The moment transformation matrix :math:`M` provided by :func:`lbmpy.moments.moment_matrix` is
        used to compute the pre-collision moments as :math:`\mathbf{M} = M \cdot \mathbf{f}`,
        which is returned element-wise.

        Args:
            pdf_symbols: List of symbols that represent the pre-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            return_monomials: Return equations for monomial moments. Use only when specifying
                              ``moment_exponents`` in constructor!
        """
        simplification = self._get_simp_strategy(simplification, 'forward')

        if return_monomials:
            assert len(self.moment_exponents) == self.q, "Could not derive invertible monomial transform." \
                f"Expected {self.q} monomials, but got {len(self.moment_exponents)}."
            mm = moment_matrix(self.moment_exponents, self.stencil)
            background_shift = self._rm_background_shift(self.moment_exponents)
            pre_collision_moments = self.pre_collision_monomial_symbols
        else:
            mm = self.moment_matrix
            background_shift = self._rm_background_shift(self.moment_polynomials)
            pre_collision_moments = self.pre_collision_symbols

        f_to_m_vec = mm * sp.Matrix(pdf_symbols) + background_shift
        main_assignments = [Assignment(m, eq) for m, eq in zip(pre_collision_moments, f_to_m_vec)]

        symbol_gen = SymbolGen(symbol=subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpression_symbol_generator=symbol_gen)

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_k_to_f',
                           start_from_monomials=False):
        r"""Returns an assignment collection containing equations for post-collision populations,
        expressed in terms of the post-collision polynomial moments by matrix-multiplication.

        The moment transformation matrix :math:`M` provided by :func:`lbmpy.moments.moment_matrix` is
        inverted and used to compute the pre-collision moments as
        :math:`\mathbf{f}^{\ast} = M^{-1} \cdot \mathbf{M}_{\mathrm{post}}`, which is returned element-wise.

        **Simplifications**

        If simplification is enabled, the equations for populations :math:`f_i` and :math:`f_{\bar{i}}`
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

        if start_from_monomials:
            assert len(self.moment_exponents) == self.q, "Could not derive invertible monomial transform." \
                f"Expected {self.q} monomials, but got {len(self.moment_exponents)}."
            mm_inv = moment_matrix(self.moment_exponents, self.stencil).inv()
            background_shift = self._rm_background_shift(self.moment_exponents)
            post_collision_moments = self.post_collision_monomial_symbols
        else:
            mm_inv = self.inv_moment_matrix
            background_shift = self._rm_background_shift(self.moment_polynomials)
            post_collision_moments = self.post_collision_symbols

        symbol_gen = SymbolGen(subexpression_base)

        subexpressions = [Assignment(xi, m - s)
                          for xi, m, s in zip(symbol_gen, post_collision_moments, background_shift)]

        m_to_f_vec = mm_inv * sp.Matrix([s.lhs for s in subexpressions])
        main_assignments = [Assignment(f, eq) for f, eq in zip(pdf_symbols, m_to_f_vec)]

        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)

        ac.add_simplification_hint('stencil', self.stencil)
        ac.add_simplification_hint('post_collision_pdf_symbols', pdf_symbols)
        if simplification:
            ac = simplification.apply(ac)
        return ac

    #   ----------------------------- Private Members -----------------------------

    @ property
    def _default_simplification(self):
        forward_simp = SimplificationStrategy()
        # forward_simp.add(substitute_moments_in_conserved_quantity_equations)
        forward_simp.add(insert_aliases)
        forward_simp.add(add_subexpressions_for_divisions)

        from lbmpy.methods.momentbased.momentbasedsimplifications import split_pdf_main_assignments_by_symmetry

        backward_simp = SimplificationStrategy()
        backward_simp.add(insert_aliases)
        backward_simp.add(split_pdf_main_assignments_by_symmetry)
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }


# end class PdfsToMomentsByMatrixTransform

class PdfsToMomentsByChimeraTransform(AbstractRawMomentTransform):
    """Transform between populations and moment space spanned by a polynomial
    basis, using the raw-moment chimera transform in the forward direction and
    matrix-vector multiplication in the backward direction."""

    def __init__(self, stencil, moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 **kwargs):

        if moment_polynomials:
            #   Remove aliases
            moment_polynomials = non_aliased_polynomial_raw_moments(moment_polynomials, stencil)

        super(PdfsToMomentsByChimeraTransform, self).__init__(
            stencil, moment_polynomials, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations, **kwargs)

        self.inv_moment_matrix = moment_matrix(self.moment_exponents, self.stencil).inv()
        self.mono_to_poly_matrix = monomial_to_polynomial_transformation_matrix(self.moment_exponents,
                                                                                self.moment_polynomials)
        self.poly_to_mono_matrix = self.mono_to_poly_matrix.inv()

    @ property
    def absorbs_conserved_quantity_equations(self):
        return True

    def get_cq_to_moment_symbols_dict(self, moment_symbol_base):
        """Returns a dictionary mapping the density and velocity symbols to the correspondig
        zeroth- and first-order raw moment symbols"""
        if self.cqe is None:
            return dict()

        rho = self.equilibrium_density
        u = self.equilibrium_velocity
        cq_symbols_to_moments = dict()
        if isinstance(rho, sp.Symbol) and rho in self.cqe.defined_symbols:
            cq_symbols_to_moments[rho] = sq_sym(moment_symbol_base, (0,) * self.dim)
        for d, u_sym in enumerate(u):
            if isinstance(u_sym, sp.Symbol) and u_sym in self.cqe.defined_symbols:
                cq_symbols_to_moments[u_sym] = sq_sym(moment_symbol_base, tuple(
                    (1 if i == d else 0) for i in range(self.dim)))
        return cq_symbols_to_moments

    def forward_transform(self, pdf_symbols, simplification=True,
                          subexpression_base='sub_f_to_m',
                          return_monomials=False):
        r"""Returns an assignment collection containing equations for pre-collision polynomial
        moments, expressed in terms of the pre-collision populations, using the raw moment chimera transform.

        The chimera transform for raw moments is given by :cite:`geier2015` :

        .. math::

            f_{xyz} &:= f_i \text{ such that } c_i = (x,y,z)^T \\
            m_{xy|\gamma} &:= \sum_{z \in \{-1, 0, 1\} } f_{xyz} \cdot z^{\gamma} \\
            m_{x|\beta \gamma} &:= \sum_{y \in \{-1, 0, 1\}} m_{xy|\gamma} \cdot y^{\beta} \\
            m_{\alpha \beta \gamma} &:= \sum_{x \in \{-1, 0, 1\}} m_{x|\beta \gamma} \cdot x^{\alpha}


        The obtained raw moments are afterward combined to the desired polynomial moments.

        **Conserved Quantity Equations**

        If given, this transform absorbs the conserved quantity equations and simplifies them
        using the monomial raw moment equations, if simplification is enabled.

        **De-Aliasing**

        If more than :math:`q` monomial moments are extracted from the polynomial set, the polynomials
        are de-aliased by eliminating aliases according to the stencil
        using `lbmpy.moments.non_aliased_polynomial_raw_moments`.

        **Simplification**

        If simplification is enabled, the absorbed conserved quantity equations are - if possible -
        rewritten using the monomial symbols. If the conserved quantities originate somewhere else
        than in the lower-order moments (like from an external field), they are not affected by this
        simplification. Furthermore, aliases and constants are propagated in the chimera equations.

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
                    summation += next_partial * d ** exponents[dimension]

                if dimension == 0:
                    lhs_symbol = sq_sym(monomial_symbol_base, exponents)
                    monomial_eqs.append(Assignment(lhs_symbol, summation))
                else:
                    lhs_symbol = _partial_kappa_symbol(fixed_directions, exponents[dimension:])
                    partial_sums_dict[lhs_symbol] = summation
                return lhs_symbol

        for e in self.moment_exponents:
            collect_partial_sums(e)

        main_assignments = self.cqe.main_assignments.copy() if self.cqe is not None else []
        subexpressions = self.cqe.subexpressions.copy() if self.cqe is not None else []
        subexpressions += [Assignment(lhs, rhs) for lhs, rhs in partial_sums_dict.items()]

        if return_monomials:
            shift = self._rm_background_shift(self.moment_exponents)
            main_assignments += [Assignment(a.lhs, a.rhs + s) for a, s in zip(monomial_eqs, shift)]
        else:
            subexpressions += monomial_eqs
            moment_eqs = self.mono_to_poly_matrix * sp.Matrix(self.pre_collision_monomial_symbols)
            moment_eqs += self._rm_background_shift(self.moment_polynomials)
            main_assignments += [Assignment(m, v) for m, v in zip(self.pre_collision_symbols, moment_eqs)]

        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)
        ac.add_simplification_hint('cq_symbols_to_moments', self.get_cq_to_moment_symbols_dict(monomial_symbol_base))

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, simplification=True,
                           subexpression_base='sub_k_to_f',
                           start_from_monomials=False):
        r"""Returns an assignment collection containing equations for post-collision populations,
        expressed in terms of the post-collision polynomial moments by matrix-multiplication.

        The post-collision monomial moments :math:`\mathbf{m}_{\mathrm{post}}` are first obtained from the polynomials.
        Then, the monomial transformation matrix :math:`M_r` provided by :func:`lbmpy.moments.moment_matrix`
        is inverted and used to compute the post-collision populations as
        :math:`\mathbf{f}_{\mathrm{post}} = M_r^{-1} \cdot \mathbf{m}_{\mathrm{post}}`.

        **De-Aliasing**

        See `PdfsToMomentsByChimeraTransform.forward_transform`.

        **Simplifications**

        If simplification is enabled, the equations for populations :math:`f_i` and :math:`f_{\bar{i}}`
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

        post_collision_moments = self.post_collision_symbols
        post_collision_monomial_moments = self.post_collision_monomial_symbols

        symbol_gen = SymbolGen(subexpression_base)

        subexpressions = []
        if not start_from_monomials:
            background_shift = self._rm_background_shift(self.moment_polynomials)
            shift_equations = [Assignment(xi, m - s)
                               for xi, m, s in zip(symbol_gen, post_collision_moments, background_shift)]
            raw_moment_eqs = self.poly_to_mono_matrix * sp.Matrix([s.lhs for s in shift_equations])
            subexpressions += shift_equations
            subexpressions += [Assignment(rm, v) for rm, v in zip(post_collision_monomial_moments, raw_moment_eqs)]
        else:
            background_shift = self._rm_background_shift(self.moment_exponents)
            shift_equations = [Assignment(xi, m - s)
                               for xi, m, s in zip(symbol_gen, post_collision_monomial_moments, background_shift)]
            subexpressions += shift_equations
            post_collision_monomial_moments = [s.lhs for s in shift_equations]

        rm_to_f_vec = self.inv_moment_matrix * sp.Matrix(post_collision_monomial_moments)
        main_assignments = [Assignment(f, eq) for f, eq in zip(pdf_symbols, rm_to_f_vec)]

        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)
        ac.add_simplification_hint('stencil', self.stencil)
        ac.add_simplification_hint('post_collision_pdf_symbols', pdf_symbols)
        if simplification:
            ac = simplification.apply(ac)
        return ac

    #   ----------------------------- Private Members -----------------------------

    @ property
    def _default_simplification(self):
        from lbmpy.methods.momentbased.momentbasedsimplifications import (
            substitute_moments_in_conserved_quantity_equations,
            split_pdf_main_assignments_by_symmetry
        )

        cq = (self.equilibrium_density,) + self.equilibrium_velocity
        fw_skip = cq + self.pre_collision_monomial_symbols

        forward_simp = SimplificationStrategy()
        forward_simp.add(substitute_moments_in_conserved_quantity_equations)
        forward_simp.add(partial(insert_aliases, skip=fw_skip))
        forward_simp.add(partial(insert_constants, skip=fw_skip))

        bw_skip = self.post_collision_monomial_symbols

        backward_simp = SimplificationStrategy()
        backward_simp.add(partial(insert_aliases, skip=bw_skip))
        backward_simp.add(split_pdf_main_assignments_by_symmetry)
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }

# end class PdfsToMomentsByChimeraTransform
