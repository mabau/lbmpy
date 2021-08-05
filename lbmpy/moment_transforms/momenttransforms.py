import sympy as sp

from pystencils import Assignment, AssignmentCollection
from pystencils.simp import (
    SimplificationStrategy, add_subexpressions_for_divisions, add_subexpressions_for_constants)
from pystencils.simp.assignment_collection import SymbolGen

from lbmpy.moments import (
    moment_matrix, monomial_to_polynomial_transformation_matrix, non_aliased_polynomial_moments)
from lbmpy.moments import statistical_quantity_symbol as sq_sym

from .abstractmomenttransform import (
    AbstractMomentTransform,
    PRE_COLLISION_MOMENT, POST_COLLISION_MOMENT,
    PRE_COLLISION_RAW_MOMENT, POST_COLLISION_RAW_MOMENT
)


class PdfsToMomentsByMatrixTransform(AbstractMomentTransform):
    """Transform between populations and moment space spanned by a polynomial
    basis, using matrix-vector multiplication."""

    def __init__(self, stencil, moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 pre_collision_moment_base=PRE_COLLISION_MOMENT,
                 post_collision_moment_base=POST_COLLISION_MOMENT,
                 **kwargs):
        assert len(moment_polynomials) == len(stencil), 'Number of moments must match stencil'

        super(PdfsToMomentsByMatrixTransform, self).__init__(
            stencil, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations,
            moment_polynomials=moment_polynomials,
            **kwargs)

        self.m_pre = pre_collision_moment_base
        self.m_post = post_collision_moment_base

        self.moment_matrix = moment_matrix(self.moment_polynomials, stencil)
        self.inv_moment_matrix = self.moment_matrix.inv()

    @property
    def absorbs_conserved_quantity_equations(self):
        return False

    @property
    def pre_collision_moment_symbols(self):
        """List of symbols corresponding to the pre-collision moments
        that will be the left-hand sides of assignments returned by :func:`forward_transform`."""
        return sp.symbols(f'{self.m_pre}_:{self.q}')

    @property
    def post_collision_moment_symbols(self):
        """List of symbols corresponding to the post-collision moments
        that are input to the right-hand sides of assignments returned by:func:`backward_transform`."""
        return sp.symbols(f'{self.m_post}_:{self.q}')

    def forward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_f_to_M'):
        r"""Returns an assignment collection containing equations for pre-collision polynomial
        moments, expressed in terms of the pre-collision populations by matrix-multiplication.
        
        The moment transformation matrix :math:`M` provided by :func:`lbmpy.moments.moment_matrix` is
        used to compute the pre-collision moments as :math:`\mathbf{m}_{pre} = M \cdot \mathbf{f}_{pre}`,
        which is returned element-wise.

        Args:
            pdf_symbols: List of symbols that represent the pre-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
        """
        simplification = self._get_simp_strategy(simplification, 'forward')

        f_to_m_vec = self.moment_matrix * sp.Matrix(pdf_symbols)
        pre_collision_moments = self.pre_collision_moment_symbols
        main_assignments = [Assignment(m, eq) for m, eq in zip(pre_collision_moments, f_to_m_vec)]

        symbol_gen = SymbolGen(symbol=subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpression_symbol_generator=symbol_gen)

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, simplification=True, subexpression_base='sub_k_to_f'):
        r"""Returns an assignment collection containing equations for post-collision populations, 
        expressed in terms of the post-collision polynomial moments by matrix-multiplication.
        
        The moment transformation matrix :math:`M` provided by :func:`lbmpy.moments.moment_matrix` is
        inverted and used to compute the pre-collision moments as 
        :math:`\mathbf{f}_{\mathrm{post}} = M^{-1} \cdot \mathbf{m}_{\mathrm{post}}`, which is returned element-wise.

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
        """
        simplification = self._get_simp_strategy(simplification, 'backward')

        post_collision_moments = self.post_collision_moment_symbols
        m_to_f_vec = self.inv_moment_matrix * sp.Matrix(post_collision_moments)
        main_assignments = [Assignment(f, eq) for f, eq in zip(pdf_symbols, m_to_f_vec)]
        symbol_gen = SymbolGen(subexpression_base)

        ac = AssignmentCollection(main_assignments, subexpression_symbol_generator=symbol_gen)
        ac.add_simplification_hint('stencil', self.stencil)
        ac.add_simplification_hint('post_collision_pdf_symbols', pdf_symbols)
        if simplification:
            ac = simplification.apply(ac)
        return ac

    #   ----------------------------- Private Members -----------------------------

    @property
    def _default_simplification(self):
        forward_simp = SimplificationStrategy()
        # forward_simp.add(substitute_moments_in_conserved_quantity_equations)
        forward_simp.add(add_subexpressions_for_divisions)

        from lbmpy.methods.momentbased.momentbasedsimplifications import split_pdf_main_assignments_by_symmetry

        backward_simp = SimplificationStrategy()
        backward_simp.add(split_pdf_main_assignments_by_symmetry)
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }


# end class PdfsToMomentsByMatrixTransform

class PdfsToMomentsByChimeraTransform(AbstractMomentTransform):
    """Transform between populations and moment space spanned by a polynomial
    basis, using the raw-moment chimera transform in the forward direction and
    matrix-vector multiplication in the backward direction."""

    def __init__(self, stencil, moment_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 pre_collision_moment_base=PRE_COLLISION_MOMENT,
                 post_collision_moment_base=POST_COLLISION_MOMENT,
                 pre_collision_raw_moment_base=PRE_COLLISION_RAW_MOMENT,
                 post_collision_raw_moment_base=POST_COLLISION_RAW_MOMENT,
                 **kwargs):

        if moment_polynomials:
            #   Remove aliases
            moment_polynomials = non_aliased_polynomial_moments(moment_polynomials, stencil)

        super(PdfsToMomentsByChimeraTransform, self).__init__(
            stencil, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations,
            moment_polynomials=moment_polynomials,
            **kwargs)

        assert len(self.moment_polynomials) == len(stencil), 'Number of moments must match stencil'

        self.m_pre = pre_collision_moment_base
        self.m_post = post_collision_moment_base
        self.rm_pre = pre_collision_raw_moment_base
        self.rm_post = post_collision_raw_moment_base

        self.inv_moment_matrix = moment_matrix(self.moment_exponents, self.stencil).inv()
        self.mono_to_poly_matrix = monomial_to_polynomial_transformation_matrix(self.moment_exponents,
                                                                                self.moment_polynomials)
        self.poly_to_mono_matrix = self.mono_to_poly_matrix.inv()

    @property
    def absorbs_conserved_quantity_equations(self):
        return True

    @property
    def pre_collision_moment_symbols(self):
        """List of symbols corresponding to the pre-collision moments
        that will be the left-hand sides of assignments returned by :func:`forward_transform`."""
        return sp.symbols(f'{self.m_pre}_:{self.q}')

    @property
    def post_collision_moment_symbols(self):
        """List of symbols corresponding to the post-collision moments
        that are input to the right-hand sides of assignments returned by:func:`backward_transform`."""
        return sp.symbols(f'{self.m_post}_:{self.q}')

    @property
    def pre_collision_raw_moment_symbols(self):
        """List of symbols corresponding to the pre-collision raw (monomial) moments
        that exist as left-hand sides of subexpressions in the assignment collection 
        returned by :func:`forward_transform`."""
        return tuple(sq_sym(self.rm_pre, e) for e in self.moment_exponents)

    @property
    def post_collision_raw_moment_symbols(self):
        """List of symbols corresponding to the post-collision raw (monomial) moments
        that exist as left-hand sides of subexpressions in the assignment collection
        returned by :func:`backward_transform`."""
        return tuple(sq_sym(self.rm_post, e) for e in self.moment_exponents)

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
                          return_raw_moments=False):
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
        using the raw moment equations, if simplification is enabled.

        **Simplification**

        If simplification is enabled, the absorbed conserved quantity equations are - if possible - 
        rewritten using the raw moment symbols. If the conserved quantities originate somewhere else
        than in the lower-order moments (like from an external field), they are not affected by this.

        Args:
            pdf_symbols: List of symbols that represent the pre-collision populations
            simplification: Simplification specification. See :class:`AbstractMomentTransform`
            subexpression_base: The base name used for any subexpressions of the transformation.
            return_raw_moments: If True raw moment equations are returned as main assignments
        """

        simplification = self._get_simp_strategy(simplification, 'forward')
        raw_moment_symbol_base = self.rm_pre

        def _partial_kappa_symbol(fixed_directions, remaining_exponents):
            fixed_str = '_'.join(str(direction) for direction in fixed_directions).replace('-', 'm')
            exp_str = '_'.join(str(exp) for exp in remaining_exponents).replace('-', 'm')
            return sp.Symbol(f"partial_{raw_moment_symbol_base}_{fixed_str}_e_{exp_str}")

        partial_sums_dict = dict()
        raw_moment_eqs = []

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
                    lhs_symbol = sq_sym(raw_moment_symbol_base, exponents)
                    raw_moment_eqs.append(Assignment(lhs_symbol, summation))
                else:
                    lhs_symbol = _partial_kappa_symbol(fixed_directions, exponents[dimension:])
                    partial_sums_dict[lhs_symbol] = summation
                return lhs_symbol

        for e in self.moment_exponents:
            collect_partial_sums(e)

        main_assignments = self.cqe.main_assignments.copy() if self.cqe is not None else []
        subexpressions = self.cqe.subexpressions.copy() if self.cqe is not None else []
        subexpressions += [Assignment(lhs, rhs) for lhs, rhs in partial_sums_dict.items()]

        if return_raw_moments:
            main_assignments += raw_moment_eqs
        else:
            subexpressions += raw_moment_eqs
            moment_eqs = self.mono_to_poly_matrix * sp.Matrix(self.pre_collision_raw_moment_symbols)
            main_assignments += [Assignment(m, v) for m, v in zip(self.pre_collision_moment_symbols, moment_eqs)]

        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)
        ac.add_simplification_hint('cq_symbols_to_moments', self.get_cq_to_moment_symbols_dict(raw_moment_symbol_base))

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, simplification=True,
                           subexpression_base='sub_k_to_f',
                           start_from_raw_moments=False):
        r"""Returns an assignment collection containing equations for post-collision populations, 
        expressed in terms of the post-collision polynomial moments by matrix-multiplication.
        
        The post-collision raw moments :math:`\mathbf{m}_{\mathrm{post}}` are first obtained from the polynomials.
        Then, the raw moment transformation matrix :math:`M_r` provided by :func:`lbmpy.moments.moment_matrix` 
        is inverted and used to compute the pre-collision moments as 
        :math:`\mathbf{f}_{\mathrm{post}} = M_r^{-1} \cdot \mathbf{m}_{\mathrm{post}}`.

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
            start_from_raw_moments: If set to True the equations are not converted to monomials
        """

        simplification = self._get_simp_strategy(simplification, 'backward')

        post_collision_moments = self.post_collision_moment_symbols
        post_collision_raw_moments = self.post_collision_raw_moment_symbols

        subexpressions = []
        if not start_from_raw_moments:
            raw_moment_eqs = self.poly_to_mono_matrix * sp.Matrix(post_collision_moments)
            subexpressions += [Assignment(rm, v) for rm, v in zip(post_collision_raw_moments, raw_moment_eqs)]

        rm_to_f_vec = self.inv_moment_matrix * sp.Matrix(post_collision_raw_moments)
        main_assignments = [Assignment(f, eq) for f, eq in zip(pdf_symbols, rm_to_f_vec)]
        symbol_gen = SymbolGen(subexpression_base)

        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)
        ac.add_simplification_hint('stencil', self.stencil)
        ac.add_simplification_hint('post_collision_pdf_symbols', pdf_symbols)
        if simplification:
            ac = simplification.apply(ac)
        return ac

    #   ----------------------------- Private Members -----------------------------

    @property
    def _default_simplification(self):
        from lbmpy.methods.momentbased.momentbasedsimplifications import (
            substitute_moments_in_conserved_quantity_equations,
            split_pdf_main_assignments_by_symmetry
        )

        forward_simp = SimplificationStrategy()
        forward_simp.add(substitute_moments_in_conserved_quantity_equations)
        forward_simp.add(add_subexpressions_for_divisions)

        backward_simp = SimplificationStrategy()
        backward_simp.add(split_pdf_main_assignments_by_symmetry)
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }

# end class PdfsToMomentsByChimeraTransform
