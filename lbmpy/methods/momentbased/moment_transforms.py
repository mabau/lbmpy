from abc import abstractmethod
import sympy as sp

from pystencils import Assignment, AssignmentCollection
from pystencils.simp import (
    SimplificationStrategy, sympy_cse, add_subexpressions_for_divisions, add_subexpressions_for_constants)
from pystencils.simp.assignment_collection import SymbolGen
from pystencils.sympyextensions import subs_additive, fast_subs

from lbmpy.moments import moment_matrix, set_up_shift_matrix, contained_moments, moments_up_to_order

from lbmpy.methods.momentbased.momentbasedsimplifications import (
    substitute_moments_in_conserved_quantity_equations,
    split_pdf_main_assignments_by_symmetry)
from lbmpy.methods.centeredcumulant.centered_cumulants import statistical_quantity_symbol as sq_sym


#   ============================ PDFs <-> Central Moments ==============================================================

PRE_COLLISION_RAW_MOMENT = 'm'
POST_COLLISION_RAW_MOMENT = 'm_post'

PRE_COLLISION_CENTRAL_MOMENT = 'kappa'
POST_COLLISION_CENTRAL_MOMENT = 'kappa_post'


class AbstractMomentTransform:
    r"""
    Abstract Base Class for classes providing transformations between moment spaces. These transformations 
    are bijective maps between two spaces :math:`\mathcal{S}` and :math:`\mathcal{D}` (i.e. populations
    and raw moments, or central moments and cumulants). The forward map 
    :math:`F : \mathcal{S} \mapsto \mathcal{D}`
    is given by `forward_transform`, and the backward (inverse) map
    :math:`F^{-1} : \mathcal{D} \mapsto \mathcal{S}`
    is provided by `backward_transform`. It is intendet to use the transformations in lattice Boltzmann collision
    operators: The `forward_transform` to map pre-collision populations to the required moment
    space (possibly by several consecutive transformations), and the `backward_transform` to map post-
    collision quantities back to populations.

    ### Transformations

    Transformation equations must be returned by implementations of `forward_transform` and
    `backward_transform` as `AssignmentCollection`s.

    - `forward_transform` returns an AssignmentCollection which depends on quantities of the domain 
      :math:`\mathcal{S}` and contains the equations to map them to the codomain :math:`\mathcal{D}`.

    - `backward_transform` is the inverse of `forward_transform` and returns an AssignmentCollection 
      which maps quantities of the codomain :math:`\mathcal{D}` back to the domain :math:`\mathcal{S}`.

    ### Absorption of Conserved Quantity Equations

    Transformations from the population space to any moment space may *absorb* the equations defining
    the macroscopic quantities entering the equilibrium (typically the density :math:`\rho` and the
    velocity :math:`\vec{u}`). This means that the `forward_transform` will possibly rewrite the 
    assignments given in the constructor argument `conserved_quantity_equations` to reduce
    the total operation count. For example, in the transformation step from populations to
    raw moments (see `PdfsToRawMomentsTransform`), :math:`\rho` can be aliased as the zeroth-order moment
    :math:`m_{000}`. Assignments to the conserved quantities will then be part of the AssignmentCollection
    returned by `forward_transform` and need not be added to the collision rule separately. 

    ### Simplification

    Both `forward_transform` and `backward_transform` expect a keyword argument `simplification`
    which can be used to direct simplification steps applied during the derivation of the transformation
    equations. Possible values are:
     - `False` or `'none'`: No simplification is to be applied
     - `True` or `'default'`: A default simplification strategy specific to the implementation is applied.
       The actual simplification steps depend strongly on the nature of the equations. They are defined by
       the implementation. It is the responsibility of the implementation to select the most effective
       simplification strategy.
     - `'default_with_cse'`: Same as `'default'`, but with an additional pass of common subexpression elimination.


    """

    def __init__(self, stencil, moment_exponents,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 **kwargs):
        self.moment_exponents = sorted(list(moment_exponents), key=sum)
        self.stencil = stencil
        self.dim = len(stencil[0])
        self.cqe = conserved_quantity_equations
        self.equilibrium_density = equilibrium_density
        self.equilibrium_velocity = equilibrium_velocity

    @abstractmethod
    def forward_transform(self, *args, **kwargs):
        raise NotImplementedError("forward_transform must be implemented in a subclass")

    @abstractmethod
    def backward_transform(self, *args, **kwargs):
        raise NotImplementedError("backward_transform must be implemented in a subclass")

    @property
    def absorbs_conserved_quantity_equations(self):
        """
        Whether or not the given conserved quantity equations will be included in
        the assignment collection returned by `forward_transform`, possibly in simplified
        form.
        """
        return False

    @property
    def _default_simplification(self):
        return SimplificationStrategy()

    def _get_simp_strategy(self, simplification, direction=None):
        if isinstance(simplification, bool):
            simplification = 'default' if simplification else 'none'

        if simplification == 'default' or simplification == 'default_with_cse':
            simp = self._default_simplification if direction is None else self._default_simplification[direction]
            if simplification == 'default_with_cse':
                simp.add(sympy_cse)
            return simp
        else:
            return None


class PdfsToCentralMomentsByMatrix(AbstractMomentTransform):

    def __init__(self, stencil, moment_exponents, equilibrium_density, equilibrium_velocity, **kwargs):
        assert len(moment_exponents) == len(stencil), 'Number of moments must match stencil'

        super(PdfsToCentralMomentsByMatrix, self).__init__(
            stencil, moment_exponents, equilibrium_density, equilibrium_velocity, **kwargs)

        moment_matrix_without_shift = moment_matrix(self.moment_exponents, self.stencil)
        shift_matrix = set_up_shift_matrix(self.moment_exponents, self.stencil, equilibrium_velocity)

        self.forward_matrix = moment_matrix(self.moment_exponents, self.stencil, equilibrium_velocity)
        self.backward_matrix = moment_matrix_without_shift.inv() * shift_matrix.inv()

    def forward_transform(self, pdf_symbols, moment_symbol_base=PRE_COLLISION_CENTRAL_MOMENT,
                          simplification=True, subexpression_base='sub_f_to_k'):
        simplification = self._get_simp_strategy(simplification)

        f_vec = sp.Matrix(pdf_symbols)
        central_moments = self.forward_matrix * f_vec
        main_assignments = [Assignment(sq_sym(moment_symbol_base, e), eq)
                            for e, eq in zip(self.moment_exponents, central_moments)]
        symbol_gen = SymbolGen(subexpression_base)

        ac = AssignmentCollection(main_assignments, subexpression_symbol_generator=symbol_gen)
        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, moment_symbol_base=POST_COLLISION_CENTRAL_MOMENT,
                           simplification=True, subexpression_base='sub_k_to_f'):
        simplification = self._get_simp_strategy(simplification)

        moments = [sq_sym(moment_symbol_base, exp) for exp in self.moment_exponents]
        moment_vec = sp.Matrix(moments)
        pdfs_from_moments = self.backward_matrix * moment_vec
        main_assignments = [Assignment(f, eq) for f, eq in zip(pdf_symbols, pdfs_from_moments)]
        symbol_gen = SymbolGen(subexpression_base)

        ac = AssignmentCollection(main_assignments, subexpression_symbol_generator=symbol_gen)
        if simplification:
            ac = simplification.apply(ac)
        return ac

    @property
    def _default_simplification(self):
        simplification = SimplificationStrategy()
        simplification.add(add_subexpressions_for_divisions)
        return simplification
# end class PdfsToCentralMomentsByMatrix


class FastCentralMomentTransform(AbstractMomentTransform):

    def __init__(self, stencil,
                 moment_exponents,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 **kwargs):
        assert len(moment_exponents) == len(stencil), 'Number of moments must match stencil'

        super(FastCentralMomentTransform, self).__init__(
            stencil, moment_exponents, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations, **kwargs)
        self.mat_transform = PdfsToCentralMomentsByMatrix(
            stencil, moment_exponents, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations, **kwargs)

    def forward_transform(self, pdf_symbols, moment_symbol_base=PRE_COLLISION_CENTRAL_MOMENT,
                          simplification=True, subexpression_base='sub_f_to_k'):
        simplification = self._get_simp_strategy(simplification, 'forward')

        def _partial_kappa_symbol(fixed_directions, remaining_exponents):
            fixed_str = '_'.join(str(direction) for direction in fixed_directions).replace('-', 'm')
            exp_str = '_'.join(str(exp) for exp in remaining_exponents).replace('-', 'm')
            return sp.Symbol(f"partial_kappa_{fixed_str}_e_{exp_str}")

        subexpressions_dict = dict()
        main_assignments = []

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
                    lhs_symbol = sq_sym(moment_symbol_base, exponents)
                    main_assignments.append(Assignment(lhs_symbol, summation))
                else:
                    lhs_symbol = _partial_kappa_symbol(fixed_directions, exponents[dimension:])
                    subexpressions_dict[lhs_symbol] = summation
                return lhs_symbol

        for e in self.moment_exponents:
            collect_partial_sums(e)

        subexpressions = [Assignment(lhs, rhs) for lhs, rhs in subexpressions_dict.items()]
        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)
        if simplification:
            ac = self._simplify_lower_order_moments(ac, moment_symbol_base)
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, moment_symbol_base=POST_COLLISION_CENTRAL_MOMENT,
                           simplification=True, subexpression_base='sub_k_to_f'):
        simplification = self._get_simp_strategy(simplification, 'backward')

        raw_equations = self.mat_transform.backward_transform(
            pdf_symbols, moment_symbol_base=POST_COLLISION_CENTRAL_MOMENT, simplification=False)
        raw_equations = raw_equations.new_without_subexpressions()

        symbol_gen = SymbolGen(subexpression_base)

        ac = self._split_backward_equations(raw_equations, symbol_gen)
        if simplification:
            ac = simplification.apply(ac)
        return ac

    #   ----------------------------- Private Members -----------------------------

    @property
    def _default_simplification(self):
        forward_simp = SimplificationStrategy()
        forward_simp.add(add_subexpressions_for_divisions)

        backward_simp = SimplificationStrategy()
        backward_simp.add(add_subexpressions_for_divisions)
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }

    def _simplify_lower_order_moments(self, ac, moment_base):
        if self.cqe is None:
            return ac

        f_to_cm_dict = ac.main_assignments_dict
        f_to_cm_dict_reduced = ac.new_without_subexpressions().main_assignments_dict

        moment_symbols = [sq_sym(moment_base, e) for e in moments_up_to_order(1, dim=self.dim)]
        cqe_subs = self.cqe.new_without_subexpressions().main_assignments_dict
        for m in moment_symbols:
            m_eq = fast_subs(fast_subs(f_to_cm_dict_reduced[m], cqe_subs), cqe_subs)
            m_eq = m_eq.expand().cancel()
            for cqe_sym, cqe_exp in cqe_subs.items():
                m_eq = subs_additive(m_eq, cqe_sym, cqe_exp)
            f_to_cm_dict[m] = m_eq

        main_assignments = [Assignment(lhs, rhs) for lhs, rhs in f_to_cm_dict.items()]
        return ac.copy(main_assignments=main_assignments)

    def _split_backward_equations_recursive(self, assignment, all_subexpressions,
                                            stencil_direction, subexp_symgen, known_coeffs_dict,
                                            step=0):
        #   Base Case
        if step == self.dim:
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


class PdfsToRawMomentsTransform(AbstractMomentTransform):

    def __init__(self, stencil, moment_exponents,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 **kwargs):
        assert len(moment_exponents) == len(stencil), 'Number of moments must match stencil'

        super(PdfsToRawMomentsTransform, self).__init__(
            stencil, moment_exponents, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations,
            **kwargs)

        self.inv_moment_matrix = moment_matrix(self.moment_exponents, self.stencil).inv()

    @property
    def absorbs_conserved_quantity_equations(self):
        return True

    def get_cq_to_moment_symbols_dict(self, moment_symbol_base):
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

    def forward_transform(self, pdf_symbols, moment_symbol_base=PRE_COLLISION_RAW_MOMENT,
                          simplification=True, subexpression_base='sub_f_to_m'):
        simplification = self._get_simp_strategy(simplification, 'forward')

        def _partial_kappa_symbol(fixed_directions, remaining_exponents):
            fixed_str = '_'.join(str(direction) for direction in fixed_directions).replace('-', 'm')
            exp_str = '_'.join(str(exp) for exp in remaining_exponents).replace('-', 'm')
            return sp.Symbol(f"partial_{moment_symbol_base}_{fixed_str}_e_{exp_str}")

        partial_sums_dict = dict()
        main_assignments = self.cqe.main_assignments.copy() if self.cqe is not None else []
        subexpressions = self.cqe.subexpressions.copy() if self.cqe is not None else []

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
                    lhs_symbol = sq_sym(moment_symbol_base, exponents)
                    main_assignments.append(Assignment(lhs_symbol, summation))
                else:
                    lhs_symbol = _partial_kappa_symbol(fixed_directions, exponents[dimension:])
                    partial_sums_dict[lhs_symbol] = summation
                return lhs_symbol

        for e in self.moment_exponents:
            collect_partial_sums(e)

        subexpressions += [Assignment(lhs, rhs) for lhs, rhs in partial_sums_dict.items()]
        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)
        ac.add_simplification_hint('cq_symbols_to_moments', self.get_cq_to_moment_symbols_dict(moment_symbol_base))

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols, moment_symbol_base=POST_COLLISION_RAW_MOMENT,
                           simplification=True, subexpression_base='sub_k_to_f'):
        simplification = self._get_simp_strategy(simplification, 'backward')

        post_collision_moments = [sq_sym(moment_symbol_base, e) for e in self.moment_exponents]
        rm_to_f_vec = self.inv_moment_matrix * sp.Matrix(post_collision_moments)
        main_assignments = [Assignment(f, eq) for f, eq in zip(pdf_symbols, rm_to_f_vec)]
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
        forward_simp.add(substitute_moments_in_conserved_quantity_equations)
        forward_simp.add(add_subexpressions_for_divisions)

        backward_simp = SimplificationStrategy()
        backward_simp.add(split_pdf_main_assignments_by_symmetry)
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }

# end class PdfsToRawMomentsTransform


class PdfsToCentralMomentsByShiftMatrix(AbstractMomentTransform):
    def __init__(self, stencil, moment_exponents,
                 equilibrium_density,
                 equilibrium_velocity,
                 conserved_quantity_equations=None,
                 **kwargs):
        assert len(moment_exponents) == len(stencil), 'Number of moments must match stencil'

        super(PdfsToCentralMomentsByShiftMatrix, self).__init__(
            stencil, moment_exponents, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations,
            **kwargs)
        self.raw_moment_transform = PdfsToRawMomentsTransform(
            stencil, moment_exponents, equilibrium_density, equilibrium_velocity,
            conserved_quantity_equations=conserved_quantity_equations,
            **kwargs)
        self.shift_matrix = set_up_shift_matrix(self.moment_exponents, self.stencil, self.equilibrium_velocity)
        self.inv_shift_matrix = self.shift_matrix.inv()

    @property
    def absorbs_conserved_quantity_equations(self):
        return True

    def forward_transform(self, pdf_symbols,
                          moment_symbol_base=PRE_COLLISION_CENTRAL_MOMENT,
                          simplification=True,
                          subexpression_base='sub_f_to_k',
                          raw_moment_base=PRE_COLLISION_RAW_MOMENT):
        simplification = self._get_simp_strategy(simplification, 'forward')

        central_moment_base = moment_symbol_base

        symbolic_rms = [sq_sym(raw_moment_base, e) for e in self.moment_exponents]
        symbolic_cms = [sq_sym(central_moment_base, e) for e in self.moment_exponents]

        rm_ac = self.raw_moment_transform.forward_transform(pdf_symbols, raw_moment_base, False, subexpression_base)
        cq_symbols_to_moments = self.raw_moment_transform.get_cq_to_moment_symbols_dict(raw_moment_base)
        rm_to_cm_vec = self.shift_matrix * sp.Matrix(symbolic_rms)

        cq_subs = dict()
        if simplification:
            rm_ac = substitute_moments_in_conserved_quantity_equations(rm_ac)

            #   Compute replacements for conserved moments in terms of the CQE
            rm_asm_dict = rm_ac.main_assignments_dict
            for cq_sym, moment_sym in cq_symbols_to_moments.items():
                cq_eq = rm_asm_dict[cq_sym]
                solutions = sp.solve(cq_eq - cq_sym, moment_sym)
                if len(solutions) > 0:
                    cq_subs[moment_sym] = solutions[0]

            rm_to_cm_vec = fast_subs(rm_to_cm_vec, cq_subs)

        rm_to_cm_dict = {cm: rm for cm, rm in zip(symbolic_cms, rm_to_cm_vec)}

        if simplification:
            rm_to_cm_dict = self._simplify_raw_to_central_moments(
                rm_to_cm_dict, self.moment_exponents, raw_moment_base, central_moment_base)
            rm_to_cm_dict = self._undo_remaining_cq_subexpressions(rm_to_cm_dict, cq_subs)

        subexpressions = rm_ac.all_assignments
        symbol_gen = SymbolGen(subexpression_base, dtype=float)
        ac = AssignmentCollection(rm_to_cm_dict, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self, pdf_symbols,
                           moment_symbol_base=POST_COLLISION_CENTRAL_MOMENT,
                           simplification=True,
                           subexpression_base='sub_k_to_f',
                           raw_moment_base=POST_COLLISION_RAW_MOMENT):
        simplification = self._get_simp_strategy(simplification, 'backward')

        central_moment_base = moment_symbol_base

        symbolic_rms = [sq_sym(raw_moment_base, e) for e in self.moment_exponents]
        symbolic_cms = [sq_sym(central_moment_base, e) for e in self.moment_exponents]

        cm_to_rm_vec = self.inv_shift_matrix * sp.Matrix(symbolic_cms)
        cm_to_rm_dict = {rm: eq for rm, eq in zip(symbolic_rms, cm_to_rm_vec)}

        if simplification:
            cm_to_rm_dict = self._factor_backward_eqs_by_velocities(symbolic_rms, cm_to_rm_dict)

        rm_ac = self.raw_moment_transform.backward_transform(pdf_symbols, raw_moment_base, False, subexpression_base)
        cm_to_rm_assignments = [Assignment(lhs, rhs) for lhs, rhs in cm_to_rm_dict.items()]
        subexpressions = cm_to_rm_assignments + rm_ac.subexpressions
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
        forward_simp.add(add_subexpressions_for_divisions)

        backward_simp = SimplificationStrategy()
        backward_simp.add(split_pdf_main_assignments_by_symmetry)
        backward_simp.add(add_subexpressions_for_constants)

        return {
            'forward': forward_simp,
            'backward': backward_simp
        }

# end class PdfsToCentralMomentsByShiftMatrix
