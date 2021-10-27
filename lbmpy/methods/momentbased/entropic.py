import sympy as sp

from lbmpy.relaxationrates import get_shear_relaxation_rate
from pystencils import Assignment
from pystencils.sympyextensions import fast_subs


def add_entropy_condition(collision_rule, omega_output_field=None):
    """
    Transforms an update rule with two relaxation rate into a single relaxation rate rule, where the second
    rate is locally chosen to maximize an entropy condition. This function works for update rules which are
    linear in the relaxation rate, as all moment-based methods are. Cumulant update rules don't work since they are
    quadratic. For these, use :func:`add_iterative_entropy_condition`

    The entropy is approximated such that the optimality condition can be written explicitly, no Newton iterations
    have to be done.

    Args:
        collision_rule: collision rule with two relaxation times
        omega_output_field: pystencils field where computed omegas are stored

    Returns:
        new collision rule which only one relaxation rate
    """
    if collision_rule.method.conserved_quantity_computation.zero_centered_pdfs:
        raise NotImplementedError("Entropic Methods only implemented for models where pdfs are centered around 1. "
                                  "Use compressible=1")

    omega_s, omega_h = _get_relaxation_rates(collision_rule)

    decomposition = RelaxationRatePolynomialDecomposition(collision_rule, [omega_h], [omega_s])
    dh = []
    for entry in decomposition.relaxation_rate_factors(omega_h):
        assert len(entry) == 1, "The non-iterative entropic procedure works only for moment based methods, which have" \
                                "an update rule linear in the relaxation rate."
        dh.append(entry[0])
    ds = []
    for entry in decomposition.relaxation_rate_factors(omega_s):
        assert len(entry) <= 1, "The non-iterative entropic procedure works only for moment based methods, which have" \
                                "an update rule linear in the relaxation rate."
        if len(entry) == 0:
            entry.append(0)
        ds.append(entry[0])

    stencil = collision_rule.method.stencil
    f_symbols = collision_rule.method.pre_collision_pdf_symbols

    ds_symbols = [sp.Symbol(f"entropicDs_{i}") for i in range(stencil.Q)]
    dh_symbols = [sp.Symbol(f"entropicDh_{i}") for i in range(stencil.Q)]
    feq_symbols = [sp.Symbol(f"entropicFeq_{i}") for i in range(stencil.Q)]

    subexpressions = [Assignment(a, b) for a, b in zip(ds_symbols, ds)] + \
                     [Assignment(a, b) for a, b in zip(dh_symbols, dh)] + \
                     [Assignment(a, f_i + ds_i + dh_i) for a, f_i, ds_i, dh_i in
                      zip(feq_symbols, f_symbols, ds_symbols, dh_symbols)]

    optimal_omega_h = _get_entropy_maximizing_omega(omega_s, feq_symbols, ds_symbols, dh_symbols)

    subexpressions += [Assignment(omega_h, optimal_omega_h)]

    new_update_equations = []

    const_part = decomposition.constant_exprs()
    for update_eq in collision_rule.main_assignments:
        index = collision_rule.method.post_collision_pdf_symbols.index(update_eq.lhs)
        new_eq = Assignment(update_eq.lhs,
                            const_part[index] + omega_s * ds_symbols[index] + omega_h * dh_symbols[index])
        new_update_equations.append(new_eq)
    new_collision_rule = collision_rule.copy(new_update_equations, collision_rule.subexpressions + subexpressions)
    new_collision_rule.simplification_hints['entropic'] = True
    new_collision_rule.simplification_hints['entropic_newton_iterations'] = None

    if omega_output_field:
        new_collision_rule.main_assignments.append(Assignment(omega_output_field.center, omega_h))

    try:
        new_collision_rule.topological_sort()
    except ValueError as e:
        print("After adding the entropic condition, a cyclic dependency has been detected. This problem occurred most "
              "likely due to the use of a force model combined with the entropic method. As described by Silva et al. "
              "(https://doi.org/10.1103/PhysRevE.102.063307), most force schemes for the TRT collision operator depend "
              "on both relaxation times. However, the force is also needed to calculate the free relaxation parameter "
              "in the first place for entropic methods. Thus a cyclic dependency appears. The problem does not appear "
              "with the SIMPLE, LUO or EDM force model.")
        raise e

    return new_collision_rule


def add_iterative_entropy_condition(collision_rule, free_omega=None, newton_iterations=3, initial_value=1,
                                    omega_output_field=None):
    """
    More generic, but slower version of :func:`add_entropy_condition`

    A fixed number of Newton iterations is used to determine the maximum entropy relaxation rate.

    Args:
        collision_rule: collision rule with two relaxation times
        free_omega: relaxation rate which should be determined by entropy condition. If left to None, the
                   relaxation rate is automatically detected, which works only if there are 2 relaxation times
        newton_iterations: (integer) number of newton iterations
        initial_value: initial value of the relaxation rate
        omega_output_field: pystencils field where computed omegas are stored

    Returns:
        new collision rule which only one relaxation rate
    """

    if collision_rule.method.conserved_quantity_computation.zero_centered_pdfs:
        raise NotImplementedError("Entropic Methods only implemented for models where pdfs are centered around 1")

    if free_omega is None:
        _, free_omega = _get_relaxation_rates(collision_rule)

    decomposition = RelaxationRatePolynomialDecomposition(collision_rule, [free_omega], [])

    new_update_equations = []

    # 1) decompose into constant + free_omega * ent1 + free_omega**2 * ent2
    polynomial_subexpressions = []
    rr_polynomials = []
    for i, constant_expr in enumerate(decomposition.constant_exprs()):
        constant_expr_eq = Assignment(decomposition.symbolic_constant_expr(i), constant_expr)
        polynomial_subexpressions.append(constant_expr_eq)
        rr_polynomial = constant_expr_eq.lhs

        factors = decomposition.relaxation_rate_factors(free_omega)
        for idx, f in enumerate(factors[i]):
            power = idx + 1
            symbolic_factor = decomposition.symbolic_relaxation_rate_factors(free_omega, power)[i]
            polynomial_subexpressions.append(Assignment(symbolic_factor, f))
            rr_polynomial += free_omega ** power * symbolic_factor
        rr_polynomials.append(rr_polynomial)
        new_update_equations.append(Assignment(collision_rule.method.post_collision_pdf_symbols[i], rr_polynomial))

    # 2) get equilibrium from method and define subexpressions for it
    eq_terms = [eq.rhs for eq in collision_rule.method.get_equilibrium().main_assignments]
    eq_symbols = sp.symbols(f"entropicFeq_:{len(eq_terms)}")
    eq_subexpressions = [Assignment(a, b) for a, b in zip(eq_symbols, eq_terms)]

    # 3) find coefficients of entropy derivatives
    entropy_diff = sp.diff(discrete_approx_entropy(rr_polynomials, eq_symbols), free_omega)
    coefficients_first_diff = [c.expand() for c in reversed(sp.poly(entropy_diff, free_omega).all_coeffs())]
    sym_coeff_diff1 = sp.symbols(f"entropicDiffCoeff_:{len(coefficients_first_diff)}")
    coefficient_eqs = [Assignment(a, b) for a, b in zip(sym_coeff_diff1, coefficients_first_diff)]
    sym_coeff_diff2 = [(i + 1) * coeff for i, coeff in enumerate(sym_coeff_diff1[1:])]

    # 4) define Newtons method update iterations
    newton_iteration_equations = []
    intermediate_omegas = [sp.Symbol(f"omega_iter_{i}") for i in range(newton_iterations + 1)]
    intermediate_omegas[0] = initial_value
    intermediate_omegas[-1] = free_omega
    for omega_idx in range(len(intermediate_omegas) - 1):
        rhs_omega = intermediate_omegas[omega_idx]
        lhs_omega = intermediate_omegas[omega_idx + 1]
        diff1_poly = sum([coeff * rhs_omega ** i for i, coeff in enumerate(sym_coeff_diff1)])
        diff2_poly = sum([coeff * rhs_omega ** i for i, coeff in enumerate(sym_coeff_diff2)])
        newton_eq = Assignment(lhs_omega, rhs_omega - diff1_poly / diff2_poly)
        newton_iteration_equations.append(newton_eq)

    # 5) final update equations
    new_sub_exprs = polynomial_subexpressions + eq_subexpressions + coefficient_eqs + newton_iteration_equations
    new_collision_rule = collision_rule.copy(new_update_equations, collision_rule.subexpressions + new_sub_exprs)
    new_collision_rule.simplification_hints['entropic'] = True
    new_collision_rule.simplification_hints['entropic_newton_iterations'] = newton_iterations

    if omega_output_field:
        from lbmpy.updatekernels import write_quantities_to_field
        new_collision_rule = write_quantities_to_field(new_collision_rule, free_omega, omega_output_field)

    try:
        new_collision_rule.topological_sort()
    except ValueError as e:
        print("After adding the entropic condition, a cyclic dependency has been detected. This problem occurred most "
              "likely due to the use of a force model combined with the entropic method. As described by Silva et al. "
              "(https://doi.org/10.1103/PhysRevE.102.063307), most force schemes for the TRT collision operator depend "
              "on both relaxation times. However, the force is also needed to calculate the free relaxation parameter "
              "in the first place for entropic methods. Thus a cyclic dependency appears. The problem does not appear "
              "with the SIMPLE, LUO or EDM force model.")
        raise e

    return new_collision_rule


# --------------------------------- Helper Functions and Classes -------------------------------------------------------


def discrete_entropy(func, reference):
    r"""
    Computes relative entropy between a func :math:`f` and a reference func :math:`r`,
    which is chosen as the equilibrium for entropic methods

    .. math ::
        S = - \sum_i f_i \ln \frac{f_i}{r_i}
    """
    return -sum([f_i * sp.ln(f_i / r_i) for f_i, r_i in zip(func, reference)])


def discrete_approx_entropy(func, reference):
    r"""
    Computes an approximation of the relative entropy between a func :math:`f` and a reference func :math:`r`,
    which is chosen as the equilibrium for entropic methods. The non-approximated version is :func:`discrete_entropy`.

    This approximation assumes that the argument of the logarithm is close to 1, i.e. that the func and reference
    are close, then :math:`\ln \frac{f_i}{r_i} \approx  \frac{f_i}{r_i} - 1`

    .. math ::
        S = - \sum_i f_i \left( \frac{f_i}{r_i} - 1 \right)
    """
    return -sum([f_i * ((f_i / r_i) - 1) for f_i, r_i in zip(func, reference)])


def _get_entropy_maximizing_omega(omega_s, f_eq, ds, dh):
    ds_dh = sum([ds_i * dh_i / f_eq_i for ds_i, dh_i, f_eq_i in zip(ds, dh, f_eq)])
    dh_dh = sum([dh_i * dh_i / f_eq_i for dh_i, f_eq_i in zip(dh, f_eq)])
    return 1 - ((omega_s - 1) * ds_dh / dh_dh)


class RelaxationRatePolynomialDecomposition:

    def __init__(self, collision_rule, free_relaxation_rates, fixed_relaxation_rates):
        self._collisionRule = collision_rule
        self._free_relaxation_rates = free_relaxation_rates
        self._fixed_relaxation_rates = fixed_relaxation_rates
        self._all_relaxation_rates = fixed_relaxation_rates + free_relaxation_rates

    def symbolic_relaxation_rate_factors(self, relaxation_rate, power):
        q = len(self._collisionRule.method.stencil)
        omega_idx = self._all_relaxation_rates.index(relaxation_rate)
        return [sp.Symbol(f"entFacOmega_{i}_{omega_idx}_{power}") for i in range(q)]

    def relaxation_rate_factors(self, relaxation_rate):
        update_equations = self._collisionRule.main_assignments

        result = []
        for update_equation in update_equations:
            factors = []
            rhs = update_equation.rhs
            power = 0
            while True:
                power += 1
                factor = rhs.coeff(relaxation_rate ** power)
                if factor != 0:
                    if relaxation_rate in factor.atoms(sp.Symbol):
                        raise ValueError("Relaxation Rate decomposition failed - run simplification first")
                    factors.append(factor)
                else:
                    break

            result.append(factors)

        return result

    @staticmethod
    def symbolic_constant_expr(i):
        return sp.Symbol(f"entOffset_{i}")

    def constant_exprs(self):
        subs_dict = {rr: 0 for rr in self._free_relaxation_rates}
        subs_dict.update({rr: 0 for rr in self._fixed_relaxation_rates})
        update_equations = self._collisionRule.main_assignments
        return [fast_subs(eq.rhs, subs_dict) for eq in update_equations]

    def equilibrium_exprs(self):
        subs_dict = {rr: 1 for rr in self._free_relaxation_rates}
        subs_dict.update({rr: 1 for rr in self._fixed_relaxation_rates})
        update_equations = self._collisionRule.main_assignments
        return [fast_subs(eq.rhs, subs_dict) for eq in update_equations]

    def symbolic_equilibrium(self):
        q = len(self._collisionRule.method.stencil)
        return [sp.Symbol(f"entFeq_{i}") for i in range(q)]


def _get_relaxation_rates(collision_rule):
    sh = collision_rule.simplification_hints
    assert 'relaxation_rates' in sh, "Needs simplification hint 'relaxation_rates': Sequence of relaxation rates"

    relaxation_rates = set(sh['relaxation_rates'])
    if len(relaxation_rates) != 2:
        raise ValueError("Entropic methods can only be created for methods with two relaxation rates.\n"
                         "One free relaxation rate determining the viscosity and one to be determined by the "
                         "entropy condition")

    method = collision_rule.method
    omega_s = get_shear_relaxation_rate(method)

    # if the shear relaxation rate is not specified as a symbol look for its symbolic counter part in the subs dict
    for symbolic_rr, rr in method.subs_dict_relxation_rate.items():
        if omega_s == rr:
            omega_s = symbolic_rr

    assert omega_s in relaxation_rates

    relaxation_rates_without_omega_s = relaxation_rates - {omega_s}
    omega_h = list(relaxation_rates_without_omega_s)[0]
    return omega_s, omega_h
