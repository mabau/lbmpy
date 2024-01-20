import sympy as sp

from lbmpy.moment_transforms import PRE_COLLISION_MONOMIAL_CUMULANT, POST_COLLISION_CUMULANT
from lbmpy.moments import MOMENT_SYMBOLS, statistical_quantity_symbol
from lbmpy.stencils import Stencil, LBStencil
from pystencils import Assignment
from pystencils.simp.assignment_collection import AssignmentCollection
from .cumulantbasedmethod import CumulantBasedLbMethod

X, Y, Z = MOMENT_SYMBOLS
CORRECTED_FOURTH_ORDER_POLYNOMIALS = [X ** 2 * Y ** 2 - 2 * X ** 2 * Z ** 2 + Y ** 2 * Z ** 2,
                                      X ** 2 * Y ** 2 + X ** 2 * Z ** 2 - 2 * Y ** 2 * Z ** 2,
                                      X ** 2 * Y ** 2 + X ** 2 * Z ** 2 + Y ** 2 * Z ** 2,
                                      X ** 2 * Y * Z,
                                      X * Y ** 2 * Z,
                                      X * Y * Z ** 2]
FOURTH_ORDER_CORRECTION_SYMBOLS = sp.symbols("corr_fourth_:6")
FOURTH_ORDER_RELAXATION_RATE_SYMBOLS = sp.symbols("corr_rr_:10")


def add_fourth_order_correction(collision_rule, shear_relaxation_rate, bulk_relaxation_rate, limiter):
    """Adds the fourth order correction terms (:cite:`geier2017`, eq. 44-49) to a given polynomial D3Q27
    cumulant collision rule."""
    method = collision_rule.method

    if not isinstance(method, CumulantBasedLbMethod) or method.stencil != LBStencil(Stencil.D3Q27):
        raise ValueError("Fourth-order correction is only defined for D3Q27 cumulant methods.")

    polynomials = method.cumulants
    rho = method.zeroth_order_equilibrium_moment_symbol

    if not (set(CORRECTED_FOURTH_ORDER_POLYNOMIALS) < set(polynomials)):
        raise ValueError("Fourth order correction requires polynomial cumulants "
                         f"{', '.join(CORRECTED_FOURTH_ORDER_POLYNOMIALS)} to be present")

    #   Call
    #   PC1 = (x^2 * y^2 - 2 * x^2 * z^2 + y^2 * z^2),
    #   PC2 = (x^2 * y^2 + x^2 * z^2 - 2 * y^2 * z^2),
    #   PC3 = (x^2 * y^2 + x^2 * z^2 + y^2 * z^2),
    #   PC4 = (x^2 * y * z),
    #   PC5 = (x * y^2 * z),
    #   PC6 = (x * y * z^2)
    poly_symbols = [sp.Symbol(f'{POST_COLLISION_CUMULANT}_{polynomials.index(poly)}')
                    for poly in CORRECTED_FOURTH_ORDER_POLYNOMIALS]

    a_symb, b_symb = sp.symbols("a_corr, b_corr")
    a, b = get_optimal_additional_parameters(shear_relaxation_rate, bulk_relaxation_rate)
    correction_terms = get_fourth_order_correction_terms(method.relaxation_rate_dict, rho, a_symb, b_symb)
    optimal_parametrisation = get_optimal_parametrisation_with_limiters(collision_rule, shear_relaxation_rate,
                                                                        bulk_relaxation_rate, limiter)

    subexp_dict = collision_rule.subexpressions_dict
    subexp_dict = {**subexp_dict,
                   a_symb: a,
                   b_symb: b,
                   **correction_terms.subexpressions_dict,
                   **optimal_parametrisation.subexpressions_dict,
                   **correction_terms.main_assignments_dict,
                   **optimal_parametrisation.main_assignments_dict}
    for sym, cor in zip(poly_symbols, FOURTH_ORDER_CORRECTION_SYMBOLS):
        subexp_dict[sym] += cor

    collision_rule.set_sub_expressions_from_dict(subexp_dict)
    collision_rule.topological_sort()

    return collision_rule


def get_optimal_additional_parameters(shear_relaxation_rate, bulk_relaxation_rate):
    """
    Calculates the optimal parametrization for the additional parameters provided in :cite:`geier2017`
    Equations 115-116.
    """

    omega_1 = shear_relaxation_rate
    omega_2 = bulk_relaxation_rate

    omega_11 = omega_1 * omega_1
    omega_12 = omega_1 * omega_2
    omega_22 = omega_2 * omega_2

    two = sp.Float(2)
    three = sp.Float(3)
    four = sp.Float(4)
    nine = sp.Float(9)

    denom_ab = (omega_1 - omega_2) * (omega_2 * (two + three * omega_1) - sp.Float(8) * omega_1)

    a = (four * omega_11 + two * omega_12 * (omega_1 - sp.Float(6)) + omega_22 * (
        omega_1 * (sp.Float(10) - three * omega_1) - four)) / denom_ab
    b = (four * omega_12 * (nine * omega_1 - sp.Float(16)) - four * omega_11 - two * omega_22 * (
        two + nine * omega_1 * (omega_1 - two))) / (three * denom_ab)

    return a, b


def get_fourth_order_correction_terms(rrate_dict, rho, a, b):
    pc1, pc2, pc3, pc4, pc5, pc6 = CORRECTED_FOURTH_ORDER_POLYNOMIALS

    omega_1 = rrate_dict[X * Y]
    omega_2 = rrate_dict[X ** 2 + Y ** 2 + Z ** 2]

    try:
        omega_6 = rrate_dict[pc1]
        assert omega_6 == rrate_dict[pc2], \
            "Cumulants (x^2 - y^2) and (x^2 - z^2) must have the same relaxation rate"
        omega_7 = rrate_dict[pc3]
        omega_8 = rrate_dict[pc4]
        assert omega_8 == rrate_dict[pc5] == rrate_dict[pc6]
    except IndexError:
        raise ValueError("For the fourth order correction, all six polynomial cumulants"
                         "(x^2 * y^2 - 2 * x^2 * z^2 + y^2 * z^2),"
                         "(x^2 * y^2 + x^2 * z^2 - 2 * y^2 * z^2),"
                         "(x^2 * y^2 + x^2 * z^2 + y^2 * z^2),"
                         "(x^2 * y * z), (x * y^2 * z) and (x * y * z^2) must be present!")

    dxu, dyv, dzw, dxvdyu, dxwdzu, dywdzv = sp.symbols('Dx, Dy, Dz, DxvDyu, DxwDzu, DywDzv')
    c_xy = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (1, 1, 0))
    c_xz = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (1, 0, 1))
    c_yz = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (0, 1, 1))
    c_xx = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (2, 0, 0))
    c_yy = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (0, 2, 0))
    c_zz = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (0, 0, 2))

    cor1, cor2, cor3, cor4, cor5, cor6 = FOURTH_ORDER_CORRECTION_SYMBOLS

    one = sp.Float(1)
    two = sp.Float(2)
    three = sp.Float(3)

    #   Derivative Approximations
    subexpressions = [
        Assignment(dxu, - omega_1 / (two * rho) * (two * c_xx - c_yy - c_zz)
                   - omega_2 / (two * rho) * (c_xx + c_yy + c_zz - rho)),
        Assignment(dyv, dxu + (three * omega_1) / (two * rho) * (c_xx - c_yy)),
        Assignment(dzw, dxu + (three * omega_1) / (two * rho) * (c_xx - c_zz)),
        Assignment(dxvdyu, - three * omega_1 / rho * c_xy),
        Assignment(dxwdzu, - three * omega_1 / rho * c_xz),
        Assignment(dywdzv, - three * omega_1 / rho * c_yz)]

    one_half = sp.Rational(1, 2)
    one_over_three = sp.Rational(1, 3)
    two_over_three = sp.Rational(2, 3)
    four_over_three = sp.Rational(4, 3)

    #   Correction Terms
    main_assignments = [
        Assignment(cor1, two_over_three * (one / omega_1 - one_half) * omega_6 * a * rho * (dxu - two * dyv + dzw)),
        Assignment(cor2, two_over_three * (one / omega_1 - one_half) * omega_6 * a * rho * (dxu + dyv - two * dzw)),
        Assignment(cor3, - four_over_three * (one / omega_1 - one_half) * omega_7 * a * rho * (dxu + dyv + dzw)),
        Assignment(cor4, - one_over_three * (one / omega_1 - one_half) * omega_8 * b * rho * dywdzv),
        Assignment(cor5, - one_over_three * (one / omega_1 - one_half) * omega_8 * b * rho * dxwdzu),
        Assignment(cor6, - one_over_three * (one / omega_1 - one_half) * omega_8 * b * rho * dxvdyu)]

    return AssignmentCollection(main_assignments=main_assignments, subexpressions=subexpressions)


def get_optimal_parametrisation_with_limiters(collision_rule, shear_relaxation_rate, bulk_relaxation_rate, limiter):
    """
    Calculates the optimal parametrization for the relaxation rates provided in :cite:`geier2017`
    Equations 112-114.
    """

    # if omega numbers
    # assert omega_1 != omega_2, "Relaxation rates associated with shear and bulk viscosity must not be identical."
    # assert omega_1 > 7/4

    omega_1 = FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[0]
    omega_2 = FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[1]

    non_limited_omegas = sp.symbols("non_limited_omega_3:6")
    limited_omegas = sp.symbols("limited_omega_3:6")

    omega_11 = omega_1 * omega_1
    omega_12 = omega_1 * omega_2
    omega_22 = omega_2 * omega_2

    one = sp.Float(1)
    two = sp.Float(2)
    three = sp.Float(3)
    five = sp.Float(5)
    six = sp.Float(6)
    seven = sp.Float(7)
    eight = sp.Float(8)
    nine = sp.Float(9)

    omega_3 = (eight * (omega_1 - two) * (omega_2 * (three * omega_1 - one) - five * omega_1)) / (
        eight * (five - two * omega_1) * omega_1 + omega_2 * (eight + omega_1 * (nine * omega_1 - sp.Float(26))))

    omega_4 = (eight * (omega_1 - two) * (omega_1 + omega_2 * (three * omega_1 - seven))) / (
        omega_2 * (sp.Float(56) - sp.Float(42) * omega_1 + nine * omega_11) - eight * omega_1)

    omega_5 = (sp.Float(24) * (omega_1 - two) * (sp.Float(4) * omega_11 + omega_12 * (
        sp.Float(18) - sp.Float(13) * omega_1) + omega_22 * (two + omega_1 * (
            six * omega_1 - sp.Float(11))))) / (sp.Float(16) * omega_11 * (omega_1 - six) - two * omega_12 * (
                sp.Float(216) + five * omega_1 * (nine * omega_1 - sp.Float(46))) + omega_22 * (omega_1 * (
                    three * omega_1 - sp.Float(10)) * (sp.Float(15) * omega_1 - sp.Float(28)) - sp.Float(48)))

    rho = collision_rule.method.zeroth_order_equilibrium_moment_symbol

    # add limiters to improve stability
    c_xyy = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (1, 2, 0))
    c_xzz = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (1, 0, 2))
    c_xyz = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (1, 1, 1))
    abs_c_xyy_plus_xzz = sp.Abs(c_xyy + c_xzz)
    abs_c_xyy_minus_xzz = sp.Abs(c_xyy - c_xzz)
    abs_c_xyz = sp.Abs(c_xyz)

    limited_omega_3 = non_limited_omegas[0] + ((one - non_limited_omegas[0]) * abs_c_xyy_plus_xzz) / \
        (rho * limiter + abs_c_xyy_plus_xzz)
    limited_omega_4 = non_limited_omegas[1] + ((one - non_limited_omegas[1]) * abs_c_xyy_minus_xzz) / \
        (rho * limiter + abs_c_xyy_minus_xzz)
    limited_omega_5 = non_limited_omegas[2] + ((one - non_limited_omegas[2]) * abs_c_xyz) / (rho * limiter + abs_c_xyz)

    subexpressions = [
        Assignment(non_limited_omegas[0], omega_3),
        Assignment(non_limited_omegas[1], omega_4),
        Assignment(non_limited_omegas[2], omega_5),
        Assignment(limited_omegas[0], limited_omega_3),
        Assignment(limited_omegas[1], limited_omega_4),
        Assignment(limited_omegas[2], limited_omega_5)]

    #   Correction Terms
    main_assignments = [
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[0], shear_relaxation_rate),
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[1], bulk_relaxation_rate),
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[2], limited_omegas[0]),
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[3], limited_omegas[1]),
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[4], limited_omegas[2]),
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[5], one),
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[6], one),
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[7], one),
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[8], one),
        Assignment(FOURTH_ORDER_RELAXATION_RATE_SYMBOLS[9], one),
    ]

    return AssignmentCollection(main_assignments=main_assignments, subexpressions=subexpressions)
