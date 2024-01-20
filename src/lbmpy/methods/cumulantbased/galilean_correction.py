import sympy as sp

from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils import Assignment

from lbmpy.stencils import Stencil, LBStencil
from lbmpy.moments import MOMENT_SYMBOLS, statistical_quantity_symbol
from lbmpy.moment_transforms import PRE_COLLISION_MONOMIAL_CUMULANT, POST_COLLISION_CUMULANT

from .cumulantbasedmethod import CumulantBasedLbMethod

X, Y, Z = MOMENT_SYMBOLS
CORRECTED_POLYNOMIALS = [X**2 - Y**2, X**2 - Z**2, X**2 + Y**2 + Z**2]
CORRECTION_SYMBOLS = sp.symbols("corr_:3")


def contains_corrected_polynomials(polynomials):
    return all(cp in polynomials for cp in CORRECTED_POLYNOMIALS)


def add_galilean_correction(collision_rule):
    """Adds the galilean correction terms (:cite:`geier2015`, eq. 58-63) to a given polynomial D3Q27
    cumulant collision rule."""
    method = collision_rule.method

    if not isinstance(method, CumulantBasedLbMethod) or method.stencil != LBStencil(Stencil.D3Q27):
        raise ValueError("Galilean correction is only defined for D3Q27 cumulant methods.")

    polynomials = method.cumulants
    rho = method.zeroth_order_equilibrium_moment_symbol
    u = method.first_order_equilibrium_moment_symbols

    if not (set(CORRECTED_POLYNOMIALS) < set(polynomials)):
        raise ValueError("Galilean correction requires polynomial cumulants "
                         f"{', '.join(CORRECTED_POLYNOMIALS)} to be present")

    #   Call PC1 = (x^2 - y^2), PC2 = (x^2 - z^2), PC3 = (x^2 + y^2 + z^2)
    poly_symbols = [sp.Symbol(f'{POST_COLLISION_CUMULANT}_{polynomials.index(poly)}')
                    for poly in CORRECTED_POLYNOMIALS]

    correction_terms = get_galilean_correction_terms(method.relaxation_rate_dict, rho, u)

    subexp_dict = collision_rule.subexpressions_dict
    subexp_dict = {**subexp_dict,
                   **correction_terms.subexpressions_dict,
                   **correction_terms.main_assignments_dict}
    for sym, cor in zip(poly_symbols, CORRECTION_SYMBOLS):
        subexp_dict[sym] += cor

    collision_rule.set_sub_expressions_from_dict(subexp_dict)
    collision_rule.topological_sort()

    return collision_rule


def get_galilean_correction_terms(rrate_dict, rho, u_vector):

    pc1 = CORRECTED_POLYNOMIALS[0]
    pc2 = CORRECTED_POLYNOMIALS[1]
    pc3 = CORRECTED_POLYNOMIALS[2]

    try:
        omega_1 = rrate_dict[pc1]
        assert omega_1 == rrate_dict[pc2], \
            "Cumulants (x^2 - y^2) and (x^2 - z^2) must have the same relaxation rate"
        omega_2 = rrate_dict[pc3]
    except IndexError:
        raise ValueError("For the galilean correction, all three polynomial cumulants"
                         + "(x^2 - y^2), (x^2 - z^2) and (x^2 + y^2 + z^2) must be present!")

    dx, dy, dz = sp.symbols('Dx, Dy, Dz')
    c_xx = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (2, 0, 0))
    c_yy = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (0, 2, 0))
    c_zz = statistical_quantity_symbol(PRE_COLLISION_MONOMIAL_CUMULANT, (0, 0, 2))

    cor1, cor2, cor3 = CORRECTION_SYMBOLS

    #   Derivative Approximations
    subexpressions = [
        Assignment(dx, - omega_1 / (2 * rho) * (2 * c_xx - c_yy - c_zz)
                   - omega_2 / (2 * rho) * (c_xx + c_yy + c_zz - rho)),
        Assignment(dy, dx + (3 * omega_1) / (2 * rho) * (c_xx - c_yy)),
        Assignment(dz, dx + (3 * omega_1) / (2 * rho) * (c_xx - c_zz))]

    #   Correction Terms
    main_assignments = [
        Assignment(cor1, - 3 * rho * (1 - omega_1 / 2) * (u_vector[0]**2 * dx - u_vector[1]**2 * dy)),
        Assignment(cor2, - 3 * rho * (1 - omega_1 / 2) * (u_vector[0]**2 * dx - u_vector[2]**2 * dz)),
        Assignment(cor3, - 3 * rho * (1 - omega_2 / 2)
                   * (u_vector[0]**2 * dx + u_vector[1]**2 * dy + u_vector[2]**2 * dz))]
    return AssignmentCollection(main_assignments=main_assignments, subexpressions=subexpressions)
