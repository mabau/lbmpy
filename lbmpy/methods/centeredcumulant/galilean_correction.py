from pystencils.simp.assignment_collection import AssignmentCollection
import sympy as sp
from pystencils import Assignment

from lbmpy.moments import MOMENT_SYMBOLS, statistical_quantity_symbol
from lbmpy.methods.centeredcumulant.cumulant_transform import PRE_COLLISION_CUMULANT

x, y, z = MOMENT_SYMBOLS
corrected_polynomials = [x**2 - y**2, x**2 - z**2, x**2 + y**2 + z**2]


def contains_corrected_polynomials(polynomials):
    return all(cp in polynomials for cp in corrected_polynomials)


def add_galilean_correction(poly_relaxation_eqs, polynomials, correction_terms):
    #   Call PC1 = (x^2 - y^2), PC2 = (x^2 - z^2), PC3 = (x^2 + y^2 + z^2)
    try:
        index_pc1 = polynomials.index(corrected_polynomials[0])
        index_pc2 = polynomials.index(corrected_polynomials[1])
        index_pc3 = polynomials.index(corrected_polynomials[2])
    except ValueError:
        raise ValueError("For the galilean correction, all three polynomial cumulants"
                         + "(x^2 - y^2), (x^2 - z^2) and (x^2 + y^2 + z^2) need to be present!")

    cor1 = correction_terms.main_assignments[0].lhs
    cor2 = correction_terms.main_assignments[1].lhs
    cor3 = correction_terms.main_assignments[2].lhs

    poly_relaxation_eqs[index_pc1] += cor1
    poly_relaxation_eqs[index_pc2] += cor2
    poly_relaxation_eqs[index_pc3] += cor3

    return poly_relaxation_eqs


def get_galilean_correction_terms(cumulant_to_relaxation_info_dict, rho, u_vector,
                                  pre_collision_cumulant_base=PRE_COLLISION_CUMULANT):

    pc1 = corrected_polynomials[0]
    pc2 = corrected_polynomials[1]
    pc3 = corrected_polynomials[2]

    try:
        omega_1 = cumulant_to_relaxation_info_dict[pc1].relaxation_rate
        assert omega_1 == cumulant_to_relaxation_info_dict[pc2].relaxation_rate, \
            "Cumulants (x^2 - y^2) and (x^2 - z^2) must have the same relaxation rate"
        omega_2 = cumulant_to_relaxation_info_dict[pc3].relaxation_rate
    except IndexError:
        raise ValueError("For the galilean correction, all three polynomial cumulants"
                         + "(x^2 - y^2), (x^2 - z^2) and (x^2 + y^2 + z^2) must be present!")

    dx, dy, dz = sp.symbols('Dx, Dy, Dz')
    c_xx = statistical_quantity_symbol(pre_collision_cumulant_base, (2, 0, 0))
    c_yy = statistical_quantity_symbol(pre_collision_cumulant_base, (0, 2, 0))
    c_zz = statistical_quantity_symbol(pre_collision_cumulant_base, (0, 0, 2))

    cor1, cor2, cor3 = sp.symbols("corr_:3")

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
