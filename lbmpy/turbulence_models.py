import sympy as sp

from lbmpy.relaxationrates import get_shear_relaxation_rate
from pystencils import Assignment


def second_order_moment_tensor(function_values, stencil):
    """Returns (D x D) Matrix of second order moments of the given function where D is the dimension"""
    assert len(function_values) == len(stencil)
    dim = len(stencil[0])
    return sp.Matrix(dim, dim, lambda i, j: sum(c[i] * c[j] * f for f, c in zip(function_values, stencil)))


def frobenius_norm(matrix, factor=1):
    """Computes the Frobenius norm of a matrix defined as the square root of the sum of squared matrix elements
    The optional factor is added inside the square root"""
    return sp.sqrt(sum(i * i for i in matrix) * factor)


def add_smagorinsky_model(collision_rule, smagorinsky_constant, omega_output_field=None):
    method = collision_rule.method
    omega_s = get_shear_relaxation_rate(method)
    f_neq = sp.Matrix(method.pre_collision_pdf_symbols) - method.get_equilibrium_terms()

    tau_0 = sp.Symbol("tau_0_")
    second_order_neq_moments = sp.Symbol("Pi")
    rho = method.zeroth_order_equilibrium_moment_symbol if method.conserved_quantity_computation.compressible else 1
    adapted_omega = sp.Symbol("smagorinsky_omega")

    collision_rule = collision_rule.new_with_substitutions({omega_s: adapted_omega})
    # for derivation see notebook demo_custom_LES_model.ipynb
    eqs = [Assignment(tau_0, 1 / omega_s),
           Assignment(second_order_neq_moments,
                      frobenius_norm(second_order_moment_tensor(f_neq, method.stencil), factor=2) / rho),
           Assignment(adapted_omega,
                      2 / (tau_0 + sp.sqrt(18 * smagorinsky_constant ** 2 * second_order_neq_moments + tau_0 ** 2)))]
    collision_rule.subexpressions += eqs
    collision_rule.topological_sort(sort_subexpressions=True, sort_main_assignments=False)

    if omega_output_field:
        collision_rule.main_assignments.append(Assignment(omega_output_field.center, adapted_omega))

    return collision_rule
