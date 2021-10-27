import sympy as sp

from lbmpy.relaxationrates import get_shear_relaxation_rate
from pystencils import Assignment


def second_order_moment_tensor(function_values, stencil):
    """Returns (D x D) Matrix of second order moments of the given function where D is the dimension"""
    assert len(function_values) == stencil.Q
    return sp.Matrix(stencil.D, stencil.D, lambda i, j: sum(c[i] * c[j] * f for f, c in zip(function_values, stencil)))


def frobenius_norm(matrix, factor=1):
    """Computes the Frobenius norm of a matrix defined as the square root of the sum of squared matrix elements
    The optional factor is added inside the square root"""
    return sp.sqrt(sum(i * i for i in matrix) * factor)


def add_smagorinsky_model(collision_rule, smagorinsky_constant, omega_output_field=None):
    r""" Adds a smagorinsky model to a lattice Boltzmann collision rule. To add the Smagorinsky model to a LB scheme
        one has to first compute the strain rate tensor $S_{ij}$ in each cell, and compute the turbulent
        viscosity :math:`nu_t` from it. Then the local relaxation rate has to be adapted to match the total viscosity
        :math `\nu_{total}` instead of the standard viscosity :math `\nu_0`.

        A fortunate property of LB methods is, that the strain rate tensor can be computed locally from the
        non-equilibrium part of the distribution function. This is somewhat surprising, since the strain rate tensor
        contains first order derivatives. The strain rate tensor can be obtained by

        .. math ::
            S_{ij} = - \frac{3 \omega_s}{2 \rho_{(0)}} \Pi_{ij}^{(neq)}

        where :math `\omega_s` is the relaxation rate that determines the viscosity, :math `\rho_{(0)}` is :math `\rho`
        in compressible models and :math `1` for incompressible schemes.
        :math `\Pi_{ij}^{(neq)}` is the second order moment tensor of the non-equilibrium part of
        the distribution functions
        :math `f^{(neq)} = f - f^{(eq)}` and can be computed as

        .. math ::
            \Pi_{ij}^{(neq)} = \sum_q c_{qi} c_{qj} \; f_q^{(neq)}


    """
    method = collision_rule.method
    omega_s = get_shear_relaxation_rate(method)

    found_symbolic_shear_relaxation = True
    if isinstance(omega_s, float) or isinstance(omega_s, int):
        found_symbolic_shear_relaxation = False
        for eq in collision_rule.all_assignments:
            if eq.rhs == omega_s:
                found_symbolic_shear_relaxation = True
                omega_s = eq.lhs

    if not found_symbolic_shear_relaxation:
        raise ValueError("For the smagorinsky model the shear relaxation rate has to be a symbol or it has to be "
                         "assigned to a single equation in the assignment list")
    f_neq = sp.Matrix(method.pre_collision_pdf_symbols) - method.get_equilibrium_terms()

    tau_0 = sp.Symbol("tau_0_")
    second_order_neq_moments = sp.Symbol("Pi")
    rho = method.zeroth_order_equilibrium_moment_symbol if method.conserved_quantity_computation.compressible else 1
    adapted_omega = sp.Symbol("smagorinsky_omega")

    collision_rule = collision_rule.new_with_substitutions({omega_s: adapted_omega}, substitute_on_lhs=False)
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
