import sympy as sp

from lbmpy.relaxationrates import get_shear_relaxation_rate, relaxation_rate_from_lattice_viscosity, \
    lattice_viscosity_from_relaxation_rate
from lbmpy.utils import extract_shear_relaxation_rate, frobenius_norm, second_order_moment_tensor
from pystencils import Assignment

from lbmpy.enums import SubgridScaleModel


def add_sgs_model(collision_rule, subgrid_scale_model: SubgridScaleModel, model_constant=None, omega_output_field=None,
                  eddy_viscosity_field=None):
    r""" Wrapper for SGS models to provide default constants and outsource SGS model handling from creation routines."""

    if subgrid_scale_model == SubgridScaleModel.SMAGORINSKY:
        model_constant = model_constant if model_constant else sp.Float(0.12)
        return add_smagorinsky_model(collision_rule=collision_rule, smagorinsky_constant=model_constant,
                                     omega_output_field=omega_output_field, eddy_viscosity_field=eddy_viscosity_field)
    if subgrid_scale_model == SubgridScaleModel.QR:
        model_constant = model_constant if model_constant else sp.Rational(1, 3)
        return add_qr_model(collision_rule=collision_rule, qr_constant=model_constant,
                            omega_output_field=omega_output_field, eddy_viscosity_field=eddy_viscosity_field)


def add_smagorinsky_model(collision_rule, smagorinsky_constant, omega_output_field=None, eddy_viscosity_field=None):
    r""" Adds a Smagorinsky model to a lattice Boltzmann collision rule. To add the Smagorinsky model to a LB scheme
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
    omega_s, found_symbolic_shear_relaxation = extract_shear_relaxation_rate(collision_rule, omega_s)

    if not found_symbolic_shear_relaxation:
        raise ValueError("For the Smagorinsky model the shear relaxation rate has to be a symbol or it has to be "
                         "assigned to a single equation in the assignment list")
    f_neq = sp.Matrix(method.pre_collision_pdf_symbols) - method.get_equilibrium_terms()
    equilibrium = method.equilibrium_distribution

    tau_0 = sp.Symbol("tau_0_")
    second_order_neq_moments = sp.Symbol("Pi")
    rho = equilibrium.density if equilibrium.compressible else equilibrium.background_density
    adapted_omega = sp.Symbol("sgs_omega")

    collision_rule = collision_rule.new_with_substitutions({omega_s: adapted_omega}, substitute_on_lhs=False)
    # for derivation see notebook demo_custom_LES_model.ipynb
    eqs = [Assignment(tau_0, sp.Float(1) / omega_s),
           Assignment(second_order_neq_moments,
                      frobenius_norm(second_order_moment_tensor(f_neq, method.stencil), factor=2) / rho),
           Assignment(adapted_omega,
                      sp.Float(2) / (tau_0 + sp.sqrt(
                          sp.Float(18) * smagorinsky_constant ** 2 * second_order_neq_moments + tau_0 ** 2)))]
    collision_rule.subexpressions += eqs
    collision_rule.topological_sort(sort_subexpressions=True, sort_main_assignments=False)

    if eddy_viscosity_field:
        collision_rule.main_assignments.append(Assignment(
            eddy_viscosity_field.center,
            smagorinsky_constant ** 2 * sp.Rational(3, 2) * adapted_omega * second_order_neq_moments
        ))

    if omega_output_field:
        collision_rule.main_assignments.append(Assignment(omega_output_field.center, adapted_omega))

    return collision_rule


def add_qr_model(collision_rule, qr_constant, omega_output_field=None, eddy_viscosity_field=None):
    r""" Adds a QR model to a lattice Boltzmann collision rule, see :cite:`rozema15`.
    WARNING : this subgrid-scale model is only defined for isotropic grids
    """
    method = collision_rule.method
    omega_s = get_shear_relaxation_rate(method)
    omega_s, found_symbolic_shear_relaxation = extract_shear_relaxation_rate(collision_rule, omega_s)

    if not found_symbolic_shear_relaxation:
        raise ValueError("For the QR model the shear relaxation rate has to be a symbol or it has to be "
                         "assigned to a single equation in the assignment list")
    f_neq = sp.Matrix(method.pre_collision_pdf_symbols) - method.get_equilibrium_terms()
    equilibrium = method.equilibrium_distribution

    nu_0, nu_e = sp.symbols("qr_nu_0 qr_nu_e")
    c_pi_s = sp.Symbol("qr_c_pi")
    rho = equilibrium.density if equilibrium.compressible else equilibrium.background_density
    adapted_omega = sp.Symbol("sgs_omega")

    stencil = method.stencil

    pi = second_order_moment_tensor(f_neq, stencil)

    r_prime = sp.Symbol("qr_r_prime")
    q_prime = sp.Symbol("qr_q_prime")

    c_pi = qr_constant * sp.Piecewise(
        (r_prime, sp.StrictGreaterThan(r_prime, sp.Float(0))),
        (sp.Float(0), True)
    ) / q_prime

    collision_rule = collision_rule.new_with_substitutions({omega_s: adapted_omega}, substitute_on_lhs=False)

    if stencil.D == 2:
        nu_e_assignments = [Assignment(nu_e, c_pi_s)]
    elif stencil.D == 3:
        base_viscosity = sp.Symbol("qr_base_viscosity")
        nu_e_assignments = [
            Assignment(base_viscosity, sp.Float(6) * nu_0 + sp.Float(1)),
            Assignment(nu_e, (-base_viscosity + sp.sqrt(base_viscosity ** 2 + sp.Float(72) * c_pi_s / rho))
                       / sp.Float(12))
        ]
    else:
        raise ValueError("QR-model is only defined for 2- or 3-dimensional flows")

    matrix_entries = sp.Matrix(stencil.D, stencil.D, sp.symbols(f"sgs_qr_pi:{stencil.D ** 2}"))

    eqs = [Assignment(nu_0, lattice_viscosity_from_relaxation_rate(omega_s)),
           *[Assignment(matrix_entries[i], pi[i]) for i in range(stencil.D ** 2)],
           Assignment(r_prime, sp.Float(-1) ** (stencil.D + 1) * matrix_entries.det()),
           Assignment(q_prime, sp.Rational(1, 2) * (matrix_entries * matrix_entries).trace()),
           Assignment(c_pi_s, c_pi),
           *nu_e_assignments,
           Assignment(adapted_omega, relaxation_rate_from_lattice_viscosity(nu_0 + nu_e))]
    collision_rule.subexpressions += eqs
    collision_rule.topological_sort(sort_subexpressions=True, sort_main_assignments=False)

    if eddy_viscosity_field:
        collision_rule.main_assignments.append(Assignment(eddy_viscosity_field.center, nu_e))

    if omega_output_field:
        collision_rule.main_assignments.append(Assignment(omega_output_field.center, adapted_omega))

    return collision_rule
