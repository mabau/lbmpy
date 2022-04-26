from dataclasses import dataclass
import sympy as sp

from lbmpy.relaxationrates import get_shear_relaxation_rate, lattice_viscosity_from_relaxation_rate, \
    relaxation_rate_from_lattice_viscosity
from lbmpy.utils import extract_shear_relaxation_rate, frobenius_norm, second_order_moment_tensor
from pystencils import Assignment


@dataclass
class CassonsParameters:
    yield_stress: float
    """
    The yield stress controls the intensity of non-Newtonian behavior.
    """
    omega_min: float = 0.2
    """
    The minimal shear relaxation rate that is used as a lower bound
    """
    omega_max: float = 1.98
    """
    The maximal shear relaxation rate that is used as an upper bound
    """


def add_cassons_model(collision_rule, parameter: CassonsParameters, omega_output_field=None):
    r""" Adds the Cassons model to a lattice Boltzmann collision rule that can be found here :cite:`Casson`

    The only parameter of the model is the so-called yield_stress. The main idea is that no strain rate is
    observed below some stress. However, this leads to the problem that the modified relaxation rate might no longer
    lead to stable LBM simulations. Thus, an upper and lower limit for the shear relaxation rate must be given.
    All the parameters are combined in the `CassonsParameters` dataclass

    """
    yield_stress = parameter.yield_stress
    omega_min = parameter.omega_min
    omega_max = parameter.omega_max

    method = collision_rule.method
    equilibrium = method.equilibrium_distribution

    omega_s = get_shear_relaxation_rate(method)
    omega_s, found_symbolic_shear_relaxation = extract_shear_relaxation_rate(collision_rule, omega_s)

    if not found_symbolic_shear_relaxation:
        raise ValueError("For the Cassons model the shear relaxation rate has to be a symbol or it has to be "
                         "assigned to a single equation in the assignment list")

    sigma, theta, rhs, mu, tau, adapted_omega = sp.symbols("sigma theta rhs mu tau omega_new")

    rho = equilibrium.density if equilibrium.compressible else equilibrium.background_density
    f_neq = sp.Matrix(method.pre_collision_pdf_symbols) - method.get_equilibrium_terms()

    second_invariant_strain_rate_tensor = frobenius_norm(second_order_moment_tensor(f_neq, method.stencil))
    eta = lattice_viscosity_from_relaxation_rate(omega_s)

    one = sp.Rational(1, 1)

    # rhs of equation 14 in https://doi.org/10.1007/s10955-005-8415-x
    # Note that C_2 / C_4 = 3 for all configurations thus we directly insert it here
    eq14 = one / (one - theta) * (one + sp.sqrt(theta * (one + rho / eta * sp.Rational(1, 6) * (one - theta))))

    new_omega = one / tau
    omega_cond = sp.Piecewise((omega_min, new_omega < omega_min), (omega_max, new_omega > omega_max), (new_omega, True))

    eqs = [Assignment(sigma, second_invariant_strain_rate_tensor),
           Assignment(theta, yield_stress / sigma),
           Assignment(rhs, eq14),
           Assignment(mu, eta * rhs ** 2),
           Assignment(tau, one / relaxation_rate_from_lattice_viscosity(mu)),
           Assignment(adapted_omega, omega_cond)]

    collision_rule = collision_rule.new_with_substitutions({omega_s: adapted_omega}, substitute_on_lhs=False)
    collision_rule.subexpressions += eqs
    collision_rule.topological_sort(sort_subexpressions=True, sort_main_assignments=False)

    if omega_output_field:
        collision_rule.main_assignments.append(Assignment(omega_output_field.center, adapted_omega))

    return collision_rule
