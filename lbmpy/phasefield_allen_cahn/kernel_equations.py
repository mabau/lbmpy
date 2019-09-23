from pystencils.fd.derivation import FiniteDifferenceStencilDerivation
from lbmpy.maxwellian_equilibrium import get_weights
from pystencils import Assignment, AssignmentCollection
from pystencils.simp import sympy_cse_on_assignment_list

import sympy as sp
import numpy as np


def chemical_potential_symbolic(phi_field, stencil, beta, kappa):
    r"""
    Get a symbolic expression for the chemical potential according to equation (5) in PhysRevE.96.053301.
    Args:
        phi_field: the phase-field on which the chemical potential is applied
        stencil: stencil of the lattice Boltzmann step
        beta: coefficient related to surface tension and interface thickness
        kappa: coefficient related to surface tension and interface thickness
    """
    dimensions = len(stencil[0])

    deriv = FiniteDifferenceStencilDerivation((0, 0), stencil)
    # assume the stencil is symmetric
    deriv.assume_symmetric(0)
    deriv.assume_symmetric(1)
    if dimensions == 3:
        deriv.assume_symmetric(2)

    # set weights for missing degrees of freedom in the calculation
    if len(stencil) == 9:
        deriv.set_weight((1, 1), sp.Rational(1, 6))
        deriv.set_weight((0, -1), sp.Rational(2, 3))
        deriv.set_weight((0, 0), sp.Rational(-10, 3))
    if len(stencil) == 27:
        deriv.set_weight((1, 1, 1), sp.Rational(1, 48))

    # assume the stencil is isotropic
    res = deriv.get_stencil(isotropic=True)

    # get the chemical potential
    mu = 4.0 * beta * phi_field.center * (phi_field.center - 1.0) * (phi_field.center - 0.5) - \
        kappa * res.apply(phi_field.center)
    return mu


def isotropic_gradient_symbolic(phi_field, stencil):
    r"""
    Get a symbolic expression for the isotropic gradient of the phase-field
    Args:
        phi_field: the phase-field on which the isotropic gradient is applied
        stencil: stencil of the lattice Boltzmann step
    """
    dimensions = len(stencil[0])
    deriv = FiniteDifferenceStencilDerivation((0, ), stencil)

    deriv.assume_symmetric(0, anti_symmetric=True)
    deriv.assume_symmetric(1, anti_symmetric=False)
    if dimensions == 3:
        deriv.assume_symmetric(2, anti_symmetric=False)

    if len(stencil) == 19:
        deriv.set_weight((0, 0, 0), sp.Integer(0))
        deriv.set_weight((1, 0, 0), sp.Rational(1, 6))
        deriv.set_weight((1, 1, 0), sp.Rational(1, 12))
    elif len(stencil) == 27:
        deriv.set_weight((0, 0, 0), sp.Integer(0))
        deriv.set_weight((1, 1, 1), sp.Rational(1, 3360))

    res = deriv.get_stencil(isotropic=True)
    if dimensions == 2:
        grad = [res.apply(phi_field.center), res.rotate_weights_and_apply(phi_field.center, (0, 1)), 0]
    else:
        grad = [res.apply(phi_field.center),
                res.rotate_weights_and_apply(phi_field.center, (1, 0)),
                res.rotate_weights_and_apply(phi_field.center, (2, 1))]

    return grad


def normalized_isotropic_gradient_symbolic(phi_field, stencil):
    r"""
    Get a symbolic expression for the normalized isotropic gradient of the phase-field
    Args:
        phi_field: the phase-field on which the normalized isotropic gradient is applied
        stencil: stencil of the lattice Boltzmann step
    """
    isotropic_gradient = isotropic_gradient_symbolic(phi_field, stencil)
    tmp = (sum(map(lambda x: x * x, isotropic_gradient_symbolic(phi_field, stencil))) + 1.e-12) ** 0.5

    result = [x / tmp for x in isotropic_gradient]
    return result


def pressure_force(phi_field, stencil, density_heavy, density_light):
    r"""
    Get a symbolic expression for the pressure force
    Args:
        phi_field: phase-field
        stencil: stencil of the lattice Boltzmann step
        density_heavy: density of the heavier fluid
        density_light: density of the lighter fluid
    """
    isotropic_gradient = isotropic_gradient_symbolic(phi_field, stencil)
    result = list(map(lambda x: -sp.symbols("rho") * ((density_heavy - density_light) / 3) * x, isotropic_gradient))
    return result


def viscous_force(lb_velocity_field, phi_field, mrt_method, tau, density_heavy, density_light):
    r"""
    Get a symbolic expression for the viscous force
    Args:
        lb_velocity_field: hydrodynamic distribution function
        phi_field: phase-field
        mrt_method: mrt lattice boltzmann method used for hydrodynamics
        tau: relaxation time of the hydrodynamic lattice boltzmann step
        density_heavy: density of the heavier fluid
        density_light: density of the lighter fluid
    """
    stencil = mrt_method.stencil
    dimensions = len(stencil[0])

    isotropic_gradient = isotropic_gradient_symbolic(phi_field, stencil)

    weights = get_weights(stencil, c_s_sq=sp.Rational(1, 3))

    op = sp.symbols("rho")

    gamma = mrt_method.get_equilibrium_terms()
    gamma = gamma.subs({sp.symbols("rho"): 1})

    moment_matrix = mrt_method.moment_matrix
    relaxation_rates = sp.Matrix(np.diag(mrt_method.relaxation_rates))
    mrt_collision_op = moment_matrix.inv() * relaxation_rates * moment_matrix

    # get the non equilibrium
    non_equilibrium = [lb_velocity_field.center(i) - (weights[i] * op + gamma[i] - weights[i])
                       for i, _ in enumerate(stencil)]
    non_equilibrium = np.dot(mrt_collision_op, non_equilibrium)

    stress_tensor = [0] * 6
    # Calculate Stress Tensor MRT
    for i, d in enumerate(stencil):
        stress_tensor[0] += non_equilibrium[i] * (d[0] * d[0])
        stress_tensor[1] += non_equilibrium[i] * (d[1] * d[1])

        if dimensions == 3:
            stress_tensor[2] += non_equilibrium[i] * (d[2] * d[2])
            stress_tensor[3] += non_equilibrium[i] * (d[1] * d[2])
            stress_tensor[4] += non_equilibrium[i] * (d[0] * d[2])

        stress_tensor[5] += non_equilibrium[i] * (d[0] * d[1])

    density_difference = density_heavy - density_light

    # Calculate Viscous Force MRT
    fmx = (0.5 - tau) * (stress_tensor[0] * isotropic_gradient[0]
                         + stress_tensor[5] * isotropic_gradient[1]
                         + stress_tensor[4] * isotropic_gradient[2]) * density_difference

    fmy = (0.5 - tau) * (stress_tensor[5] * isotropic_gradient[0]
                         + stress_tensor[1] * isotropic_gradient[1]
                         + stress_tensor[3] * isotropic_gradient[2]) * density_difference

    fmz = (0.5 - tau) * (stress_tensor[4] * isotropic_gradient[0]
                         + stress_tensor[3] * isotropic_gradient[1]
                         + stress_tensor[2] * isotropic_gradient[2]) * density_difference

    return [fmx, fmy, fmz]


def surface_tension_force(phi_field, stencil, beta, kappa):
    r"""
    Get a symbolic expression for the surface tension force
    Args:
        phi_field: the phase-field on which the chemical potential is applied
        stencil: stencil of the lattice Boltzmann step
        beta: coefficient related to surface tension and interface thickness
        kappa: coefficient related to surface tension and interface thickness
    """
    chemical_potential = chemical_potential_symbolic(phi_field, stencil, beta, kappa)
    isotropic_gradient = isotropic_gradient_symbolic(phi_field, stencil)
    return [chemical_potential * x for x in isotropic_gradient]


def hydrodynamic_force(lb_velocity_field, phi_field, mrt_method, tau,
                       density_heavy, density_light, kappa, beta, body_force):
    r"""
    Get a symbolic expression for the hydrodynamic force
    Args:
        lb_velocity_field: hydrodynamic distribution function
        phi_field: phase-field
        mrt_method: mrt lattice boltzmann method used for hydrodynamics
        tau: relaxation time of the hydrodynamic lattice boltzmann step
        density_heavy: density of the heavier fluid
        density_light: density of the lighter fluid
        beta: coefficient related to surface tension and interface thickness
        kappa: coefficient related to surface tension and interface thickness
        body_force: force acting on the fluids. Usually the gravity
    """
    stencil = mrt_method.stencil
    dimensions = len(stencil[0])
    fp = pressure_force(phi_field, stencil, density_heavy, density_light)
    fm = viscous_force(lb_velocity_field, phi_field, mrt_method, tau, density_heavy, density_light)
    fs = surface_tension_force(phi_field, stencil, beta, kappa)

    result = []
    for i in range(dimensions):
        result.append(fs[i] + fp[i] + fm[i] + body_force[i])

    return result


def interface_tracking_force(phi_field, stencil, interface_thickness):
    r"""
    Get a symbolic expression for the hydrodynamic force
    Args:
        phi_field: phase-field
        stencil: stencil of the phase-field distribution lattice Boltzmann step
        interface_thickness: interface thickness
    """
    dimensions = len(stencil[0])
    normal_fd = normalized_isotropic_gradient_symbolic(phi_field, stencil)
    result = []
    for i in range(dimensions):
        result.append(((1.0 - 4.0 * (phi_field.center - 0.5) ** 2.0) / interface_thickness) * normal_fd[i])

    return result


def get_assignment_list_stream_hydro(lb_vel_field, lb_vel_field_tmp, mrt_method, force_g, velocity, rho):
    r"""
    Returns an assignment list of the streaming step for the hydrodynamic lattice Boltzmann step. In the assignment list
    also the update for the velocity is integrated
    Args:
        lb_vel_field: source field of velocity distribution function
        lb_vel_field_tmp: destination field of the velocity distribution function
        mrt_method: lattice Boltzmann method of the hydrodynamic lattice Boltzmann step
        force_g: hydrodynamic force
        velocity: velocity field
        rho: interpolated density of the two fluids
    """

    stencil = mrt_method.stencil
    dimensions = len(stencil[0])

    velocity_symbol_list = [lb_vel_field.center(i) for i, _ in enumerate(stencil)]
    velocity_tmp_symbol_list = [lb_vel_field_tmp.center(i) for i, _ in enumerate(stencil)]

    g_subs_dic = dict(zip(velocity_symbol_list, velocity_tmp_symbol_list))
    u_symp = sp.symbols("u_:{}".format(dimensions))

    a = [0] * dimensions
    for i, direction in enumerate(stencil):
        for j in range(dimensions):
            a[j] += velocity_tmp_symbol_list[i] * direction[j]

    pull_g = list()
    inv_dir = 0
    for i, direction in enumerate(stencil):
        inv_direction = tuple(-e for e in direction)
        pull_g.append(Assignment(lb_vel_field_tmp(i), lb_vel_field[inv_direction](i)))
        inv_dir += lb_vel_field[inv_direction](i)

    for i in range(dimensions):
        pull_g.append(Assignment(velocity.center_vector[i], a[i] + 0.5 * force_g[i].subs(g_subs_dic) / rho))

    subexpression = [Assignment(sp.symbols("rho"), inv_dir)]

    for i in range(dimensions):
        subexpression.append(Assignment(u_symp[i], velocity.center_vector[i]))

    ac_g = AssignmentCollection(main_assignments=pull_g, subexpressions=subexpression)

    ac_g.main_assignments = sympy_cse_on_assignment_list(ac_g.main_assignments)

    return ac_g


def initializer_kernel_phase_field_lb(phi_field_distributions, phi_field, velocity, mrt_method, interface_thickness):
    r"""
    Returns an assignment list for initializing the phase-field distribution functions
    Args:
        phi_field_distributions: source field of phase-field distribution function
        phi_field: phase-field
        velocity: velocity field
        mrt_method: lattice Boltzmann method of the phase-field lattice Boltzmann step
        interface_thickness: interface thickness
    """
    stencil = mrt_method.stencil
    dimensions = len(stencil[0])
    weights = get_weights(stencil, c_s_sq=sp.Rational(1, 3))
    u_symp = sp.symbols("u_:{}".format(dimensions))

    normal_fd = normalized_isotropic_gradient_symbolic(phi_field, stencil)

    gamma = mrt_method.get_equilibrium_terms()
    gamma = gamma.subs({sp.symbols("rho"): 1})
    gamma_init = gamma.subs({x: y for x, y in zip(u_symp, velocity.center_vector)})
    # create the kernels for the initialization of the g and h field
    h_updates = list()

    def scalar_product(a, b):
        return sum(a_i * b_i for a_i, b_i in zip(a, b))

    f = []
    for i, d in enumerate(stencil):
        f.append(weights[i] * ((1.0 - 4.0 * (phi_field.center - 0.5) ** 2.0) / interface_thickness)
                 * scalar_product(d, normal_fd[0:dimensions]))

    for i, _ in enumerate(stencil):
        h_updates.append(Assignment(phi_field_distributions.center(i), phi_field.center * gamma_init[i] - 0.5 * f[i]))

    return h_updates


def initializer_kernel_hydro_lb(velocity_distributions, velocity, mrt_method):
    r"""
    Returns an assignment list for initializing the velocity distribution functions
    Args:
        velocity_distributions: source field of velocity distribution function
        velocity: velocity field
        mrt_method: lattice Boltzmann method of the hydrodynamic lattice Boltzmann step
    """
    stencil = mrt_method.stencil
    dimensions = len(stencil[0])
    weights = get_weights(stencil, c_s_sq=sp.Rational(1, 3))
    u_symp = sp.symbols("u_:{}".format(dimensions))

    gamma = mrt_method.get_equilibrium_terms()
    gamma = gamma.subs({sp.symbols("rho"): 1})
    gamma_init = gamma.subs({x: y for x, y in zip(u_symp, velocity.center_vector)})

    g_updates = list()
    for i, _ in enumerate(stencil):
        g_updates.append(Assignment(velocity_distributions.center(i), gamma_init[i] - weights[i]))

    return g_updates
