from pystencils.fd.derivation import FiniteDifferenceStencilDerivation
from pystencils import Assignment, AssignmentCollection, Field

from lbmpy import pdf_initialization_assignments
from lbmpy.methods.abstractlbmethod import LbmCollisionRule
from lbmpy.utils import second_order_moment_tensor
from lbmpy.phasefield_allen_cahn.parameter_calculation import AllenCahnParameters

import sympy as sp


def chemical_potential_symbolic(phi_field, stencil, beta, kappa):
    r"""
    Get a symbolic expression for the chemical potential according to equation (5) in PhysRevE.96.053301.
    Args:
        phi_field: the phase-field on which the chemical potential is applied
        stencil: stencil to derive the finite difference for the laplacian (2nd order isotropic)
        beta: coefficient related to surface tension and interface thickness
        kappa: coefficient related to surface tension and interface thickness
    """
    lap = sp.simplify(0)
    for i in range(stencil.D):
        deriv = FiniteDifferenceStencilDerivation((i, i), stencil)
        for j in range(stencil.D):
            # assume the stencil is symmetric
            deriv.assume_symmetric(dim=j, anti_symmetric=False)

        # set weights for missing degrees of freedom in the calculation and assume the stencil is isotropic
        if stencil.Q == 9:
            res = deriv.get_stencil(isotropic=True)
            lap += res.apply(phi_field.center)
        elif stencil.Q == 15:
            deriv.set_weight((0, 0, 0), sp.Rational(-32, 27))
            res = deriv.get_stencil(isotropic=True)
            lap += res.apply(phi_field.center)
        elif stencil.Q == 19:
            res = deriv.get_stencil(isotropic=True)
            lap += res.apply(phi_field.center)
        else:
            deriv.set_weight((0, 0, 0), sp.Rational(-38, 27))
            res = deriv.get_stencil(isotropic=True)
            lap += res.apply(phi_field.center)

    # get the chemical potential
    four = sp.Rational(4, 1)
    one = sp.Rational(1, 1)
    half = sp.Rational(1, 2)
    mu = four * beta * phi_field.center * (phi_field.center - one) * (phi_field.center - half) - kappa * lap
    return mu


def isotropic_gradient_symbolic(phi_field, stencil):
    r"""
    Get a symbolic expression for the isotropic gradient of the phase-field
    Args:
        phi_field: the phase-field on which the isotropic gradient is applied
        stencil: stencil to derive the finite difference for the gradient (2nd order isotropic)
    """
    deriv = FiniteDifferenceStencilDerivation((0,), stencil)

    deriv.assume_symmetric(0, anti_symmetric=True)
    deriv.assume_symmetric(1, anti_symmetric=False)
    if stencil.D == 3:
        deriv.assume_symmetric(2, anti_symmetric=False)

    # set weights for missing degrees of freedom in the calculation and assume the stencil is isotropic
    # furthermore the stencils gets rotated to get the y and z components
    if stencil.Q == 9:
        res = deriv.get_stencil(isotropic=True)
        grad = [res.apply(phi_field.center), res.rotate_weights_and_apply(phi_field.center, (0, 1)), 0]
    elif stencil.Q == 15:
        res = deriv.get_stencil(isotropic=True)
        grad = [res.apply(phi_field.center),
                res.rotate_weights_and_apply(phi_field.center, (0, 1)),
                res.rotate_weights_and_apply(phi_field.center, (1, 2))]
    elif stencil.Q == 19:
        deriv.set_weight((0, 0, 0), sp.sympify(0))
        deriv.set_weight((1, 0, 0), sp.Rational(1, 6))

        res = deriv.get_stencil(isotropic=True)
        grad = [res.apply(phi_field.center),
                res.rotate_weights_and_apply(phi_field.center, (0, 1)),
                res.rotate_weights_and_apply(phi_field.center, (1, 2))]
    else:
        deriv.set_weight((0, 0, 0), sp.sympify(0))
        deriv.set_weight((1, 0, 0), sp.Rational(2, 9))

        res = deriv.get_stencil(isotropic=True)
        grad = [res.apply(phi_field.center),
                res.rotate_weights_and_apply(phi_field.center, (0, 1)),
                res.rotate_weights_and_apply(phi_field.center, (1, 2))]

    return grad


def normalized_isotropic_gradient_symbolic(phi_field, stencil, fd_stencil=None):
    r"""
    Get a symbolic expression for the normalized isotropic gradient of the phase-field
    Args:
        phi_field: the phase-field on which the normalized isotropic gradient is applied
        stencil: stencil of the lattice Boltzmann step
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    if fd_stencil is None:
        fd_stencil = stencil

    tmp = (sum(map(lambda x: x * x, isotropic_gradient_symbolic(phi_field, fd_stencil))) + sp.Float(1e-32)) ** 0.5

    result = [x / tmp for x in isotropic_gradient_symbolic(phi_field, fd_stencil)]
    return result


def pressure_force(phi_field, lb_method, stencil, density_heavy, density_light, fd_stencil=None):
    r"""
    Get a symbolic expression for the pressure force
    Args:
        phi_field: phase-field
        lb_method: lattice boltzmann method used for hydrodynamics
        stencil: stencil of the lattice Boltzmann step
        density_heavy: density of the heavier fluid
        density_light: density of the lighter fluid
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    if fd_stencil is None:
        fd_stencil = stencil

    cqc = lb_method.conserved_quantity_computation
    rho = cqc.density_deviation_symbol

    iso_grad = isotropic_gradient_symbolic(phi_field, fd_stencil)
    result = list(map(lambda x: sp.Rational(-1, 3) * rho * (density_heavy - density_light) * x, iso_grad))
    return result


def viscous_force(lb_velocity_field, phi_field, lb_method, tau, density_heavy, density_light, fd_stencil=None):
    r"""
    Get a symbolic expression for the viscous force
    Args:
        lb_velocity_field: hydrodynamic distribution function
        phi_field: phase-field
        lb_method: lattice boltzmann method used for hydrodynamics
        tau: relaxation time of the hydrodynamic lattice boltzmann step
        density_heavy: density of the heavier fluid
        density_light: density of the lighter fluid
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    stencil = lb_method.stencil

    if fd_stencil is None:
        fd_stencil = stencil

    iso_grad = sp.Matrix(isotropic_gradient_symbolic(phi_field, fd_stencil)[:stencil.D])

    f_neq = lb_velocity_field.center_vector - lb_method.get_equilibrium_terms()
    stress_tensor = second_order_moment_tensor(f_neq, lb_method.stencil)
    normal_stress_tensor = stress_tensor * iso_grad

    density_difference = density_heavy - density_light

    # Calculate Viscous Force MRT
    half = sp.Rational(1, 2)
    fmx = (half - tau) * normal_stress_tensor[0] * density_difference
    fmy = (half - tau) * normal_stress_tensor[1] * density_difference
    fmz = (half - tau) * normal_stress_tensor[2] * density_difference if stencil.D == 3 else 0

    return [fmx, fmy, fmz]


def surface_tension_force(phi_field, stencil, beta, kappa, fd_stencil=None):
    r"""
    Get a symbolic expression for the surface tension force
    Args:
        phi_field: the phase-field on which the chemical potential is applied
        stencil: stencil of the lattice Boltzmann step
        beta: coefficient related to surface tension and interface thickness
        kappa: coefficient related to surface tension and interface thickness
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    if fd_stencil is None:
        fd_stencil = stencil

    chemical_potential = chemical_potential_symbolic(phi_field, fd_stencil, beta, kappa)
    iso_grad = isotropic_gradient_symbolic(phi_field, fd_stencil)
    return [chemical_potential * x for x in iso_grad]


def hydrodynamic_force(lb_velocity_field, phi_field, lb_method, parameters: AllenCahnParameters,
                       body_force, fd_stencil=None):
    r"""
    Get a symbolic expression for the hydrodynamic force
    Args:
        lb_velocity_field: hydrodynamic distribution function
        phi_field: phase-field
        lb_method: Lattice boltzmann method used for hydrodynamics
        parameters: AllenCahnParameters
        body_force: force acting on the fluids. Usually the gravity
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    stencil = lb_method.stencil

    if fd_stencil is None:
        fd_stencil = stencil

    density_heavy = parameters.symbolic_density_heavy
    density_light = parameters.symbolic_density_light
    tau_L = parameters.symbolic_tau_light
    tau_H = parameters.symbolic_tau_heavy
    tau = sp.Rational(1, 2) + tau_L + phi_field.center * (tau_H - tau_L)
    beta = parameters.beta
    kappa = parameters.kappa

    fp = pressure_force(phi_field, lb_method, stencil, density_heavy, density_light, fd_stencil)
    fm = viscous_force(lb_velocity_field, phi_field, lb_method, tau, density_heavy, density_light, fd_stencil)
    fs = surface_tension_force(phi_field, stencil, beta, kappa, fd_stencil)

    result = []
    for i in range(stencil.D):
        result.append(fs[i] + fp[i] + fm[i] + body_force[i])

    return result


def interface_tracking_force(phi_field, stencil, parameters: AllenCahnParameters, fd_stencil=None,
                             phi_heavy=1, phi_light=0):
    r"""
    Get a symbolic expression for the hydrodynamic force
    Args:
        phi_field: phase-field
        stencil: stencil of the phase-field distribution lattice Boltzmann step
        parameters: AllenCahnParameters
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
        phi_heavy: phase field value in the bulk of the heavy fluid
        phi_light: phase field value in the bulk of the light fluid

    """
    if fd_stencil is None:
        fd_stencil = stencil

    phi_zero = sp.Rational(1, 2) * (phi_light + phi_heavy)

    normal_fd = normalized_isotropic_gradient_symbolic(phi_field, stencil, fd_stencil)
    result = []
    interface_thickness = parameters.symbolic_interface_thickness
    for i in range(stencil.D):
        fraction = (sp.Rational(1, 1) - sp.Rational(4, 1) * (phi_field.center - phi_zero) ** 2) / interface_thickness
        result.append(sp.Rational(1, 3) * fraction * normal_fd[i])

    return result


def hydrodynamic_force_assignments(lb_velocity_field, velocity_field, phi_field, lb_method,
                                   parameters: AllenCahnParameters,
                                   body_force, fd_stencil=None, sub_iterations=2):

    r"""
    Get a symbolic expression for the hydrodynamic force
    Args:
        lb_velocity_field: hydrodynamic distribution function
        velocity_field: velocity
        phi_field: phase-field
        lb_method: Lattice boltzmann method used for hydrodynamics
        parameters: AllenCahnParameters
        body_force: force acting on the fluids. Usually the gravity
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
        sub_iterations: number of sub iterations for the hydrodynamic force
    """

    rho_L = parameters.symbolic_density_light
    rho_H = parameters.symbolic_density_heavy
    density = rho_L + phi_field.center * (rho_H - rho_L)

    stencil = lb_method.stencil
    # method has to have a force model
    symbolic_force = lb_method.force_model.symbolic_force_vector

    force = hydrodynamic_force(lb_velocity_field, phi_field, lb_method, parameters, body_force, fd_stencil=fd_stencil)

    cqc = lb_method.conserved_quantity_computation

    u_symp = cqc.velocity_symbols
    cqe = cqc.equilibrium_input_equations_from_pdfs(lb_velocity_field.center_vector)
    cqe = cqe.new_without_subexpressions()

    cqe_velocity = [eq.rhs for eq in cqe.main_assignments[1:]]
    index = 0
    aleph = sp.symbols(f"aleph_:{stencil.D * sub_iterations}")

    force_Assignments = []

    for i in range(stencil.D):
        force_Assignments.append(Assignment(aleph[i], velocity_field.center_vector[i]))
        index += 1

    for k in range(sub_iterations - 1):
        subs_dict = dict(zip(u_symp, aleph[k * stencil.D:index]))
        for i in range(stencil.D):
            new_force = force[i].subs(subs_dict) / density
            force_Assignments.append(Assignment(aleph[index], cqe_velocity[i].subs({symbolic_force[i]: new_force})))
            index += 1

    subs_dict = dict(zip(u_symp, aleph[index - stencil.D:index]))

    for i in range(stencil.D):
        force_Assignments.append(Assignment(symbolic_force[i], force[i].subs(subs_dict)))

    return force_Assignments


def add_interface_tracking_force(update_rule: LbmCollisionRule, force):
    r"""
     Adds the interface tracking force to a lattice Boltzmann update rule
     Args:
         update_rule: lattice Boltzmann update rule
         force: interface tracking force
     """
    method = update_rule.method
    symbolic_force = method.force_model.symbolic_force_vector

    for i in range(method.stencil.D):
        update_rule.subexpressions += [Assignment(symbolic_force[i], force[i])]

    update_rule.topological_sort(sort_subexpressions=True, sort_main_assignments=False)

    return update_rule


def add_hydrodynamic_force(update_rule: LbmCollisionRule, force, phi_field,
                           hydro_pdfs, parameters: AllenCahnParameters):
    r"""
     Adds the interface tracking force to a lattice Boltzmann update rule
     Args:
         update_rule: lattice Boltzmann update rule
         force: interface tracking force
         phi_field: phase-field
         hydro_pdfs: source field of the hydrodynamic PDFs
         parameters: AllenCahnParameters
     """
    rho_L = parameters.symbolic_density_light
    rho_H = parameters.symbolic_density_heavy
    density = rho_L + phi_field.center * (rho_H - rho_L)

    method = update_rule.method
    symbolic_force = method.force_model.symbolic_force_vector
    cqc = method.conserved_quantity_computation
    rho = cqc.density_deviation_symbol

    force_subs = {f: f / density for f in symbolic_force}

    update_rule = update_rule.subs(force_subs)

    update_rule.subexpressions += [Assignment(rho, sum(hydro_pdfs.center_vector))]
    update_rule.subexpressions += force
    update_rule.topological_sort(sort_subexpressions=True, sort_main_assignments=False)

    return update_rule


def initializer_kernel_phase_field_lb(lb_method, phi, velocity, ac_pdfs, parameters: AllenCahnParameters,
                                      fd_stencil=None):
    r"""
    Returns an assignment list for initializing the phase-field distribution functions
    Args:
        lb_method: lattice Boltzmann method of the phase-field lattice Boltzmann step
        phi: order parameter of the Allen-Cahn LB step (phase field)
        velocity: initial velocity
        ac_pdfs: source field of the Allen-Cahn PDFs
        parameters: AllenCahnParameters
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """

    h_updates = pdf_initialization_assignments(lb_method, phi, velocity, ac_pdfs)
    force_h = interface_tracking_force(phi, lb_method.stencil, parameters,
                                       fd_stencil=fd_stencil)

    cqc = lb_method.conserved_quantity_computation

    rho = cqc.density_symbol
    u_symp = cqc.velocity_symbols
    symbolic_force = lb_method.force_model.symbolic_force_vector

    macro_quantities = []

    if isinstance(velocity, Field):
        velocity = velocity.center_vector

    if isinstance(phi, Field):
        phi = phi.center

    for i in range(lb_method.stencil.D):
        macro_quantities.append(Assignment(symbolic_force[i], force_h[i]))

    for i in range(lb_method.stencil.D):
        macro_quantities.append(Assignment(u_symp[i],
                                           velocity[i] - sp.Rational(1, 2) * symbolic_force[i]))

    h_updates = AssignmentCollection(main_assignments=h_updates.main_assignments, subexpressions=macro_quantities)
    h_updates = h_updates.new_with_substitutions({rho: phi})

    return h_updates


def initializer_kernel_hydro_lb(lb_method, pressure, velocity, hydro_pdfs):
    r"""
    Returns an assignment list for initializing the velocity distribution functions
    Args:
        lb_method: lattice Boltzmann method of the hydrodynamic lattice Boltzmann step
        pressure: order parameter of the hydrodynamic LB step (pressure)
        velocity: initial velocity
        hydro_pdfs: source field of the hydrodynamic PDFs
    """
    symbolic_force = lb_method.force_model.symbolic_force_vector
    force_subs = {f: 0 for f in symbolic_force}

    g_updates = pdf_initialization_assignments(lb_method, pressure, velocity, hydro_pdfs)
    g_updates = g_updates.new_with_substitutions(force_subs)

    return g_updates
