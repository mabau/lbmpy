from pystencils.fd.derivation import FiniteDifferenceStencilDerivation
from pystencils import Assignment

from lbmpy.methods.momentbased.centralmomentbasedmethod import CentralMomentBasedLbMethod
from lbmpy.moments import get_order
from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.fieldaccess import StreamPullTwoFieldsAccessor, StreamPushTwoFieldsAccessor, CollideOnlyInplaceAccessor
from lbmpy.methods.abstractlbmethod import LbmCollisionRule

from lbmpy.phasefield_allen_cahn.phasefield_simplifications import create_phasefield_simplification_strategy
from lbmpy.phasefield_allen_cahn.force_model import CentralMomentMultiphaseForceModel

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
    mu = 4.0 * beta * phi_field.center * (phi_field.center - 1.0) * (phi_field.center - 0.5) - kappa * lap
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

    tmp = (sum(map(lambda x: x * x, isotropic_gradient_symbolic(phi_field, fd_stencil))) + 1.e-32) ** 0.5

    result = [x / tmp for x in isotropic_gradient_symbolic(phi_field, fd_stencil)]
    return result


def pressure_force(phi_field, stencil, density_heavy, density_light, fd_stencil=None):
    r"""
    Get a symbolic expression for the pressure force
    Args:
        phi_field: phase-field
        stencil: stencil of the lattice Boltzmann step
        density_heavy: density of the heavier fluid
        density_light: density of the lighter fluid
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    if fd_stencil is None:
        fd_stencil = stencil

    iso_grad = isotropic_gradient_symbolic(phi_field, fd_stencil)
    result = list(map(lambda x: sp.Rational(-1, 3) * sp.symbols("rho") * (density_heavy - density_light) * x, iso_grad))
    return result


def viscous_force(lb_velocity_field, phi_field, mrt_method, tau, density_heavy, density_light, fd_stencil=None):
    r"""
    Get a symbolic expression for the viscous force
    Args:
        lb_velocity_field: hydrodynamic distribution function
        phi_field: phase-field
        mrt_method: mrt lattice boltzmann method used for hydrodynamics
        tau: relaxation time of the hydrodynamic lattice boltzmann step
        density_heavy: density of the heavier fluid
        density_light: density of the lighter fluid
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    stencil = mrt_method.stencil
    dimensions = stencil.D

    if fd_stencil is None:
        fd_stencil = stencil

    iso_grad = isotropic_gradient_symbolic(phi_field, fd_stencil)

    non_equilibrium = lb_velocity_field.center_vector - mrt_method.get_equilibrium_terms()

    stress_tensor = [0] * 6
    # Calculate Stress Tensor MRT
    for i, d in enumerate(stencil):
        stress_tensor[0] = sp.Add(stress_tensor[0], non_equilibrium[i] * (d[0] * d[0]))
        stress_tensor[1] = sp.Add(stress_tensor[1], non_equilibrium[i] * (d[1] * d[1]))

        if dimensions == 3:
            stress_tensor[2] = sp.Add(stress_tensor[2], non_equilibrium[i] * (d[2] * d[2]))
            stress_tensor[3] = sp.Add(stress_tensor[3], non_equilibrium[i] * (d[1] * d[2]))
            stress_tensor[4] = sp.Add(stress_tensor[4], non_equilibrium[i] * (d[0] * d[2]))

        stress_tensor[5] = sp.Add(stress_tensor[5], non_equilibrium[i] * (d[0] * d[1]))

    density_difference = density_heavy - density_light

    # Calculate Viscous Force MRT
    fmx = (0.5 - tau) * (stress_tensor[0] * iso_grad[0]
                         + stress_tensor[5] * iso_grad[1]
                         + stress_tensor[4] * iso_grad[2]) * density_difference

    fmy = (0.5 - tau) * (stress_tensor[5] * iso_grad[0]
                         + stress_tensor[1] * iso_grad[1]
                         + stress_tensor[3] * iso_grad[2]) * density_difference

    fmz = (0.5 - tau) * (stress_tensor[4] * iso_grad[0]
                         + stress_tensor[3] * iso_grad[1]
                         + stress_tensor[2] * iso_grad[2]) * density_difference

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


def hydrodynamic_force(lb_velocity_field, phi_field, lb_method, tau,
                       density_heavy, density_light, kappa, beta, body_force, fd_stencil=None):
    r"""
    Get a symbolic expression for the hydrodynamic force
    Args:
        lb_velocity_field: hydrodynamic distribution function
        phi_field: phase-field
        lb_method: Lattice boltzmann method used for hydrodynamics
        tau: relaxation time of the hydrodynamic lattice boltzmann step
        density_heavy: density of the heavier fluid
        density_light: density of the lighter fluid
        beta: coefficient related to surface tension and interface thickness
        kappa: coefficient related to surface tension and interface thickness
        body_force: force acting on the fluids. Usually the gravity
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    stencil = lb_method.stencil

    if fd_stencil is None:
        fd_stencil = stencil

    fp = pressure_force(phi_field, stencil, density_heavy, density_light, fd_stencil)
    fm = viscous_force(lb_velocity_field, phi_field, lb_method, tau, density_heavy, density_light, fd_stencil)
    fs = surface_tension_force(phi_field, stencil, beta, kappa, fd_stencil)

    result = []
    for i in range(stencil.D):
        result.append(fs[i] + fp[i] + fm[i] + body_force[i])

    return result


def interface_tracking_force(phi_field, stencil, interface_thickness, fd_stencil=None):
    r"""
    Get a symbolic expression for the hydrodynamic force
    Args:
        phi_field: phase-field
        stencil: stencil of the phase-field distribution lattice Boltzmann step
        interface_thickness: interface thickness
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    if fd_stencil is None:
        fd_stencil = stencil

    normal_fd = normalized_isotropic_gradient_symbolic(phi_field, stencil, fd_stencil)
    result = []
    for i in range(stencil.D):
        result.append(((1.0 - 4.0 * (phi_field.center - 0.5) ** 2) / interface_thickness) * normal_fd[i])

    return result


def get_update_rules_velocity(src_field, u_in, lb_method, force_model, density, sub_iterations=2):
    r"""
     Get assignments to update the velocity with a force shift
     Args:
         src_field: the source field of the hydrodynamic distribution function
         u_in: velocity field
         lb_method: mrt lattice boltzmann method used for hydrodynamics
         force_model: one of the phase_field force models which are applied in the collision space
         density: the interpolated density of the simulation
         sub_iterations: number of updates of the velocity field
     """
    stencil = lb_method.stencil

    rho = lb_method.conserved_quantity_computation.zeroth_order_moment_symbol
    u_symp = lb_method.conserved_quantity_computation.first_order_moment_symbols

    force = force_model._force
    force_symp = force_model.force_symp

    moment_matrix = lb_method.moment_matrix

    moments = lb_method.moments
    indices = list()
    for i in range(len(moments)):
        if get_order(moments[i]) == 1:
            indices.append(i)

    m0 = moment_matrix * sp.Matrix(src_field.center_vector)

    update_u = list()
    update_u.append(Assignment(rho, m0[0]))

    index = 0
    aleph = sp.symbols(f"aleph_:{stencil.D * sub_iterations}")

    for i in range(stencil.D):
        update_u.append(Assignment(aleph[i], u_in.center_vector[i]))
        index += 1

    for k in range(sub_iterations - 1):
        subs_dict = dict(zip(u_symp, aleph[k * stencil.D:index]))
        for i in range(stencil.D):
            update_u.append(Assignment(aleph[index], m0[indices[i]] + force[i].subs(subs_dict) / density / 2))
            index += 1

    subs_dict = dict(zip(u_symp, aleph[index - stencil.D:index]))

    for i in range(stencil.D):
        update_u.append(Assignment(force_symp[i], force[i].subs(subs_dict)))

    for i in range(stencil.D):
        update_u.append(Assignment(u_symp[i], m0[indices[i]] + force_symp[i] / density / 2))

    return update_u


def get_collision_assignments_hydro(lb_method, density, velocity_input, force_model, sub_iterations, symbolic_fields,
                                    kernel_type):
    r"""
     Get collision assignments for the hydrodynamic lattice Boltzmann step. Here the force gets applied in the moment
     space. Afterwards the transformation back to the pdf space happens.
     Args:
         lb_method: moment based lattice Boltzmann method
         density: the interpolated density of the simulation
         velocity_input: velocity field for the hydrodynamic and Allen-Chan LB step
         force_model: one of the phase_field force models which are applied in the collision space
         sub_iterations: number of updates of the velocity field
         symbolic_fields: PDF fields for source and destination
         kernel_type: collide_stream_push or collide_only
     """

    if isinstance(lb_method, CentralMomentBasedLbMethod) and not \
            isinstance(force_model, CentralMomentMultiphaseForceModel):
        raise ValueError("For central moment lb methods a central moment force model needs the be applied")

    stencil = lb_method.stencil

    rho = lb_method.conserved_quantity_computation.zeroth_order_moment_symbol

    src_field = symbolic_fields['symbolic_field']
    dst_field = symbolic_fields['symbolic_temporary_field']

    if kernel_type == 'collide_stream_push':
        accessor = StreamPushTwoFieldsAccessor()
    else:
        accessor = CollideOnlyInplaceAccessor()

    u_symp = lb_method.conserved_quantity_computation.first_order_moment_symbols

    moment_matrix = lb_method.moment_matrix
    rel = sp.diag(*lb_method.relaxation_rates)
    eq = sp.Matrix(lb_method.moment_equilibrium_values)

    force_terms = force_model(lb_method)
    eq = eq - sp.Rational(1, 2) * force_terms

    pre = sp.symbols(f"pre_:{stencil.Q}")
    post = sp.symbols(f"post_:{stencil.Q}")

    to_moment_space = moment_matrix * sp.Matrix(accessor.read(src_field, stencil))
    to_moment_space[0] = rho

    main_assignments = list()
    subexpressions = get_update_rules_velocity(src_field, velocity_input, lb_method, force_model,
                                               density, sub_iterations=sub_iterations)

    for i in range(0, stencil.Q):
        subexpressions.append(Assignment(pre[i], to_moment_space[i]))

    if isinstance(lb_method, CentralMomentBasedLbMethod):
        n0 = lb_method.shift_matrix * sp.Matrix(pre)
        to_central = sp.Matrix(sp.symbols(f"kappa_:{stencil.Q}"))
        for i in range(0, stencil.Q):
            subexpressions.append(Assignment(to_central[i], n0[i]))
        pre = to_central

    collision = sp.Matrix(pre) - rel * (sp.Matrix(pre) - eq) + force_terms

    for i in range(0, stencil.Q):
        subexpressions.append(Assignment(post[i], collision[i]))

    if isinstance(lb_method, CentralMomentBasedLbMethod):
        n0_back = lb_method.shift_matrix.inv() * sp.Matrix(post)
        from_central = sp.Matrix(sp.symbols(f"kappa_post:{stencil.Q}"))
        for i in range(0, stencil.Q):
            subexpressions.append(Assignment(from_central[i], n0_back[i]))
        post = from_central

    to_pdf_space = moment_matrix.inv() * sp.Matrix(post)

    for i in range(0, stencil.Q):
        main_assignments.append(Assignment(accessor.write(dst_field, stencil)[i], to_pdf_space[i]))

    for i in range(stencil.D):
        main_assignments.append(Assignment(velocity_input.center_vector[i], u_symp[i]))

    collision_rule = LbmCollisionRule(lb_method, main_assignments, subexpressions)

    simplification = create_phasefield_simplification_strategy(lb_method)
    collision_rule = simplification(collision_rule)

    return collision_rule


def get_collision_assignments_phase(lb_method, velocity_input, output, force_model, symbolic_fields, kernel_type):
    r"""
     Get collision assignments for the phasefield lattice Boltzmann step. Here the force gets applied in the moment
     space. Afterwards the transformation back to the pdf space happens.
     Args:
         lb_method: moment based lattice Boltzmann method
         velocity_input: velocity field for the hydrodynamic and Allen-Chan LB step
         output: output field for the phasefield (calles density as for normal LB update rules)
         force_model: one of the phase_field force models which are applied in the collision space
         symbolic_fields: PDF fields for source and destination
         kernel_type: stream_pull_collide or collide_only
     """

    stencil = lb_method.stencil

    src_field = symbolic_fields['symbolic_field']
    dst_field = symbolic_fields['symbolic_temporary_field']
    output_phase_field = output['density']

    if kernel_type == 'stream_pull_collide':
        accessor = StreamPullTwoFieldsAccessor()
    else:
        accessor = CollideOnlyInplaceAccessor()

    subexpressions = list()
    main_assignments = list()

    rho = lb_method.conserved_quantity_computation.zeroth_order_moment_symbol
    u_symp = lb_method.conserved_quantity_computation.first_order_moment_symbols

    moment_matrix = lb_method.moment_matrix
    rel = sp.diag(*lb_method.relaxation_rates)
    eq = sp.Matrix(lb_method.moment_equilibrium_values)

    force_terms = force_model(lb_method)
    eq = eq - sp.Rational(1, 2) * force_terms

    pre = sp.symbols(f"pre_:{stencil.Q}")
    post = sp.symbols(f"post_:{stencil.Q}")

    to_moment_space = moment_matrix * sp.Matrix(accessor.read(src_field, stencil))
    to_moment_space[0] = rho

    subexpressions.append(Assignment(rho, sum(accessor.read(src_field, stencil))))
    for i in range(lb_method.dim):
        subexpressions.append(Assignment(u_symp[i], velocity_input.center_vector[i]))
    subexpressions.extend(force_model.subs_terms)

    for i in range(stencil.Q):
        subexpressions.append(Assignment(pre[i], to_moment_space[i]))

    if isinstance(lb_method, CentralMomentBasedLbMethod):
        n0 = lb_method.shift_matrix * sp.Matrix(pre)
        to_central = sp.Matrix(sp.symbols(f"kappa_:{stencil.Q}"))
        for i in range(stencil.Q):
            subexpressions.append(Assignment(to_central[i], n0[i]))
        pre = to_central

    collision = sp.Matrix(pre) - rel * (sp.Matrix(pre) - eq) + force_terms

    for i in range(stencil.Q):
        subexpressions.append(Assignment(post[i], collision[i]))

    if isinstance(lb_method, CentralMomentBasedLbMethod):
        n0_back = lb_method.shift_matrix.inv() * sp.Matrix(post)
        from_central = sp.Matrix(sp.symbols(f"kappa_post:{stencil.Q}"))
        for i in range(stencil.Q):
            subexpressions.append(Assignment(from_central[i], n0_back[i]))
        post = from_central

    to_pdf_space = moment_matrix.inv() * sp.Matrix(post)

    for i in range(stencil.Q):
        main_assignments.append(Assignment(accessor.write(dst_field, stencil)[i], to_pdf_space[i]))

    main_assignments.append(Assignment(output_phase_field.center, sum(accessor.write(dst_field, stencil))))

    collision_rule = LbmCollisionRule(lb_method, main_assignments, subexpressions)

    simplification = create_phasefield_simplification_strategy(lb_method)
    collision_rule = simplification(collision_rule)

    return collision_rule


def initializer_kernel_phase_field_lb(lb_phase_field, phi_field, velocity_field, mrt_method, interface_thickness,
                                      fd_stencil=None):
    r"""
    Returns an assignment list for initializing the phase-field distribution functions
    Args:
        lb_phase_field: source field of phase-field distribution function
        phi_field: phase-field
        velocity_field: velocity field
        mrt_method: lattice Boltzmann method of the phase-field lattice Boltzmann step
        interface_thickness: interface thickness
        fd_stencil: stencil to derive the finite differences of the isotropic gradient and the laplacian of the phase
        field. If it is not given the stencil of the LB method will be applied.
    """
    stencil = mrt_method.stencil

    if fd_stencil is None:
        fd_stencil = stencil

    weights = get_weights(stencil, c_s_sq=sp.Rational(1, 3))
    u_symp = sp.symbols(f"u_:{stencil.D}")

    normal_fd = normalized_isotropic_gradient_symbolic(phi_field, stencil, fd_stencil)

    gamma = mrt_method.get_equilibrium_terms()
    gamma = gamma.subs({sp.symbols("rho"): 1})
    gamma_init = gamma.subs({x: y for x, y in zip(u_symp, velocity_field.center_vector)})
    # create the kernels for the initialization of the h field
    h_updates = list()

    def scalar_product(a, b):
        return sum(a_i * b_i for a_i, b_i in zip(a, b))

    f = []
    for i, d in enumerate(stencil):
        f.append(weights[i] * ((1.0 - 4.0 * (phi_field.center - 0.5) ** 2) / interface_thickness)
                 * scalar_product(d, normal_fd[0:stencil.D]))

    for i, _ in enumerate(stencil):
        h_updates.append(Assignment(lb_phase_field.center(i), phi_field.center * gamma_init[i] - 0.5 * f[i]))

    return h_updates


def initializer_kernel_hydro_lb(lb_velocity_field, velocity_field, mrt_method):
    r"""
    Returns an assignment list for initializing the velocity distribution functions
    Args:
        lb_velocity_field: source field of velocity distribution function
        velocity_field: velocity field
        mrt_method: lattice Boltzmann method of the hydrodynamic lattice Boltzmann step
    """
    stencil = mrt_method.stencil
    weights = get_weights(stencil, c_s_sq=sp.Rational(1, 3))
    u_symp = sp.symbols(f"u_:{stencil.D}")

    gamma = mrt_method.get_equilibrium_terms()
    gamma = gamma.subs({sp.symbols("rho"): 1})
    gamma_init = gamma.subs({x: y for x, y in zip(u_symp, velocity_field.center_vector)})

    g_updates = list()
    for i, _ in enumerate(stencil):
        g_updates.append(Assignment(lb_velocity_field.center(i), gamma_init[i] - weights[i]))

    return g_updates
