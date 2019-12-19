from collections import OrderedDict

import numpy as np

from lbmpy.creationfunctions import create_lb_method, create_lb_update_rule
from lbmpy.methods.creationfunctions import create_with_discrete_maxwellian_eq_moments
from lbmpy.phasefield_allen_cahn.analytical import analytic_rising_speed
from lbmpy.phasefield_allen_cahn.force_model import MultiphaseForceModel
from lbmpy.phasefield_allen_cahn.kernel_equations import (
    get_collision_assignments_hydro, hydrodynamic_force, initializer_kernel_hydro_lb, initializer_kernel_phase_field_lb,
    interface_tracking_force)
from lbmpy.phasefield_allen_cahn.parameter_calculation import (
    calculate_dimensionless_rising_bubble, calculate_parameters_rti)
from lbmpy.stencils import get_stencil
from pystencils import AssignmentCollection, fields


def test_codegen_3d():
    stencil_phase = get_stencil("D3Q15")
    stencil_hydro = get_stencil("D3Q27")
    assert (len(stencil_phase[0]) == len(stencil_hydro[0]))
    dimensions = len(stencil_hydro[0])

    parameters = calculate_dimensionless_rising_bubble(reference_time=18000,
                                                       density_heavy=1.0,
                                                       bubble_radius=16,
                                                       bond_number=30,
                                                       reynolds_number=420,
                                                       density_ratio=1000,
                                                       viscosity_ratio=100)

    np.isclose(parameters["density_light"], 0.001, rtol=1e-05, atol=1e-08, equal_nan=False)
    np.isclose(parameters["gravitational_acceleration"], -9.876543209876543e-08, rtol=1e-05, atol=1e-08, equal_nan=False)

    parameters = calculate_parameters_rti(reference_length=128,
                                          reference_time=18000,
                                          density_heavy=1.0,
                                          capillary_number=9.1,
                                          reynolds_number=128,
                                          atwood_number=1.0,
                                          peclet_number=744,
                                          density_ratio=3,
                                          viscosity_ratio=3)

    np.isclose(parameters["density_light"], 1/3, rtol=1e-05, atol=1e-08, equal_nan=False)
    np.isclose(parameters["gravitational_acceleration"], -3.9506172839506174e-07, rtol=1e-05, atol=1e-08, equal_nan=False)
    np.isclose(parameters["mobility"], 0.0012234169653524492, rtol=1e-05, atol=1e-08, equal_nan=False)

    rs = analytic_rising_speed(1-6, 20, 0.01)
    np.isclose(rs, 16666.666666666668, rtol=1e-05, atol=1e-08, equal_nan=False)

    density_liquid = 1.0
    density_gas = 0.001
    surface_tension = 0.0001
    M = 0.02

    # phase-field parameter
    drho3 = (density_liquid - density_gas) / 3
    # interface thickness
    W = 5
    # coefficient related to surface tension
    beta = 12.0 * (surface_tension / W)
    # coefficient related to surface tension
    kappa = 1.5 * surface_tension * W
    # relaxation rate allen cahn (h)
    w_c = 1.0 / (0.5 + (3.0 * M))

    # fields
    u = fields("vel_field(" + str(dimensions) + "): [" + str(dimensions) + "D]", layout='fzyx')
    C = fields("phase_field: [" + str(dimensions) + "D]", layout='fzyx')

    h = fields("lb_phase_field(" + str(len(stencil_phase)) + "): [" + str(dimensions) + "D]", layout='fzyx')
    h_tmp = fields("lb_phase_field_tmp(" + str(len(stencil_phase)) + "): [" + str(dimensions) + "D]", layout='fzyx')

    g = fields("lb_velocity_field(" + str(len(stencil_hydro)) + "): [" + str(dimensions) + "D]", layout='fzyx')
    g_tmp = fields("lb_velocity_field_tmp(" + str(len(stencil_hydro)) + "): [" + str(dimensions) + "D]", layout='fzyx')

    # calculate the relaxation rate for the hydro lb as well as the body force
    density = density_gas + C.center * (density_liquid - density_gas)
    # force acting on all phases of the model
    body_force = np.array([0, 1e-6, 0])

    relaxation_time = 0.03 + 0.5
    relaxation_rate = 1.0 / relaxation_time

    method_phase = create_lb_method(stencil=stencil_phase, method='srt', relaxation_rate=w_c, compressible=True)

    mrt = create_lb_method(method="mrt", weighted=False, stencil=stencil_hydro,
                           relaxation_rates=[1, 1, relaxation_rate, 1, 1, 1, 1])
    rr_dict = OrderedDict(zip(mrt.moments, mrt.relaxation_rates))

    method_hydro = create_with_discrete_maxwellian_eq_moments(stencil_hydro, rr_dict, compressible=False)

    # create the kernels for the initialization of the g and h field
    h_updates = initializer_kernel_phase_field_lb(h, C, u, method_phase, W)
    g_updates = initializer_kernel_hydro_lb(g, u, method_hydro)

    force_h = [f / 3 for f in interface_tracking_force(C, stencil_phase, W)]
    force_model_h = MultiphaseForceModel(force=force_h)

    force_g = hydrodynamic_force(g, C, method_hydro,
                                 relaxation_time, density_liquid, density_gas, kappa, beta, body_force)
    force_model_g = MultiphaseForceModel(force=force_g, rho=density)

    h_tmp_symbol_list = [h_tmp.center(i) for i, _ in enumerate(stencil_phase)]
    sum_h = np.sum(h_tmp_symbol_list[:])

    method_phase = create_lb_method(stencil=stencil_phase,
                                    method='srt',
                                    relaxation_rate=w_c,
                                    compressible=True,
                                    force_model=force_model_h)

    allen_cahn_lb = create_lb_update_rule(lb_method=method_phase,
                                          velocity_input=u,
                                          compressible=True,
                                          optimization={"symbolic_field": h,
                                                        "symbolic_temporary_field": h_tmp},
                                          kernel_type='stream_pull_collide')

    allen_cahn_lb.set_main_assignments_from_dict({**allen_cahn_lb.main_assignments_dict, **{C.center: sum_h}})
    allen_cahn_update_rule = AssignmentCollection(main_assignments=allen_cahn_lb.main_assignments,
                                                  subexpressions=allen_cahn_lb.subexpressions)
    # ---------------------------------------------------------------------------------------------------------

    method_hydro = create_with_discrete_maxwellian_eq_moments(stencil_hydro, rr_dict, force_model=force_model_g)

    hydro_lb_update_rule_normal = get_collision_assignments_hydro(lb_method=method_hydro,
                                                                  density=density,
                                                                  velocity_input=u,
                                                                  force=force_g,
                                                                  optimization={"symbolic_field": g,
                                                                                "symbolic_temporary_field": g_tmp},
                                                                  kernel_type='collide_only')

    hydro_lb_update_rule_push = get_collision_assignments_hydro(lb_method=method_hydro,
                                                                density=density,
                                                                velocity_input=u,
                                                                force=force_g,
                                                                optimization={"symbolic_field": g,
                                                                              "symbolic_temporary_field": g_tmp},
                                                                kernel_type='collide_stream_push')

    hydro_lb_update_rule_generic_fields = get_collision_assignments_hydro(lb_method=method_hydro,
                                                                          density=density,
                                                                          velocity_input=u,
                                                                          force=force_g,
                                                                          kernel_type='collide_only')
