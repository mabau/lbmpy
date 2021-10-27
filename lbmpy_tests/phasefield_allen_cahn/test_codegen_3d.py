from lbmpy.creationfunctions import create_lb_method, LBMConfig, LBMOptimisation
from lbmpy.enums import Method, Stencil
from lbmpy.phasefield_allen_cahn.force_model import MultiphaseForceModel
from lbmpy.phasefield_allen_cahn.kernel_equations import (get_collision_assignments_phase,
                                                          get_collision_assignments_hydro, hydrodynamic_force,
                                                          initializer_kernel_hydro_lb,
                                                          initializer_kernel_phase_field_lb,
                                                          interface_tracking_force)
from lbmpy.stencils import LBStencil
from pystencils import fields


def test_allen_cahn_lb():
    stencil_phase = LBStencil(Stencil.D3Q15)
    # fields
    u = fields("vel_field(" + str(stencil_phase.D) + "): [" + str(stencil_phase.D) + "D]", layout='fzyx')
    C = fields("phase_field: [" + str(stencil_phase.D) + "D]", layout='fzyx')
    C_tmp = fields("phase_field_tmp: [" + str(stencil_phase.D) + "D]", layout='fzyx')

    h = fields("lb_phase_field(" + str(len(stencil_phase)) + "): [" + str(stencil_phase.D) + "D]", layout='fzyx')
    h_tmp = fields("lb_phase_field_tmp(" + str(len(stencil_phase)) + "): [" + str(stencil_phase.D) + "D]", layout='fzyx')

    M = 0.02
    W = 5
    w_c = 1.0 / (0.5 + (3.0 * M))

    lbm_config = LBMConfig(stencil=stencil_phase, method=Method.SRT,
                           relaxation_rate=w_c, compressible=True)

    method_phase = create_lb_method(lbm_config=lbm_config)

    h_updates = initializer_kernel_phase_field_lb(h, C, u, method_phase, W)

    force_h = [f / 3 for f in interface_tracking_force(C, stencil_phase, W)]
    force_model_h = MultiphaseForceModel(force=force_h)

    allen_cahn_lb = get_collision_assignments_phase(lb_method=method_phase,
                                                    velocity_input=u,
                                                    output={'density': C_tmp},
                                                    force_model=force_model_h,
                                                    symbolic_fields={"symbolic_field": h,
                                                                     "symbolic_temporary_field": h_tmp},
                                                    kernel_type='stream_pull_collide')

    allen_cahn_lb = get_collision_assignments_phase(lb_method=method_phase,
                                                    velocity_input=u,
                                                    output={'density': C_tmp},
                                                    force_model=force_model_h,
                                                    symbolic_fields={"symbolic_field": h,
                                                                     "symbolic_temporary_field": h_tmp},
                                                    kernel_type='collide_only')


def test_hydro_lb():
    stencil_hydro = LBStencil(Stencil.D3Q27)

    density_liquid = 1.0
    density_gas = 0.001
    surface_tension = 0.0001
    W = 5

    # phase-field parameter
    drho3 = (density_liquid - density_gas) / 3
    # coefficient related to surface tension
    beta = 12.0 * (surface_tension / W)
    # coefficient related to surface tension
    kappa = 1.5 * surface_tension * W

    u = fields("vel_field(" + str(stencil_hydro.D) + "): [" + str(stencil_hydro.D) + "D]", layout='fzyx')
    C = fields("phase_field: [" + str(stencil_hydro.D) + "D]", layout='fzyx')

    g = fields("lb_velocity_field(" + str(stencil_hydro.Q) + "): [" + str(stencil_hydro.D) + "D]", layout='fzyx')
    g_tmp = fields("lb_velocity_field_tmp(" + str(stencil_hydro.Q) + "): [" + str(stencil_hydro.D) + "D]", layout='fzyx')

    # calculate the relaxation rate for the hydro lb as well as the body force
    density = density_gas + C.center * (density_liquid - density_gas)
    # force acting on all phases of the model
    body_force = [0, 0, 0]

    relaxation_time = 0.03 + 0.5
    relaxation_rate = 1.0 / relaxation_time

    lbm_config = LBMConfig(stencil=stencil_hydro, method=Method.MRT,
                           weighted=True, relaxation_rates=[relaxation_rate, 1, 1, 1, 1, 1])

    method_hydro = create_lb_method(lbm_config=lbm_config)

    # create the kernels for the initialization of the g and h field
    g_updates = initializer_kernel_hydro_lb(g, u, method_hydro)

    force_g = hydrodynamic_force(g, C, method_hydro,
                                 relaxation_time, density_liquid, density_gas, kappa, beta, body_force)
    force_model_g = MultiphaseForceModel(force=force_g, rho=density)

    hydro_lb_update_rule_normal = get_collision_assignments_hydro(lb_method=method_hydro,
                                                                  density=density,
                                                                  velocity_input=u,
                                                                  force_model=force_model_g,
                                                                  sub_iterations=2,
                                                                  symbolic_fields={"symbolic_field": g,
                                                                                   "symbolic_temporary_field": g_tmp},
                                                                  kernel_type='collide_only')

    hydro_lb_update_rule_push = get_collision_assignments_hydro(lb_method=method_hydro,
                                                                density=density,
                                                                velocity_input=u,
                                                                force_model=force_model_g,
                                                                sub_iterations=2,
                                                                symbolic_fields={"symbolic_field": g,
                                                                                 "symbolic_temporary_field": g_tmp},
                                                                kernel_type='collide_stream_push')
