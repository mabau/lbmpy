"""
Test Shan-Chen two-component implementation against reference implementation
"""

import lbmpy

import pystencils as ps
import sympy as sp
import numpy as np


def test_shan_chen_two_component():
    from lbmpy.enums import Stencil
    from lbmpy import LBMConfig, ForceModel, create_lb_update_rule
    from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
    from lbmpy.creationfunctions import create_stream_pull_with_output_kernel
    from lbmpy.maxwellian_equilibrium import get_weights

    N = 64
    omega_a = 1.
    omega_b = 1.

    # interaction strength
    g_aa = 0.
    g_ab = g_ba = 6.
    g_bb = 0.

    rho0 = 1.

    stencil = lbmpy.LBStencil(Stencil.D2Q9)
    weights = get_weights(stencil, c_s_sq=sp.Rational(1, 3))

    dim = stencil.D
    dh = ps.create_data_handling((N,) * dim, periodicity=True, default_target=ps.Target.CPU)

    src_a = dh.add_array('src_a', values_per_cell=stencil.Q)
    dst_a = dh.add_array_like('dst_a', 'src_a')

    src_b = dh.add_array('src_b', values_per_cell=stencil.Q)
    dst_b = dh.add_array_like('dst_b', 'src_b')

    ρ_a = dh.add_array('rho_a')
    ρ_b = dh.add_array('rho_b')
    u_a = dh.add_array('u_a', values_per_cell=stencil.D)
    u_b = dh.add_array('u_b', values_per_cell=stencil.D)
    u_bary = dh.add_array_like('u_bary', u_a.name)

    f_a = dh.add_array('f_a', values_per_cell=stencil.D)
    f_b = dh.add_array_like('f_b', f_a.name)

    def psi(dens):
        return rho0 * (1. - sp.exp(-dens / rho0))

    zero_vec = sp.Matrix([0] * stencil.D)

    force_a = zero_vec
    for factor, ρ in zip([g_aa, g_ab], [ρ_a, ρ_b]):
        force_a += sum((psi(ρ[d]) * w_d * sp.Matrix(d)
                        for d, w_d in zip(stencil, weights)),
                       zero_vec) * psi(ρ_a.center) * -1 * factor

    force_b = zero_vec
    for factor, ρ in zip([g_ba, g_bb], [ρ_a, ρ_b]):
        force_b += sum((psi(ρ[d]) * w_d * sp.Matrix(d)
                        for d, w_d in zip(stencil, weights)),
                       zero_vec) * psi(ρ_b.center) * -1 * factor

    f_expressions = ps.AssignmentCollection([
        ps.Assignment(f_a.center_vector, force_a),
        ps.Assignment(f_b.center_vector, force_b)
    ])

    # calculate the velocity without force correction
    u_temp = ps.AssignmentCollection(ps.Assignment(u_bary.center_vector,
                                                   (ρ_a.center * u_a.center_vector
                                                    - f_a.center_vector / 2 + ρ_b.center * u_b.center_vector
                                                    - f_b.center_vector / 2) / (ρ_a.center + ρ_b.center)))

    # add the force correction to the velocity
    u_corr = ps.AssignmentCollection(ps.Assignment(u_bary.center_vector,
                                                   u_bary.center_vector
                                                   + (f_a.center_vector / 2 + f_b.center_vector / 2) / (
                                                               ρ_a.center + ρ_b.center)))

    lbm_config_a = LBMConfig(stencil=stencil, relaxation_rate=omega_a, compressible=True,
                             velocity_input=u_bary, density_input=ρ_a, force_model=ForceModel.GUO,
                             force=f_a, kernel_type='collide_only')

    lbm_config_b = LBMConfig(stencil=stencil, relaxation_rate=omega_b, compressible=True,
                             velocity_input=u_bary, density_input=ρ_b, force_model=ForceModel.GUO,
                             force=f_b, kernel_type='collide_only')

    collision_a = create_lb_update_rule(lbm_config=lbm_config_a,
                                        optimization={'symbolic_field': src_a})

    collision_b = create_lb_update_rule(lbm_config=lbm_config_b,
                                        optimization={'symbolic_field': src_b})

    stream_a = create_stream_pull_with_output_kernel(collision_a.method, src_a, dst_a,
                                                     {'density': ρ_a, 'velocity': u_a})
    stream_b = create_stream_pull_with_output_kernel(collision_b.method, src_b, dst_b,
                                                     {'density': ρ_b, 'velocity': u_b})

    config = ps.CreateKernelConfig(target=dh.default_target)

    stream_a_kernel = ps.create_kernel(stream_a, config=config).compile()
    stream_b_kernel = ps.create_kernel(stream_b, config=config).compile()
    collision_a_kernel = ps.create_kernel(collision_a, config=config).compile()
    collision_b_kernel = ps.create_kernel(collision_b, config=config).compile()

    force_kernel = ps.create_kernel(f_expressions, config=config).compile()
    u_temp_kernel = ps.create_kernel(u_temp, config=config).compile()
    u_corr_kernel = ps.create_kernel(u_corr, config=config).compile()

    init_a = macroscopic_values_setter(collision_a.method, velocity=(0, 0),
                                       pdfs=src_a.center_vector, density=ρ_a.center)
    init_b = macroscopic_values_setter(collision_b.method, velocity=(0, 0),
                                       pdfs=src_b.center_vector, density=ρ_b.center)
    init_a_kernel = ps.create_kernel(init_a, ghost_layers=0).compile()
    init_b_kernel = ps.create_kernel(init_b, ghost_layers=0).compile()

    sync_pdfs = dh.synchronization_function([src_a.name, src_b.name])
    sync_ρs = dh.synchronization_function([ρ_a.name, ρ_b.name])

    dh.fill(ρ_a.name, 0.1, slice_obj=ps.make_slice[:, :0.5])
    dh.fill(ρ_a.name, 0.9, slice_obj=ps.make_slice[:, 0.5:])

    dh.fill(ρ_b.name, 0.9, slice_obj=ps.make_slice[:, :0.5])
    dh.fill(ρ_b.name, 0.1, slice_obj=ps.make_slice[:, 0.5:])

    dh.fill(u_a.name, 0.0)
    dh.fill(u_b.name, 0.0)
    dh.fill(f_a.name, 0.0)
    dh.fill(f_b.name, 0.0)
    dh.run_kernel(u_temp_kernel)

    dh.run_kernel(init_a_kernel)
    dh.run_kernel(init_b_kernel)

    for i in range(1000):
        sync_ρs()
        dh.run_kernel(force_kernel)
        dh.run_kernel(u_corr_kernel)
        dh.run_kernel(collision_a_kernel)
        dh.run_kernel(collision_b_kernel)

        sync_pdfs()
        dh.run_kernel(stream_a_kernel)
        dh.run_kernel(stream_b_kernel)
        dh.run_kernel(u_temp_kernel)

        dh.swap(src_a.name, dst_a.name)
        dh.swap(src_b.name, dst_b.name)

    # reference generated from https://github.com/lbm-principles-practice/code/blob/master/chapter9/shanchen.cpp with
    # const int nsteps = 1000;
    # const int noutput = 1000;
    # const int nfluids = 2;
    # const double gA = 0;

    ref_a = np.array([0.213948, 0.0816724, 0.0516763, 0.0470179, 0.0480882, 0.0504771, 0.0531983, 0.0560094, 0.0588071,
                      0.0615311, 0.064102, 0.0664467, 0.0684708, 0.070091, 0.0712222, 0.0718055, 0.0718055, 0.0712222,
                      0.070091, 0.0684708, 0.0664467, 0.064102, 0.0615311, 0.0588071, 0.0560094, 0.0531983, 0.0504771,
                      0.0480882, 0.0470179, 0.0516763, 0.0816724, 0.213948, 0.517153, 0.833334, 0.982884, 1.0151,
                      1.01361, 1.0043, 0.993178, 0.981793, 0.970546, 0.959798, 0.949751, 0.940746, 0.933035, 0.926947,
                      0.922713, 0.920548, 0.920548, 0.922713, 0.926947, 0.933035, 0.940746, 0.949751, 0.959798,
                      0.970546, 0.981793, 0.993178, 1.0043, 1.01361, 1.0151, 0.982884, 0.833334, 0.517153])
    ref_b = np.array([0.517153, 0.833334, 0.982884, 1.0151, 1.01361, 1.0043, 0.993178, 0.981793, 0.970546, 0.959798,
                      0.949751, 0.940746, 0.933035, 0.926947, 0.922713, 0.920548, 0.920548, 0.922713, 0.926947,
                      0.933035, 0.940746, 0.949751, 0.959798, 0.970546, 0.981793, 0.993178, 1.0043, 1.01361, 1.0151,
                      0.982884, 0.833334, 0.517153, 0.213948, 0.0816724, 0.0516763, 0.0470179, 0.0480882, 0.0504771,
                      0.0531983, 0.0560094, 0.0588071, 0.0615311, 0.064102, 0.0664467, 0.0684708, 0.070091, 0.0712222,
                      0.0718055, 0.0718055, 0.0712222, 0.070091, 0.0684708, 0.0664467, 0.064102, 0.0615311, 0.0588071,
                      0.0560094, 0.0531983, 0.0504771, 0.0480882, 0.0470179, 0.0516763, 0.0816724, 0.213948])
    assert np.allclose(dh.gather_array(ρ_a.name)[0], ref_a)
    assert np.allclose(dh.gather_array(ρ_b.name)[0], ref_b)
