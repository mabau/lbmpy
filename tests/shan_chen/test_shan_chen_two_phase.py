"""
Test Shan-Chen two-phase implementation against reference implementation
"""

import lbmpy

import pystencils as ps
import sympy as sp
import numpy as np


def test_shan_chen_two_phase():
    from lbmpy.enums import Stencil
    from lbmpy import LBMConfig, ForceModel, create_lb_update_rule
    from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
    from lbmpy.creationfunctions import create_stream_pull_with_output_kernel, create_lb_method
    from lbmpy.maxwellian_equilibrium import get_weights

    N = 64
    omega = 1.
    g_aa = -4.7
    rho0 = 1.

    stencil = lbmpy.LBStencil(Stencil.D2Q9)
    weights = get_weights(stencil, c_s_sq=sp.Rational(1, 3))

    dh = ps.create_data_handling((N, ) * stencil.D, periodicity=True, default_target=ps.Target.CPU)

    src = dh.add_array('src', values_per_cell=stencil.Q)
    dst = dh.add_array_like('dst', 'src')

    ρ = dh.add_array('rho')

    def psi(dens):
        return rho0 * (1. - sp.exp(-dens / rho0))

    zero_vec = sp.Matrix([0] * stencil.D)

    force = sum((psi(ρ[d]) * w_d * sp.Matrix(d)
                 for d, w_d in zip(stencil, weights)), zero_vec) * psi(ρ.center) * -1 * g_aa

    lbm_config = LBMConfig(stencil=stencil, relaxation_rate=omega, compressible=True,
                           force_model=ForceModel.GUO, force=force, kernel_type='collide_only')

    collision = create_lb_update_rule(lbm_config=lbm_config,
                                      optimization={'symbolic_field': src})

    stream = create_stream_pull_with_output_kernel(collision.method, src, dst, {'density': ρ})

    config = ps.CreateKernelConfig(target=dh.default_target, cpu_openmp=False)

    stream_kernel = ps.create_kernel(stream, config=config).compile()
    collision_kernel = ps.create_kernel(collision, config=config).compile()

    method_without_force = create_lb_method(LBMConfig(stencil=stencil, relaxation_rate=omega, compressible=True))
    init_assignments = macroscopic_values_setter(method_without_force, velocity=(0, 0),
                                                 pdfs=src.center_vector, density=ρ.center)

    init_kernel = ps.create_kernel(init_assignments, ghost_layers=0, config=config).compile()

    for x in range(N):
        for y in range(N):
            if (x - N / 2)**2 + (y - N / 2)**2 <= 15**2:
                dh.fill(ρ.name, 2.1, slice_obj=[x, y])
            else:
                dh.fill(ρ.name, 0.15, slice_obj=[x, y])

    dh.run_kernel(init_kernel)

    sync_pdfs = dh.synchronization_function([src.name])
    sync_ρs = dh.synchronization_function([ρ.name])

    for i in range(1000):
        sync_ρs()
        dh.run_kernel(collision_kernel)

        sync_pdfs()
        dh.run_kernel(stream_kernel)

        dh.swap(src.name, dst.name)

    # reference generated from https://github.com/lbm-principles-practice/code/blob/master/chapter9/shanchen.cpp with
    # const int nsteps = 1000;
    # const int noutput = 1000;

    ref = np.array([0.185757, 0.185753, 0.185743, 0.185727, 0.185703, 0.185672, 0.185636, 0.185599, 0.185586, 0.185694,
                    0.186302, 0.188901, 0.19923, 0.238074, 0.365271, 0.660658, 1.06766, 1.39673, 1.56644, 1.63217,
                    1.65412, 1.66064, 1.66207, 1.66189, 1.66123, 1.66048, 1.65977, 1.65914, 1.65861, 1.6582, 1.6579,
                    1.65772, 1.65766, 1.65772, 1.6579, 1.6582, 1.65861, 1.65914, 1.65977, 1.66048, 1.66123, 1.66189,
                    1.66207, 1.66064, 1.65412, 1.63217, 1.56644, 1.39673, 1.06766, 0.660658, 0.365271, 0.238074,
                    0.19923, 0.188901, 0.186302, 0.185694, 0.185586, 0.185599, 0.185636, 0.185672, 0.185703, 0.185727,
                    0.185743, 0.185753])
    assert np.allclose(dh.gather_array(ρ.name)[N // 2], ref)
