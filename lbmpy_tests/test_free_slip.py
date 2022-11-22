from dataclasses import replace

import pystencils as ps
import lbmpy as lp
from pystencils.slicing import slice_from_direction

from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.boundaries.boundaryconditions import FreeSlip, NoSlip, ExtrapolationOutflow, UBB

from lbmpy.creationfunctions import create_lb_method, create_lb_update_rule
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments

import numpy as np


def velocity_info_callback(boundary_data, **_):
    boundary_data['vel_1'] = 0
    boundary_data['vel_2'] = 0
    u_max = 0.1
    x, y = boundary_data.link_positions(0), boundary_data.link_positions(1)
    dist = (15 - y) / 15
    boundary_data['vel_0'] = u_max * (1 - dist)


def test_free_slip():
    stencil = lp.LBStencil(lp.Stencil.D3Q27)
    domain_size = (30, 15, 30)
    dim = len(domain_size)

    dh = ps.create_data_handling(domain_size=domain_size)

    src = dh.add_array('src', values_per_cell=stencil.Q)
    dh.fill('src', 0.0, ghost_layers=True)
    dst = dh.add_array('dst', values_per_cell=stencil.Q)
    dh.fill('dst', 0.0, ghost_layers=True)

    velField = dh.add_array('velField', values_per_cell=stencil.D)
    dh.fill('velField', 0.0, ghost_layers=True)

    lbm_config = lp.LBMConfig(stencil=stencil, method=lp.Method.SRT, relaxation_rate=1.8,
                              output={'velocity': velField}, kernel_type='stream_pull_collide')
    method = create_lb_method(lbm_config=lbm_config)

    init = pdf_initialization_assignments(method, 1.0, (0, 0, 0), src.center_vector)

    config = ps.CreateKernelConfig(target=dh.default_target, cpu_openmp=False)
    ast_init = ps.create_kernel(init, config=config)
    kernel_init = ast_init.compile()

    dh.run_kernel(kernel_init)

    lbm_opt = lp.LBMOptimisation(symbolic_field=src, symbolic_temporary_field=dst)
    lbm_config = replace(lbm_config, lb_method=method)

    update = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    ast_kernel = ps.create_kernel(update, config=config)
    kernel = ast_kernel.compile()

    bh = LatticeBoltzmannBoundaryHandling(method, dh, 'src', name="bh")

    inflow = UBB(velocity_info_callback, dim=dim)
    outflow = ExtrapolationOutflow(stencil[4], method)
    wall = NoSlip("wall")
    freeslip = FreeSlip(stencil, (0, -1, 0))

    bh.set_boundary(inflow, slice_from_direction('W', dim))
    bh.set_boundary(outflow, slice_from_direction('E', dim))
    bh.set_boundary(wall, slice_from_direction('S', dim))
    bh.set_boundary(wall, slice_from_direction('T', dim))
    bh.set_boundary(wall, slice_from_direction('B', dim))
    bh.set_boundary(freeslip, slice_from_direction('N', dim))

    for i in range(2000):
        bh()
        dh.run_kernel(kernel)
        dh.swap("src", "dst")

    vel_profile = dh.gather_array('velField')[-2, :, domain_size[2] // 2, 0]
    np.testing.assert_almost_equal(np.gradient(vel_profile)[-1], 0, decimal=3)
