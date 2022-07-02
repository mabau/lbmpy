import math
import pytest
from dataclasses import replace

import pystencils as ps
from pystencils.slicing import slice_from_direction

from lbmpy import pdf_initialization_assignments

from lbmpy.creationfunctions import create_lb_method, create_lb_update_rule, LBMConfig, LBMOptimisation
from lbmpy.enums import Method, Stencil
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.boundaries.boundaryconditions import DiffusionDirichlet, NeumannByCopy
from lbmpy.geometry import add_box_boundary
from lbmpy.stencils import LBStencil
from lbmpy.maxwellian_equilibrium import get_weights

import sympy as sp
import numpy as np


def test_diffusion_boundary():
    domain_size = (10, 10)
    stencil = LBStencil(Stencil.D2Q9)
    weights = get_weights(stencil, c_s_sq=sp.Rational(1, 3))
    concentration = 1.0

    # Data Handling
    dh = ps.create_data_handling(domain_size=domain_size)

    dh.add_array('pdfs', values_per_cell=stencil.Q)
    dh.fill("pdfs", 0.0, ghost_layers=True)

    lbm_config = LBMConfig(stencil=stencil, method=Method.SRT, relaxation_rate=1.8,
                           compressible=True, zero_centered=False)
    method = create_lb_method(lbm_config=lbm_config)

    # Boundary Handling
    bh = LatticeBoltzmannBoundaryHandling(method, dh, 'pdfs', name="bh")
    add_box_boundary(bh, boundary=DiffusionDirichlet(concentration))

    bh()

    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[0, 1:-2, 4] - 2 * weights[4]) < 1e-14)
    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[0, 1:-2, 6] - 2 * weights[6]) < 1e-14)
    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[0, 2:-1, 8] - 2 * weights[8]) < 1e-14)

    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[-1, 1:-2, 3] - 2 * weights[3]) < 1e-14)
    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[-1, 1:-2, 5] - 2 * weights[5]) < 1e-14)
    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[-1, 2:-1, 7] - 2 * weights[7]) < 1e-14)

    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[1:-2, 0, 1] - 2 * weights[1]) < 1e-14)
    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[2:, 0, 5] - 2 * weights[5]) < 1e-14)
    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[:-2, 0, 6] - 2 * weights[6]) < 1e-14)

    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[1:-2, -1, 2] - 2 * weights[2]) < 1e-14)
    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[2:, -1, 7] - 2 * weights[7]) < 1e-14)
    assert all(np.abs(dh.gather_array("pdfs", ghost_layers=True)[:-2, -1, 8] - 2 * weights[8]) < 1e-14)


@pytest.mark.longrun
def test_diffusion():
    """
      Runs the "Diffusion from Plate in Uniform Flow" benchmark as it is described
      in [ch. 8.6.3, The Lattice Boltzmann Method, Krüger et al.].

                dC/dy = 0
            ┌───────────────┐
            │     → → →     │
      C = 0 │     → u →     │ dC/dx = 0
            │     → → →     │
            └───────────────┘
                  C = 1

      The analytical solution is given by:
        C(x,y) = 1 * erfc(y / sqrt(4Dx/u))

      The hydrodynamic field is not simulated, instead a constant velocity is assumed.
    """
    pytest.importorskip("pycuda")
    # Parameters
    domain_size = (1600, 160)
    omega = 1.38
    diffusion = (1 / omega - 0.5) / 3
    velocity = 0.05
    time_steps = 50000
    stencil = LBStencil(Stencil.D2Q9)
    target = ps.Target.GPU

    # Data Handling
    dh = ps.create_data_handling(domain_size=domain_size, default_target=target)

    vel_field = dh.add_array('vel_field', values_per_cell=stencil.D)
    dh.fill('vel_field', velocity, 0, ghost_layers=True)
    dh.fill('vel_field', 0.0, 1, ghost_layers=True)

    con_field = dh.add_array('con_field', values_per_cell=1)
    dh.fill('con_field', 0.0, ghost_layers=True)

    pdfs = dh.add_array('pdfs', values_per_cell=stencil.Q)
    dh.fill('pdfs', 0.0, ghost_layers=True)
    pdfs_tmp = dh.add_array('pdfs_tmp', values_per_cell=stencil.Q)
    dh.fill('pdfs_tmp', 0.0, ghost_layers=True)

    # Lattice Boltzmann method
    lbm_config = LBMConfig(stencil=stencil, method=Method.MRT, relaxation_rates=[1, 1.5, 1, 1.5, 1],
                           zero_centered=False,
                           velocity_input=vel_field, output={'density': con_field}, compressible=True,
                           weighted=True, kernel_type='stream_pull_collide')

    lbm_opt = LBMOptimisation(symbolic_field=pdfs, symbolic_temporary_field=pdfs_tmp)
    config = ps.CreateKernelConfig(target=dh.default_target, cpu_openmp=True)

    method = create_lb_method(lbm_config=lbm_config)
    method.set_conserved_moments_relaxation_rate(omega)

    lbm_config = replace(lbm_config, lb_method=method)
    update_rule = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)
    kernel = ps.create_kernel(update_rule, config=config).compile()

    # PDF initalization
    init = pdf_initialization_assignments(method, con_field.center,
                                          vel_field.center_vector, pdfs.center_vector)
    dh.run_kernel(ps.create_kernel(init).compile())

    dh.all_to_gpu()

    # Boundary Handling
    bh = LatticeBoltzmannBoundaryHandling(update_rule.method, dh, 'pdfs', name="bh", target=dh.default_target)
    add_box_boundary(bh, boundary=NeumannByCopy())
    bh.set_boundary(DiffusionDirichlet(0), slice_from_direction('W', dh.dim))
    bh.set_boundary(DiffusionDirichlet(1), slice_from_direction('S', dh.dim))

    # Timeloop
    for i in range(time_steps):
        bh()
        dh.run_kernel(kernel)
        dh.swap("pdfs", "pdfs_tmp")

    dh.all_to_cpu()
    # Verification
    x = np.arange(1, domain_size[0], 1)
    y = np.arange(0, domain_size[1], 1)
    X, Y = np.meshgrid(x, y)
    analytical = np.zeros(domain_size)
    analytical[1:, :] = np.vectorize(math.erfc)(Y / np.vectorize(math.sqrt)(4 * diffusion * X / velocity)).transpose()
    simulated = dh.gather_array('con_field', ghost_layers=False)

    residual = 0
    for i in x:
        for j in y:
            residual += (simulated[i, j] - analytical[i, j]) ** 2
    residual = math.sqrt(residual / (domain_size[0] * domain_size[1]))

    assert residual < 1e-2
