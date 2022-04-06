"""
Test shear flow velocity and pressureagainst analytical solutions
"""


import numpy as np
import pytest
import sympy as sp

from lbmpy.boundaries import UBB
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.creationfunctions import create_lb_update_rule, create_stream_pull_with_output_kernel,\
    LBMConfig, LBMOptimisation
from lbmpy.enums import Method, Stencil
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity
from lbmpy.stencils import LBStencil

import pystencils as ps


def shear_flow(x, t, nu, v, h, k_max):
    """
    Analytical solution for driven shear flow between two plates.

    Parameters
    ----------
    x : :obj:`float`
        Position from the left plane.
    t : :obj:`float`
        Time since start of the shearing.
    nu : :obj:`float`
        Kinematic viscosity.
    v : :obj:`float`
        Shear rate.
    h : :obj:`float`
        Distance between the plates.
    k_max : :obj:`int`
        Maximum considered wave number.

    Returns
    -------
    :obj:`double` : Analytical velocity

    """

    u = x / h - 0.5
    for k in np.arange(1, k_max + 1):
        u += 1.0 / (np.pi * k) * np.exp(
            -4 * np.pi ** 2 * nu * k ** 2 / h ** 2 * t) * np.sin(2 * np.pi / h * k * x)
    return -v * u


rho_0 = 2.2  # density
eta = 1.6  # kinematic viscosity
width = 40  # of box
wall_thickness = 2
actual_width = width - wall_thickness  # subtract boundary layer from box width
shear_velocity = 0.2  # scale by width to keep stable
t_max = 2000


@pytest.mark.parametrize('target', (ps.Target.CPU, ps.Target.GPU))
@pytest.mark.parametrize('stencil_name', (Stencil.D2Q9, Stencil.D3Q19))
@pytest.mark.parametrize('zero_centered', [True, False])
def test_shear_flow(target, stencil_name, zero_centered):

    # Cuda
    if target == ps.Target.GPU:
        pytest.importorskip("pycuda")

    # LB parameters
    stencil = LBStencil(stencil_name)

    if stencil.D == 2:
        L = (4, width)
    elif stencil.D == 3:
        L = (4, width, 4)
    else:
        raise Exception()
    periodicity = [True, False] + [True] * (stencil.D - 2)

    omega = relaxation_rate_from_lattice_viscosity(eta)

    # ## Data structures
    dh = ps.create_data_handling(L, periodicity=periodicity, default_target=target)

    src = dh.add_array('src', values_per_cell=stencil.Q)
    dst = dh.add_array_like('dst', 'src')
    ρ = dh.add_array('rho', latex_name='\\rho', values_per_cell=1)
    u = dh.add_array('u', values_per_cell=stencil.D)
    p = dh.add_array('p', values_per_cell=stencil.D**2)

    # LB Setup
    lbm_config = LBMConfig(stencil=stencil, relaxation_rate=omega, method=Method.TRT,
                           compressible=True, zero_centered=zero_centered,
                           kernel_type='collide_only')
    lbm_opt = LBMOptimisation(symbolic_field=src)
    collision = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    stream = create_stream_pull_with_output_kernel(collision.method, src, dst, {'velocity': u})
    config = ps.CreateKernelConfig(cpu_openmp=False, target=dh.default_target)

    stream_kernel = ps.create_kernel(stream, config=config).compile()
    collision_kernel = ps.create_kernel(collision, config=config).compile()

    # Boundaries
    lbbh = LatticeBoltzmannBoundaryHandling(collision.method, dh, src.name, target=dh.default_target)

    # Second moment test setup
    cqc = collision.method.conserved_quantity_computation
    getter_eqs = cqc.output_equations_from_pdfs(src.center_vector,
                                                {'moment2': p})

    kernel_compute_p = ps.create_kernel(getter_eqs, config=config).compile()

    # ## Set up the simulation

    init = macroscopic_values_setter(collision.method, velocity=(0,) * dh.dim,
                                     pdfs=src.center_vector, density=ρ.center)
    init_kernel = ps.create_kernel(init, ghost_layers=0).compile()

    vel_vec = sp.Matrix([0.5 * shear_velocity] + [0] * (stencil.D - 1))
    if stencil.D == 2:
        lbbh.set_boundary(UBB(velocity=vel_vec), ps.make_slice[:, :wall_thickness])
        lbbh.set_boundary(UBB(velocity=-vel_vec), ps.make_slice[:, -wall_thickness:])
    elif stencil.D == 3:
        lbbh.set_boundary(UBB(velocity=vel_vec), ps.make_slice[:, :wall_thickness, :])
        lbbh.set_boundary(UBB(velocity=-vel_vec), ps.make_slice[:, -wall_thickness:, :])
    else:
        raise Exception()

    for bh in lbbh, :
        assert len(bh._boundary_object_to_boundary_info) == 2, "Restart kernel to clear boundaries"

    def init():
        dh.fill(ρ.name, rho_0)
        dh.fill(u.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(u.name, 0)

        dh.run_kernel(init_kernel)

    sync_pdfs = dh.synchronization_function([src.name])

    # Time loop
    def time_loop(steps):
        dh.all_to_gpu()
        for i in range(steps):
            dh.run_kernel(collision_kernel)
            sync_pdfs()
            lbbh()
            dh.run_kernel(stream_kernel)
            dh.run_kernel(kernel_compute_p)

            dh.swap(src.name, dst.name)

        if u.name in dh.gpu_arrays:
            dh.to_cpu(u.name)
        uu = dh.gather_array(u.name)
        # average periodic directions
        if stencil.D == 3:  # dont' swap order
            uu = np.average(uu, axis=2)
        uu = np.average(uu, axis=0)

        if p.name in dh.gpu_arrays:
            dh.to_cpu(p.name)
        pp = dh.gather_array(p.name)
        # average periodic directions
        if stencil.D == 3:  # dont' swap order
            pp = np.average(pp, axis=2)
        pp = np.average(pp, axis=0)

        # cut off wall regions
        uu = uu[wall_thickness:-wall_thickness]
        pp = pp[wall_thickness:-wall_thickness]

        if stencil.D == 2:
            pp = pp.reshape((len(pp), 2, 2))
        if stencil.D == 3:
            pp = pp.reshape((len(pp), 3, 3))
        return uu, pp

    init()
    # Simulation
    profile, pressure_profile = time_loop(t_max)

    expected = shear_flow(x=(np.arange(0, actual_width) + .5),
                          t=t_max,
                          nu=eta / rho_0,
                          v=shear_velocity,
                          h=actual_width,
                          k_max=100)

    if stencil.D == 2:
        shear_direction = np.array((1, 0), dtype=float)
        shear_plane_normal = np.array((0, 1), dtype=float)
    if stencil.D == 3:
        shear_direction = np.array((1, 0, 0), dtype=float)
        shear_plane_normal = np.array((0, 1, 0), dtype=float)

    shear_rate = shear_velocity / actual_width
    dynamic_viscosity = eta * rho_0
    correction_factor = eta / (eta - 1. / 6.)

    p_expected = rho_0 * np.identity(dh.dim) / 3.0 + dynamic_viscosity * shear_rate / correction_factor * (
        np.outer(shear_plane_normal, shear_direction) + np.transpose(np.outer(shear_plane_normal, shear_direction)))

    # Sustract the tensorproduct of the velosity to get the pressure
    pressure_profile[:, 0, 0] -= rho_0 * profile[:, 0]**2
    
    np.testing.assert_allclose(profile[:, 0], expected[1:-1], atol=1E-9)
    for i in range(actual_width - 2):
        np.testing.assert_allclose(pressure_profile[i], p_expected, atol=1E-9, rtol=1E-3)
