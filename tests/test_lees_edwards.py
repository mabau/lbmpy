from functools import partial

import pystencils as ps
from pystencils.astnodes import LoopOverCoordinate
from pystencils.slicing import get_periodic_boundary_functor

from lbmpy.creationfunctions import create_lb_update_rule, LBMConfig, LBMOptimisation
from lbmpy.enums import Stencil, ForceModel
from lbmpy.stencils import LBStencil
from lbmpy.updatekernels import create_stream_pull_with_output_kernel
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from lbmpy.relaxationrates import lattice_viscosity_from_relaxation_rate

import sympy as sp
import numpy as np


def get_le_boundary_functor(neighbor_stencil, shear_offset, ghost_layers=1, thickness=None, n=64):
    functor_2 = get_periodic_boundary_functor(neighbor_stencil, ghost_layers, thickness)

    def functor(pdfs, **_):

        functor_2(pdfs)
        weight = np.fmod(shear_offset[0] + n, 1.)

        # First, we interpolate the upper pdfs
        for i in range(len(pdfs[:, ghost_layers, :])):
            ind1 = int(np.floor(i - shear_offset[0]) % n)
            ind2 = int(np.ceil(i - shear_offset[0]) % n)

            res = (1 - weight) * pdfs[ind1, ghost_layers, :] + weight * pdfs[ind2, ghost_layers, :]
            pdfs[i, -ghost_layers, :] = res

        # Second, we interpolate the lower pdfs
        for i in range(len(pdfs[:, -ghost_layers, :])):
            ind1 = int(np.floor(i + shear_offset[0]) % n)
            ind2 = int(np.ceil(i + shear_offset[0]) % n)

            res = (1 - weight) * pdfs[ind1, -ghost_layers - 1, :] + weight * pdfs[ind2, -ghost_layers - 1, :]
            pdfs[i, ghost_layers - 1, :] = res

    return functor


def get_solution_navier_stokes(x, t, viscosity, velocity=1.0, height=1.0, max_iterations=10):
    w = x / height - 0.5
    for k in np.arange(1, max_iterations + 1):
        w += 1.0 / (np.pi * k) * np.exp(-4 * np.pi ** 2 * viscosity * k ** 2 / height ** 2 * t) * np.sin(
            2 * np.pi / height * k * x)
    return velocity * w


def test_lees_edwards():

    domain_size = (64, 64)
    omega = 1.0  # relaxation rate of first component
    shear_velocity = 0.1  # shear velocity
    shear_dir = 0  # direction of shear flow
    shear_dir_normal = 1  # direction normal to shear plane, for interpolation

    stencil = LBStencil(Stencil.D2Q9)

    dh = ps.create_data_handling(domain_size, periodicity=True, default_target=ps.Target.CPU)

    src = dh.add_array('src', values_per_cell=stencil.Q)
    dh.fill('src', 1.0, ghost_layers=True)

    dst = dh.add_array_like('dst', 'src')
    dh.fill('dst', 0.0, ghost_layers=True)

    force = dh.add_array('force', values_per_cell=stencil.D)
    dh.fill('force', 0.0, ghost_layers=True)

    rho = dh.add_array('rho', values_per_cell=1)
    dh.fill('rho', 1.0, ghost_layers=True)
    u = dh.add_array('u', values_per_cell=stencil.D)
    dh.fill('u', 0.0, ghost_layers=True)

    counters = [LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(stencil.D)]
    points_up = sp.Symbol('points_up')
    points_down = sp.Symbol('points_down')

    u_p = sp.Piecewise((1, sp.And(counters[1] <= 1, points_down)),
                       (-1, sp.And(counters[1] >= src.shape[1] - 2, points_up)), (0, True)) * shear_velocity

    lbm_config = LBMConfig(stencil=stencil, relaxation_rate=omega, compressible=True,
                           velocity_input=u.center_vector + sp.Matrix([u_p, 0]), density_input=rho,
                           force_model=ForceModel.LUO, force=force.center_vector,
                           kernel_type='collide_only')
    lbm_opt = LBMOptimisation(symbolic_field=src)
    collision = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    to_insert = [s.lhs for s in collision.subexpressions
                 if collision.method.first_order_equilibrium_moment_symbols[shear_dir]
                 in s.free_symbols]
    for s in to_insert:
        collision = collision.new_with_inserted_subexpression(s)
    ma = []
    for a, c in zip(collision.main_assignments, collision.method.stencil):
        if c[shear_dir_normal] == -1:
            b = (True, False)
        elif c[shear_dir_normal] == 1:
            b = (False, True)
        else:
            b = (False, False)
        a = ps.Assignment(a.lhs, a.rhs.replace(points_down, b[0]))
        a = ps.Assignment(a.lhs, a.rhs.replace(points_up, b[1]))
        ma.append(a)
    collision.main_assignments = ma

    stream = create_stream_pull_with_output_kernel(collision.method, src, dst,
                                                   {'density': rho, 'velocity': u})

    config = ps.CreateKernelConfig(target=dh.default_target)
    stream_kernel = ps.create_kernel(stream, config=config).compile()
    collision_kernel = ps.create_kernel(collision, config=config).compile()

    init = macroscopic_values_setter(collision.method, velocity=(0, 0),
                                     pdfs=src.center_vector, density=rho.center)
    init_kernel = ps.create_kernel(init, ghost_layers=0).compile()

    offset = [0.0]

    sync_pdfs = dh.synchronization_function([src.name],
                                            functor=partial(get_le_boundary_functor, shear_offset=offset))

    dh.run_kernel(init_kernel)

    time = 500

    dh.all_to_gpu()
    for i in range(time):
        dh.run_kernel(collision_kernel)

        sync_pdfs()
        dh.run_kernel(stream_kernel)

        dh.swap(src.name, dst.name)
        offset[0] += shear_velocity
    dh.all_to_cpu()

    nu = lattice_viscosity_from_relaxation_rate(omega)
    h = domain_size[0]
    k_max = 100

    analytical_solution = get_solution_navier_stokes(np.linspace(0.5, h - 0.5, h), time, nu, shear_velocity, h, k_max)
    np.testing.assert_array_almost_equal(analytical_solution, dh.gather_array(u.name)[0, :, 0], decimal=5)

    dh.fill(rho.name, 1.0, ghost_layers=True)
    dh.run_kernel(init_kernel)
    dh.fill(u.name, 0.0, ghost_layers=True)
    dh.fill('force', 0.0, ghost_layers=True)
    dh.cpu_arrays[force.name][64 // 3, 1, :] = [1e-2, -1e-1]

    offset[0] = 0

    time = 20

    dh.all_to_gpu()
    for i in range(time):
        dh.run_kernel(collision_kernel)

        sync_pdfs()
        dh.run_kernel(stream_kernel)

        dh.swap(src.name, dst.name)
    dh.all_to_cpu()

    vel_unshifted = np.array(dh.gather_array(u.name)[:, -3:-1, :])

    dh.fill(rho.name, 1.0, ghost_layers=True)
    dh.run_kernel(init_kernel)
    dh.fill(u.name, 0.0, ghost_layers=True)
    dh.fill('force', 0.0, ghost_layers=True)
    dh.cpu_arrays[force.name][64 // 3, 1, :] = [1e-2, -1e-1]

    offset[0] = 10

    time = 20

    dh.all_to_gpu()
    for i in range(time):
        dh.run_kernel(collision_kernel)

        sync_pdfs()
        dh.run_kernel(stream_kernel)

        dh.swap(src.name, dst.name)
    dh.all_to_cpu()

    vel_shifted = np.array(dh.gather_array(u.name)[:, -3:-1, :])

    vel_rolled = np.roll(vel_shifted, -offset[0], axis=0)

    np.testing.assert_array_almost_equal(vel_unshifted, vel_rolled)
