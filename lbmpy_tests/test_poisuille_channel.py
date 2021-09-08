"""
Test Poiseuille flow against analytical solution
"""


import numpy as np
import pytest
import sympy as sp

import lbmpy
from lbmpy.boundaries import NoSlip
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.creationfunctions import create_lb_update_rule, create_stream_pull_with_output_kernel
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from lbmpy.stencils import get_stencil

import pystencils as ps


def poiseuille_flow(z, H, ext_force_density, dyn_visc):
    """
    Analytical solution for plane Poiseuille flow.

    Parameters
    ----------
    z : :obj:`float`
        Distance to the mid plane of the channel.
    H : :obj:`float`
        Distance between the boundaries.
    ext_force_density : :obj:`float`
        Force density on the fluid normal to the boundaries.
    dyn_visc : :obj:`float`
        Dynamic viscosity of the LB fluid.

    """
    return ext_force_density * 1. / (2 * dyn_visc) * (H**2.0 / 4.0 - z**2.0)


rho_0 = 1.2  # density
eta = 0.2  # kinematic viscosity
width = 41  # of box
actual_width = width - 2  # subtract boundary layer from box width
ext_force_density = 0.2 / actual_width**2  # scale by width to keep stable


@pytest.mark.parametrize('target', ('cpu', 'gpu', 'opencl'))
@pytest.mark.parametrize('stencil_name', ("D2Q9", "D3Q19",))
def test_poiseuille_channel(target, stencil_name):
    # OpenCL and Cuda
    if target == 'opencl':
        import pytest
        pytest.importorskip("pyopencl")
        import pystencils.opencl.autoinit
    elif target == 'gpu':
        import pytest
        pytest.importorskip("pycuda")

    # LB parameters
    lb_stencil = get_stencil(stencil_name)
    dim = len(lb_stencil[0])

    if dim == 2:
        L = [4, width]
    elif dim == 3:
        L = [4, width, 4]
    else:
        raise Exception()
    periodicity = [True, False] + [True] * (dim - 2)

    omega = lbmpy.relaxationrates.relaxation_rate_from_lattice_viscosity(eta)

    # ## Data structures
    dh = ps.create_data_handling(L, periodicity=periodicity, default_target=target)

    src = dh.add_array('src', values_per_cell=len(lb_stencil))
    dst = dh.add_array_like('dst', 'src')
    ρ = dh.add_array('rho', latex_name='\\rho')
    u = dh.add_array('u', values_per_cell=dh.dim)

    # LB Setup
    collision = create_lb_update_rule(stencil=lb_stencil,
                                      relaxation_rate=omega,
                                      method="trt",
                                      compressible=True,
                                      force_model="guo",
                                      force=sp.Matrix([ext_force_density] + [0] * (dim - 1)),
                                      kernel_type='collide_only',
                                      optimization={'symbolic_field': src})

    stream = create_stream_pull_with_output_kernel(collision.method, src, dst, {'velocity': u})

    opts = {'cpu_openmp': False,
            'cpu_vectorize_info': None,
            'target': dh.default_target}

    stream_kernel = ps.create_kernel(stream, **opts).compile()
    collision_kernel = ps.create_kernel(collision, **opts).compile()

    # Boundaries
    lbbh = LatticeBoltzmannBoundaryHandling(collision.method, dh, src.name, target=dh.default_target)

    # ## Set up the simulation

    init = macroscopic_values_setter(collision.method, velocity=(0,) * dh.dim,
                                     pdfs=src.center_vector, density=ρ.center)
    init_kernel = ps.create_kernel(init, ghost_layers=0).compile()

    noslip = NoSlip()
    wall_thickness = 2
    if dim == 2:
        lbbh.set_boundary(noslip, ps.make_slice[:, :wall_thickness])
        lbbh.set_boundary(noslip, ps.make_slice[:, -wall_thickness:])
    elif dim == 3:
        lbbh.set_boundary(noslip, ps.make_slice[:, :wall_thickness, :])
        lbbh.set_boundary(noslip, ps.make_slice[:, -wall_thickness:, :])
    else:
        raise Exception()

    for bh in lbbh, :
        assert len(bh._boundary_object_to_boundary_info) == 1, "Restart kernel to clear boundaries"

    def init():
        dh.fill(ρ.name, rho_0)
        dh.fill(u.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(u.name, 0)

        dh.run_kernel(init_kernel)

    # In[6]:

    sync_pdfs = dh.synchronization_function([src.name])

    # Time loop
    def time_loop(steps):
        dh.all_to_gpu()
        i = -1
        last_max_vel = -1
        for i in range(steps):
            dh.run_kernel(collision_kernel)
            sync_pdfs()
            lbbh()
            dh.run_kernel(stream_kernel)

            dh.swap(src.name, dst.name)

            # Consider early termination
            if i % 100 == 0:
                if u.name in dh.gpu_arrays:
                    dh.to_cpu(u.name)
                uu = dh.gather_array(u.name)
                # average periodic directions
                if dim == 3:  # dont' swap order
                    uu = np.average(uu, axis=2)
                uu = np.average(uu, axis=0)

                max_vel = np.nanmax(uu)
                if np.abs(max_vel / last_max_vel - 1) < 5E-6:
                    break
                last_max_vel = max_vel

        # cut off wall regions
        uu = uu[wall_thickness:-wall_thickness]

        # correct for f/2 term
        uu -= np.array([ext_force_density / 2 / rho_0] + [0] * (dim - 1))

        return uu

    init()
    # Simulation
    profile = time_loop(5000)

    # compare against analytical solution
    # The profile is of shape (n,3). Force is in x-direction
    y = np.arange(len(profile[:, 0]))
    mid = (y[-1] - y[0]) / 2  # Mid point of channel

    expected = poiseuille_flow((y - mid), actual_width, ext_force_density, rho_0 * eta)

    np.testing.assert_allclose(profile[:, 0], expected, rtol=0.006)

    # Test zero vel in other directions
    np.testing.assert_allclose(profile[:, 1:], np.zeros_like(profile[:, 1:]), atol=1E-9)
