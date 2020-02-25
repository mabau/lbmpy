"""
This test revealed a problem with OpenCL (https://i10git.cs.fau.de/pycodegen/lbmpy/issues/9#note_9521)
"""


import numpy as np
import pytest

import lbmpy
from lbmpy.boundaries import NoSlip
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from lbmpy.session import *
from pystencils.session import *

previous_vmids = {}
previous_endresults = {}


@pytest.mark.parametrize('target', ('cpu', 'gpu', 'opencl'))
def test_poiseuille_channel(target):
    # # Lattice Boltzmann
    #
    # ## Definitions
    if target == 'opencl':
        import pytest
        pytest.importorskip("pyopencl")
        import pystencils.opencl.autoinit
    elif target == 'gpu':
        import pytest
        pytest.importorskip("pycuda")

    # In[2]:

    L = (34, 34)

    lb_stencil = get_stencil("D2Q9")
    eta = 1
    omega = lbmpy.relaxationrates.relaxation_rate_from_lattice_viscosity(eta)

    f_pre = 0.00001

    # ## Data structures

    # In[3]:

    dh = ps.create_data_handling(L, periodicity=(True, False), default_target=target)

    opts = {'cpu_openmp': True,
            'cpu_vectorize_info': None,
            'target': dh.default_target}

    src = dh.add_array('src', values_per_cell=len(lb_stencil))
    dst = dh.add_array_like('dst', 'src')
    ρ = dh.add_array('rho', latex_name='\\rho')
    u = dh.add_array('u', values_per_cell=dh.dim)

    # In[4]:

    collision = create_lb_update_rule(stencil=lb_stencil,
                                      relaxation_rate=omega,
                                      compressible=True,
                                      force_model='guo',
                                      force=sp.Matrix([f_pre, 0]),
                                      kernel_type='collide_only',
                                      optimization={'symbolic_field': src})

    stream = create_stream_pull_with_output_kernel(collision.method, src, dst, {'velocity': u})

    lbbh = LatticeBoltzmannBoundaryHandling(collision.method, dh, src.name, target=dh.default_target)

    stream_kernel = ps.create_kernel(stream, **opts).compile()
    collision_kernel = ps.create_kernel(collision, **opts).compile()

    # ## Set up the simulation

    # In[5]:

    init = macroscopic_values_setter(collision.method, velocity=(0,)*dh.dim,
                                     pdfs=src.center_vector, density=ρ.center)
    init_kernel = ps.create_kernel(init, ghost_layers=0).compile()

    noslip = NoSlip()
    lbbh.set_boundary(noslip, make_slice[:, :4])
    lbbh.set_boundary(noslip, make_slice[:, -4:])

    for bh in lbbh, :
        assert len(bh._boundary_object_to_boundary_info) == 1, "Restart kernel to clear boundaries"

    def init():
        dh.fill(ρ.name, 1)
        dh.fill(u.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(u.name, 0)

        dh.run_kernel(init_kernel)

    # In[6]:

    sync_pdfs = dh.synchronization_function([src.name])

    def time_loop(steps):
        dh.all_to_gpu()
        vmid = np.empty((2, steps//10+1))
        i = -1
        for i in range(steps):
            dh.run_kernel(collision_kernel)
            sync_pdfs()
            lbbh()
            dh.run_kernel(stream_kernel)

            dh.swap(src.name, dst.name)

            if i % 10 == 0:
                if u.name in dh.gpu_arrays:
                    dh.to_cpu(u.name)
                uu = dh.gather_array(u.name)
                uu = uu[L[0]//2-1:L[0]//2+1, L[1]//2-1:L[1]//2+1, 0].mean()
                vmid[:, i//10] = [i, uu]
                if 1/np.sqrt(3) < uu:
                    break
        if dh.is_on_gpu(u.name):
            dh.to_cpu(u.name)

        return vmid[:, :i//10+1]

    # In[7]:

    # def plot():

        # plt.subplot(2, 2, 1)
        # plt.title("$u$")
        # plt.xlabel("$x$")
        # plt.ylabel("$y$")
        # plt.vector_field_magnitude(uu)
        # plt.colorbar()

        # plt.subplot(2, 2, 2)
        # plt.title("$u$")
        # plt.xlabel("$x/2$")
        # plt.ylabel("$y/2$")
        # plt.vector_field(uu, step=2)

        # actualwidth = np.sum(1-np.isnan(uu[0, :, 0]))
        # uref = f_pre*actualwidth**2/(8*(eta))

        # plt.subplot(2, 2, 3)
        # plt.title("flow profile")
        # plt.xlabel("$y$")
        # plt.ylabel(r"$u_x$")
        # plt.plot((uu[L[0]//2-1, :, 0]+uu[L[0]//2, :, 0])/2)

        # plt.subplot(2, 2, 4)
        # plt.title("convergence")
        # plt.xlabel("$t$")
        # plt.plot()

    # ## Run the simulation

    # In[8]:

    init()
    vmid = time_loop(1000)
    # plot()

    uu = dh.gather_array(u.name)

    for target, prev_endresult in previous_endresults.items():
        assert np.allclose(uu[4:-4, 4:-4, 0], prev_endresult[4:-4, 4:-4, 0],
                           atol=1e-5), f'uu does not agree with result from {target}'

    for target, prev_vmid in previous_vmids.items():
        assert np.allclose(vmid, prev_vmid, atol=1e-5), f'vmid does not agree with result from {target}'

    # uref = f_pre*actualwidth**2/(8*(eta))
    # actualwidth = np.sum(1-np.isnan(uu[0, :, 0]))
    # assert np.allclose(vmid[1, -1]/uref, 1, atol=1e-3)
    # print(vmid[1, -1])

    previous_vmids[target] = vmid
    previous_endresults[target] = uu
