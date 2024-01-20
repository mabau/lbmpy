import numpy as np

from lbmpy.advanced_streaming import LBMPeriodicityHandling
from lbmpy.analytical_solutions import poiseuille_flow
from lbmpy.boundaries import NoSlip
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.creationfunctions import create_lb_update_rule, LBMConfig, LBMOptimisation
from lbmpy.enums import Method, ForceModel
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from lbmpy.stencils import LBStencil
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity

from pystencils import CreateKernelConfig
import pystencils as ps


def poiseuille_channel(target, stencil_name, **kwargs):
    # physical parameters
    rho_0 = 1.2  # density
    eta = 0.2  # kinematic viscosity
    width = 41  # of box
    actual_width = width - 2  # subtract boundary layer from box width
    ext_force_density = 0.2 / actual_width ** 2  # scale by width to keep stable

    # LB parameters
    lb_stencil = LBStencil(stencil_name)

    if lb_stencil.D == 2:
        L = (4, width)
    elif lb_stencil.D == 3:
        L = (4, width, 4)
    else:
        raise Exception()
    periodicity = [True, False] + [True] * (lb_stencil.D - 2)

    omega = relaxation_rate_from_lattice_viscosity(eta)

    # ## Data structures
    dh = ps.create_data_handling(L, periodicity=periodicity, default_target=target)

    src = dh.add_array('src', values_per_cell=len(lb_stencil))
    dh.fill(src.name, 0.0, ghost_layers=True)
    dst = dh.add_array_like('dst', 'src')
    dh.fill(dst.name, 0.0, ghost_layers=True)
    u = dh.add_array('u', values_per_cell=dh.dim)
    dh.fill(u.name, 0.0, ghost_layers=True)

    # LB Setup
    lbm_config = LBMConfig(stencil=lb_stencil, relaxation_rate=omega, method=Method.TRT,
                           compressible=True,
                           force_model=ForceModel.GUO,
                           force=tuple([ext_force_density] + [0] * (lb_stencil.D - 1)),
                           output={'velocity': u}, **kwargs)

    lbm_opt = LBMOptimisation(symbolic_field=src, symbolic_temporary_field=dst)
    update = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)
    method = update.method

    config = CreateKernelConfig(cpu_openmp=False, target=dh.default_target)
    stream_collide = ps.create_kernel(update, config=config).compile()

    lbbh = LatticeBoltzmannBoundaryHandling(method, dh, src.name, target=dh.default_target)

    # ## Set up the simulation
    init = macroscopic_values_setter(method, velocity=(0,) * dh.dim,
                                     pdfs=src.center_vector, density=rho_0)
    init_kernel = ps.create_kernel(init).compile()
    dh.run_kernel(init_kernel)

    noslip = NoSlip()
    wall_thickness = 2
    if lb_stencil.D == 2:
        lbbh.set_boundary(noslip, ps.make_slice[:, :wall_thickness])
        lbbh.set_boundary(noslip, ps.make_slice[:, -wall_thickness:])
    elif lb_stencil.D == 3:
        lbbh.set_boundary(noslip, ps.make_slice[:, :wall_thickness, :])
        lbbh.set_boundary(noslip, ps.make_slice[:, -wall_thickness:, :])
    else:
        raise Exception()

    for bh in lbbh,:
        assert len(bh._boundary_object_to_boundary_info) == 1, "Restart kernel to clear boundaries"

    sync_pdfs = LBMPeriodicityHandling(lb_stencil, dh, src.name)

    # Time loop
    def time_loop(steps):
        dh.all_to_gpu()
        last_max_vel = -1
        for i in range(steps):
            sync_pdfs()
            lbbh()
            dh.run_kernel(stream_collide)
            dh.swap(src.name, dst.name)

            # Consider early termination
            if i % 100 == 0:
                if u.name in dh.gpu_arrays:
                    dh.to_cpu(u.name)
                uu = dh.gather_array(u.name)
                # average periodic directions
                if lb_stencil.D == 3:  # dont' swap order
                    uu = np.average(uu, axis=2)
                uu = np.average(uu, axis=0)

                max_vel = np.nanmax(uu)
                if np.abs(max_vel / last_max_vel - 1) < 5E-6:
                    break
                last_max_vel = max_vel

        # cut off wall regions
        uu = uu[wall_thickness:-wall_thickness]

        # correct for f/2 term
        uu -= np.array([ext_force_density / 2 / rho_0] + [0] * (lb_stencil.D - 1))

        return uu

    # Simulation
    profile = time_loop(10000)

    # compare against analytical solution
    # The profile is of shape (n, 3). Force is in x-direction
    y = np.arange(len(profile[:, 0]))
    mid = (y[-1] - y[0]) / 2  # Mid point of channel

    expected = poiseuille_flow((y - mid), actual_width, ext_force_density, rho_0 * eta)

    np.testing.assert_allclose(profile[:, 0], expected, rtol=0.006)

    # Test zero vel in other directions
    np.testing.assert_allclose(profile[:, 1:], np.zeros_like(profile[:, 1:]), atol=1E-9)
