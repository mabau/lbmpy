from lbmpy.updatekernels import create_stream_pull_with_output_kernel
from lbmpy import create_lb_update_rule, relaxation_rate_from_lattice_viscosity, ForceModel, Method, LBStencil
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from pystencils.boundaries.boundaryhandling import BoundaryHandling
from pystencils.boundaries.boundaryconditions import Neumann, Dirichlet
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.boundaries import NoSlip

from lbmpy.oldroydb import *
import pytest

# # Lattice Boltzmann with Finite-Volume Oldroyd-B
# # taken from the electronic supplement of https://doi.org/10.1140/epje/s10189-020-00005-6,
# # available at https://doi.org/10.24416/UU01-2AFZSW

pytest.importorskip('scipy.optimize')


def test_oldroydb():
    import scipy.optimize

    # ## Definitions

    L = (34, 34)
    lambda_p = sp.Symbol("lambda_p")
    eta_p = sp.Symbol("eta_p")

    lb_stencil = LBStencil("D2Q9")
    fv_stencil = LBStencil("D2Q9")
    eta = 1 - eta_p
    omega = relaxation_rate_from_lattice_viscosity(eta)

    f_pre = 0.00001

    # ## Data structures

    dh = ps.create_data_handling(L, periodicity=(True, False), default_target=ps.Target.CPU)

    opts = {'cpu_openmp': False,
            'cpu_vectorize_info': None,
            'target': dh.default_target}

    src = dh.add_array('src', values_per_cell=len(lb_stencil), layout='c')
    dst = dh.add_array_like('dst', 'src')
    ρ = dh.add_array('rho', layout='c', latex_name='\\rho')
    u = dh.add_array('u', values_per_cell=dh.dim, layout='c')
    tauface = dh.add_array('tau_face', values_per_cell=(len(fv_stencil) // 2, dh.dim, dh.dim), latex_name='\\tau_f',
                           field_type=ps.FieldType.STAGGERED, layout='c')

    tau = dh.add_array('tau', values_per_cell=(dh.dim, dh.dim), layout='c', latex_name='\\tau')
    tauflux = dh.add_array('j_tau', values_per_cell=(len(fv_stencil) // 2, dh.dim, dh.dim), latex_name='j_\\tau',
                           field_type=ps.FieldType.STAGGERED_FLUX, layout='c')
    F = dh.add_array('F', values_per_cell=dh.dim, layout='c')

    fluxbh = BoundaryHandling(dh, tauflux.name, fv_stencil, name="flux_boundary_handling",
                              openmp=opts['cpu_openmp'], target=dh.default_target)
    ubh = BoundaryHandling(dh, u.name, lb_stencil, name="velocity_boundary_handling",
                           openmp=opts['cpu_openmp'], target=dh.default_target)
    taufacebh = BoundaryHandling(dh, tauface.name, fv_stencil, name="tauface_boundary_handling",
                                 openmp=opts['cpu_openmp'], target=dh.default_target)

    # ## Solver

    collision = create_lb_update_rule(stencil=lb_stencil,
                                      method=Method.TRT,
                                      relaxation_rate=omega,
                                      compressible=True,
                                      force_model=ForceModel.GUO,
                                      force=F.center_vector + sp.Matrix([f_pre, 0]),
                                      kernel_type='collide_only',
                                      optimization={'symbolic_field': src})

    stream = create_stream_pull_with_output_kernel(collision.method, src, dst, {'density': ρ, 'velocity': u})

    lbbh = LatticeBoltzmannBoundaryHandling(collision.method, dh, src.name,
                                            openmp=opts['cpu_openmp'], target=dh.default_target)

    stream_kernel = ps.create_kernel(stream, **opts).compile()
    collision_kernel = ps.create_kernel(collision, **opts).compile()

    ob = OldroydB(dh.dim, u, tau, F, tauflux, tauface, lambda_p, eta_p)
    flux_kernel = ps.create_staggered_kernel(ob.flux(), **opts).compile()
    tauface_kernel = ps.create_staggered_kernel(ob.tauface(), **opts).compile()
    continuity_kernel = ps.create_kernel(ob.continuity(), **opts).compile()
    force_kernel = ps.create_kernel(ob.force(), **opts).compile()

    # ## Set up the simulation

    init = macroscopic_values_setter(collision.method, velocity=(0,) * dh.dim,
                                     pdfs=src.center_vector, density=ρ.center)
    init_kernel = ps.create_kernel(init, ghost_layers=0).compile()

    # no-slip for the fluid, no-flux for the stress
    noslip = NoSlip()
    noflux = Flux(fv_stencil)
    nostressdiff = Flux(fv_stencil, tau.center_vector)

    # put some good values into the boundaries so we can take derivatives
    noforce = Neumann()  # put the same stress into the boundary cells that is in the nearest fluid cell
    noflow = Dirichlet((0,) * dh.dim)  # put zero velocity into the boundary cells

    lbbh.set_boundary(noslip, ps.make_slice[:, :4])
    lbbh.set_boundary(noslip, ps.make_slice[:, -4:])
    fluxbh.set_boundary(noflux, ps.make_slice[:, :4])
    fluxbh.set_boundary(noflux, ps.make_slice[:, -4:])
    ubh.set_boundary(noflow, ps.make_slice[:, :4])
    ubh.set_boundary(noflow, ps.make_slice[:, -4:])
    taufacebh.set_boundary(nostressdiff, ps.make_slice[:, :4])
    taufacebh.set_boundary(nostressdiff, ps.make_slice[:, -4:])

    for bh in lbbh, fluxbh, ubh, taufacebh:
        assert len(bh._boundary_object_to_boundary_info) == 1, "Restart kernel to clear boundaries"

    def init():
        dh.fill(ρ.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(ρ.name, 1)
        dh.fill(u.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(u.name, 0)
        dh.fill(tau.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(tau.name, 0)
        dh.fill(tauflux.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(tauface.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(F.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(F.name, 0)  # needed for LB initialization

        sync_tau()  # force calculation inside the initialization needs neighbor taus
        dh.run_kernel(init_kernel)
        dh.fill(F.name, np.nan)

    sync_pdfs = dh.synchronization_function([src.name])  # needed before stream, but after collision
    sync_u = dh.synchronization_function([u.name])  # needed before continuity, but after stream
    sync_tau = dh.synchronization_function([tau.name])  # needed before flux and tauface, but after continuity

    def time_loop(steps, lambda_p_val, eta_p_val):
        dh.all_to_gpu()
        vmid = np.empty((2, steps // 10 + 1))
        sync_tau()
        sync_u()
        ubh()
        i = -1
        for i in range(steps):
            dh.run_kernel(flux_kernel)
            fluxbh()  # zero the fluxes into/out of boundaries
            dh.run_kernel(continuity_kernel, **{lambda_p.name: lambda_p_val, eta_p.name: eta_p_val})

            sync_tau()
            dh.run_kernel(tauface_kernel)  # needed for force
            taufacebh()
            dh.run_kernel(force_kernel)

            dh.run_kernel(collision_kernel, **{eta_p.name: eta_p_val})
            sync_pdfs()
            lbbh()  # bounce-back populations into boundaries
            dh.run_kernel(stream_kernel)
            sync_u()
            ubh()  # need neighboring us for flux and continuity

            dh.swap(src.name, dst.name)

            if i % 10 == 0:
                if u.name in dh.gpu_arrays:
                    dh.to_cpu(u.name)
                uu = dh.gather_array(u.name)
                uu = uu[L[0] // 2 - 1:L[0] // 2 + 1, L[1] // 2 - 1:L[1] // 2 + 1, 0].mean()
                if np.isnan(uu):
                    raise Exception(f"NaN encountered after {i} steps")
                vmid[:, i // 10] = [i, uu]
        sync_u()
        dh.all_to_cpu()

        return vmid[:, :i // 10 + 1]

    # ## Analytical solution
    # 
    # comes from Waters and King, Unsteady flow of an elastico-viscous liquid, Rheologica Acta 9, 345–355 (1970).

    def N(n):
        return (2 * n - 1) * np.pi

    def Alpha_n(N, El, eta_p):
        return 1 + (1 - eta_p) * El * N * N / 4

    def Beta_n(alpha_n, N, El):
        return np.sqrt(np.abs(alpha_n * alpha_n - El * N * N))

    def Gamma_n(N, El, eta_p):
        return 1 - (1 + eta_p) * El * N * N / 4

    def G(alpha_n, beta_n, gamma_n, flag, T):
        if (flag):
            return ((1.0 - gamma_n / beta_n) * np.exp(-(alpha_n + beta_n) * T / 2) +
                    (1.0 + gamma_n / beta_n) * np.exp((beta_n - alpha_n) * T / 2))
        else:
            return 2 * np.exp(-alpha_n * T / 2) * (np.cos(beta_n * T / 2) + (gamma_n / beta_n) * np.sin(beta_n * T / 2))

    def W(T, El, eta_p):
        W_ = 1.5
        for n in range(1, 1000):
            N_ = N(n)
            alpha_n = Alpha_n(N_, El, eta_p)

            if alpha_n * alpha_n - El * N_ * N_ < 0:
                flag_ = False
            else:
                flag_ = True

            beta_n = Beta_n(alpha_n, N_, El)
            gamma_n = Gamma_n(N_, El, eta_p)
            G_ = G(alpha_n, beta_n, gamma_n, flag_, T)

            W_ -= 24 * (np.sin(N_ / 2) / (N_ * N_ * N_)) * G_

        return W_

    # ## Run the simulation

    lambda_p_val = 3000
    eta_p_val = 0.9

    init()
    vmid = time_loop(lambda_p_val * 4, lambda_p_val, eta_p_val)

    actual_width = sum(dh.gather_array(lbbh.flag_array_name)[L[0] // 2, :] == 1)
    uref = float(f_pre * actual_width ** 2 / (8 * (eta + eta_p)))

    Wi = lambda_p_val * uref / (actual_width / 2)
    Re = uref * (actual_width / 2) / (eta + eta_p)
    El = float(Wi / Re)

    pref = 1 / W(vmid[0, -1] / lambda_p_val, El, eta_p_val)

    El_measured, pref_measured = scipy.optimize.curve_fit(lambda a, b, c: W(a, b, eta_p_val) * c,
                                                          vmid[0, :] / lambda_p_val, vmid[1, :] / vmid[1, -1],
                                                          p0=(El, pref))[0]
    measured_width = np.sqrt(4 * lambda_p_val * float(eta + eta_p) / El_measured)

    print(f"El={El}, El_measured={El_measured}")
    print(f"L={actual_width}, L_measured={measured_width}")

    assert abs(measured_width - actual_width) < 1, "effective channel width differs significantly from defined width"

    an = W(vmid[0, :] / lambda_p_val, El, eta_p_val) * pref
    an_measured = W(vmid[0, :] / lambda_p_val, El_measured, eta_p_val) * pref_measured

    diff = abs(vmid[1, :] / vmid[1, -1] - an_measured) / an_measured
    assert diff[lambda_p_val // 5:].max() < 0.03, "maximum velocity deviation is too large"

#    from pystencils import plot as plt
#
#     plt.xlabel("$t$")
#     plt.ylabel(r"$u_{max}/u_{max}^{Newtonian}$")
#     plt.plot(vmid[0,:], vmid[1,:]/vmid[1,-1] if vmid[1,-1] != 0 else 0, label='FVM')
#     plt.plot(vmid[0,:], np.ones_like(vmid[0,:]), 'k--', label='Newtonian')
#
#     plt.plot(vmid[0,:], an, label="analytic")
#     plt.plot(vmid[0,:], an_measured, label="analytic, fit width")
#     plt.legend()
#
#     if eta_p_val == 0.1:
#         plt.ylim(0.9, 1.15)
#     elif lambda_p_val == 9000:
#         plt.ylim(0.8, 1.5)
#     elif eta_p_val == 0.3:
#         plt.ylim(0.8, 1.4)
#     plt.show()
