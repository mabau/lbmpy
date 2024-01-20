import numpy as np
import pytest

from pystencils import Assignment, create_kernel, create_data_handling

from lbmpy.stencils import LBStencil, Stencil

from lbmpy.phasefield_allen_cahn.analytical import analytical_solution_microchannel
from lbmpy.phasefield_allen_cahn.numerical_solver import get_runge_kutta_update_assignments


@pytest.mark.parametrize('solver', ["RK2", "RK4"])
def test_runge_kutta_solver(solver):
    stencil = LBStencil(Stencil.D2Q9)

    L0 = 25
    domain_size = (2 * L0, L0)

    # time step
    timesteps = 2000

    rho_H = 1.0
    rho_L = 1.0

    mu_L = 0.25

    W = 4

    T_h = 20
    T_c = 10
    T_0 = 4

    sigma_T = -5e-4

    cp_h = 1
    cp_l = 1
    k_h = 0.2
    k_l = 0.2

    # create a datahandling object
    dh = create_data_handling(domain_size, periodicity=(True, False))

    u = dh.add_array("u", values_per_cell=dh.dim)
    dh.fill("u", 0.0, ghost_layers=True)

    C = dh.add_array("C", values_per_cell=1)
    dh.fill("C", 0.0, ghost_layers=True)

    temperature = dh.add_array("temperature", values_per_cell=1)
    dh.fill("temperature", T_c, ghost_layers=True)

    RK1 = dh.add_array("RK1", values_per_cell=1)
    dh.fill("RK1", 0.0, ghost_layers=True)

    rk_fields = [RK1, ]
    init_RK = [Assignment(RK1.center, temperature.center), ]

    if solver == "RK4":
        RK2 = dh.add_array("RK2", values_per_cell=1)
        dh.fill("RK2", 0.0, ghost_layers=True)

        RK3 = dh.add_array("RK3", values_per_cell=1)
        dh.fill("RK3", 0.0, ghost_layers=True)

        rk_fields += [RK2, RK3]
        init_RK += [Assignment(RK2.center, temperature.center),
                    Assignment(RK3.center, temperature.center)]

    rho = rho_L + C.center * (rho_H - rho_L)

    for block in dh.iterate(ghost_layers=True, inner_ghost_layers=False):
        x = np.zeros_like(block.midpoint_arrays[0])
        x[:, :] = block.midpoint_arrays[0]

        normalised_x = np.zeros_like(x[:, 0])
        normalised_x[:] = x[:, 0] - L0
        omega = np.pi / L0
        # bottom wall
        block["temperature"][:, 0] = T_h + T_0 * np.cos(omega * normalised_x)
        # top wall
        block["temperature"][:, -1] = T_c

        y = np.zeros_like(block.midpoint_arrays[1])
        y[:, :] = block.midpoint_arrays[1] + (domain_size[1] // 2)

        block["C"][:, :] = 0.5 + 0.5 * np.tanh((y - domain_size[1]) / (W / 2))

    a = get_runge_kutta_update_assignments(stencil, C, temperature, u, rk_fields,
                                           k_h, k_l, cp_h, cp_l, rho)

    init_RK_kernel = create_kernel(init_RK, target=dh.default_target).compile()

    temperature_update_kernel = []
    for ac in a:
        temperature_update_kernel.append(create_kernel(ac, target=dh.default_target).compile())

    periodic_BC_T = dh.synchronization_function(temperature.name)

    x, y, u_x, u_y, t_a = analytical_solution_microchannel(L0, domain_size[0], domain_size[1],
                                                           k_h, k_l,
                                                           T_h, T_c, T_0,
                                                           sigma_T, mu_L)

    for i in range(0, timesteps + 1):
        dh.run_kernel(init_RK_kernel)
        for kernel in temperature_update_kernel:
            dh.run_kernel(kernel)
        periodic_BC_T()

    num = 0.0
    den = 0.0
    T = dh.gather_array(temperature.name, ghost_layers=False)
    for ix in range(domain_size[0]):
        for iy in range(domain_size[1]):
            num += (T[ix, iy] - t_a.T[ix, iy]) ** 2
            den += (t_a.T[ix, iy]) ** 2

    error = np.sqrt(num / den)

    assert error < 0.02
