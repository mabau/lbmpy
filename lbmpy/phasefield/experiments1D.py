import numpy as np
import sympy as sp

from pystencils import make_slice
from pystencils.fd import Diff


def plot_status(phase_field_step, from_x=None, to_x=None):
    import lbmpy.plot as plt

    domain_size = phase_field_step.data_handling.shape
    assert len(domain_size) == 2 and domain_size[1] == 1, "Not a 1D scenario"

    dh = phase_field_step.data_handling

    num_phases = phase_field_step.num_order_parameters

    plt.subplot(1, 3, 1)
    plt.title('φ')
    phi_name = phase_field_step.phi_field_name
    for i in range(num_phases):
        plt.plot(dh.gather_array(phi_name, make_slice[from_x:to_x, 0, i]), marker='x', label='φ_%d' % (i,))
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("μ")
    mu_name = phase_field_step.mu_field_name
    for i in range(num_phases):
        plt.plot(dh.gather_array(mu_name, make_slice[from_x:to_x, 0, i]), marker='x', label='μ_%d' % (i,))
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Force and Velocity")
    plt.plot(dh.gather_array(phase_field_step.force_field_name, make_slice[from_x:to_x, 0, 0]), label='F', marker='x')
    plt.plot(dh.gather_array(phase_field_step.vel_field_name, make_slice[from_x:to_x, 0, 0]), label='u', marker='v')
    plt.legend()


def plot_free_energy_bulk_contours(free_energy, order_parameters, phase0=0, phase1=1,
                                   x_range=(-0.2, 1.2), y_range=(-0.2, 1.2), **kwargs):
    import lbmpy.plot as plt

    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    xg, yg = np.meshgrid(x, y)
    substitutions = {op: 0 for i, op in enumerate(order_parameters) if i not in (phase0, phase1)}
    substitutions.update({d: 0 for d in free_energy.atoms(Diff)})  # remove interface components of free energy
    free_energy_lambda = sp.lambdify([order_parameters[phase0], order_parameters[phase1]],
                                     free_energy.subs(substitutions))
    if 'levels' not in kwargs:
        kwargs['levels'] = np.linspace(0, 1, 60)
    plt.contour(x, y, free_energy_lambda(xg, yg), **kwargs)


def init_sharp_interface(pf_step, phase_idx, x1=None, x2=None, inverse=False):
    domain_size = pf_step.data_handling.shape
    if x1 is None:
        x1 = domain_size[0] // 4
    if x2 is None:
        x2 = 3 * x1

    if phase_idx >= pf_step.num_order_parameters:
        return

    for b in pf_step.data_handling.iterate():
        x = b.cell_index_arrays[0]
        mid = np.logical_and(x1 < x, x < x2)

        phi = b[pf_step.phi_field_name]
        val1, val2 = (1, 0) if inverse else (0, 1)

        phi[..., phase_idx].fill(val1)
        phi[mid, phase_idx] = val2

    pf_step.set_pdf_fields_from_macroscopic_values()


def init_tanh(pf_step, phase_idx, x1=None, x2=None, width=1):
    domain_size = pf_step.data_handling.shape
    if x1 is None:
        x1 = domain_size[0] // 4
    if x2 is None:
        x2 = 3 * x1

    if phase_idx >= pf_step.num_order_parameters:
        return

    for b in pf_step.data_handling.iterate():
        x = b.cell_index_arrays[0]
        phi = b[pf_step.phi_field_name]

        phi[..., phase_idx] = (1 + np.tanh((x - x1) / (2 * width))) / 2 + \
                              (1 + np.tanh((-x + x2) / (2 * width))) / 2 - 1

    pf_step.set_pdf_fields_from_macroscopic_values()


def tanh_test(pf_step, phase0, phase1, expected_interface_width=1, time_steps=10000):
    """
    Initializes a sharp interface and checks if tanh-shaped profile is developing

    Args:
        pf_step: phase field scenario / step
        phase0: index of first phase to initialize
        phase1: index of second phase to initialize inversely
        expected_interface_width: interface width parameter alpha that is used in analytical form
        time_steps: number of time steps run before evaluation

    Returns:
        deviation of simulated profile from analytical solution as average(abs(simulation-analytic))
    """
    import lbmpy.plot as plt
    from lbmpy.phasefield.analytical import analytic_interface_profile

    domain_size = pf_step.data_handling.shape
    pf_step.reset()
    pf_step.data_handling.fill(pf_step.phi_field_name, 0)
    init_sharp_interface(pf_step, phase_idx=phase0, inverse=False)
    init_sharp_interface(pf_step, phase_idx=phase1, inverse=True)
    pf_step.set_pdf_fields_from_macroscopic_values()
    pf_step.run(time_steps)

    vis_width = 20
    x = np.arange(vis_width) - (vis_width // 2)
    analytic = np.array([analytic_interface_profile(x_i - 0.5, expected_interface_width) for x_i in x],
                        dtype=np.float64)

    step_location = domain_size[0] // 4
    simulated = pf_step.phi[step_location - vis_width // 2:step_location + vis_width // 2, 0, phase0]
    plt.plot(analytic, label='analytic', marker='o')
    plt.plot(simulated, label='simulated', marker='x')
    plt.legend()

    return np.average(np.abs(simulated - analytic))


def galilean_invariance_test(pf_step, velocity=0.05, rounds=3, phase0=0, phase1=1,
                             expected_interface_width=1, init_time_steps=5000):
    """
    Moves interface at constant speed through periodic domain - check if momentum is conserved

    Args:
        pf_step: phase field scenario / step
        velocity: constant velocity to move interface
        rounds: how many times the interface should travel through the domain
        phase0: index of first phase to initialize
        phase1: index of second phase to initialize inversely
        expected_interface_width: interface width parameter alpha that is used in analytical form
        init_time_steps: before velocity is set, this many time steps are run to let interface settle to tanh shape

    Returns:
        change in velocity
    """
    import lbmpy.plot as plt
    from lbmpy.phasefield.analytical import analytic_interface_profile

    domain_size = pf_step.data_handling.shape
    round_time_steps = int((domain_size[0] + 0.25) / velocity)

    print("Velocity:", velocity, " Time steps for round:", round_time_steps)

    pf_step.reset()
    pf_step.data_handling.fill(pf_step.phi_field_name, 0)
    init_sharp_interface(pf_step, phase_idx=phase0, inverse=False)
    init_sharp_interface(pf_step, phase_idx=phase1, inverse=True)
    pf_step.set_pdf_fields_from_macroscopic_values()

    print("Running", init_time_steps, "initial time steps")
    pf_step.run(init_time_steps)
    pf_step.data_handling.fill(pf_step.vel_field_name, velocity, value_idx=0)
    pf_step.set_pdf_fields_from_macroscopic_values()

    step_location = domain_size[0] // 4
    vis_width = 20

    simulated_profiles = []

    def capture_profile():
        simulated = pf_step.phi[step_location - vis_width // 2:step_location + vis_width // 2, 0, phase0].copy()
        simulated_profiles.append(simulated)

    capture_profile()
    for rt in range(rounds):
        print("Running round %d/%d" % (rt + 1, rounds))
        pf_step.run(round_time_steps)
        capture_profile()

    x = np.arange(vis_width) - (vis_width // 2)
    ref = np.array([analytic_interface_profile(x_i - 0.5, expected_interface_width) for x_i in x], dtype=np.float64)

    plt.plot(x, ref, label='analytic', marker='o')
    for i, profile in enumerate(simulated_profiles):
        plt.plot(x, profile, label="After %d rounds" % (i,))

    plt.legend()

    return np.average(pf_step.velocity[:, 0, 0]) - velocity
