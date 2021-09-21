"""
Testcase as in
Boesch, Chikatamarla, Karlin: Entropic multirelaxation lattice Boltzmann models for turbulent flows (2015)
"""
import numpy as np
import sympy as sp

import pystencils as ps
from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.relaxationrates import (
    relaxation_rate_from_lattice_viscosity, relaxation_rate_from_magic_number)
from pystencils.runhelper import ParameterStudy

# --------------------------------------------- Setup ------------------------------------------------------------------


def set_initial_velocity(lb_step: LatticeBoltzmannStep, u_0: float) -> None:
    """Initializes velocity field of a fully periodic scenario with eddies.

    Args:
        lb_step: fully periodic 3D scenario
        u_0: maximum velocity
    """
    from numpy import cos, sin

    assert lb_step.dim == 3, "Works only for 3D scenarios"
    assert tuple(lb_step.data_handling.periodicity) == (True, True, True), "Scenario has to be fully periodic"

    for b in lb_step.data_handling.iterate(ghost_layers=False, inner_ghost_layers=False):
        velocity = b[lb_step.velocity_data_name]
        coordinates = b.midpoint_arrays
        x, y, z = [c / s * 2 * np.pi for c, s in zip(coordinates, lb_step.data_handling.shape)]

        velocity[..., 0] = u_0 * sin(x) * (cos(3 * y) * cos(z) - cos(y) * cos(3 * z))
        velocity[..., 1] = u_0 * sin(y) * (cos(3 * z) * cos(x) - cos(z) * cos(3 * x))
        velocity[..., 2] = u_0 * sin(z) * (cos(3 * x) * cos(y) - cos(x) * cos(3 * y))

    lb_step.set_pdf_fields_from_macroscopic_values()


def relaxation_rate_from_reynolds_number(re, u_0, domain_size):
    nu = u_0 * domain_size / re
    return relaxation_rate_from_lattice_viscosity(nu)


def time_step_to_normalized_time(time_step, domain_size, u_0):
    return time_step / (domain_size / u_0)


def normalized_time_to_time_step(normalized_time, domain_size, u_0):
    return int(normalized_time * domain_size / u_0)

# --------------------------------------------- Analysis ---------------------------------------------------------------


def energy_density_spectrum(velocity):
    """Computes energy density for different wave lengths.

    Fourier transformation gives again a 3D field, which is then summed up radially.

    Args:
        velocity: numpy array indexed as [x, y, z, velocity_component]

    Returns:
        1D array where entries correspond to energy at different wave lengths
    """
    def radial_profile(data):
        """Sums up given multidimensional field at constant r (integrating over φ and θ) in spherical coordinates."""
        z, y, x = np.indices(data.shape)
        center = tuple(c // 2 for c in data.shape)
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
        np.round(r, out=r)
        r = r.astype(np.int)
        binned_data = np.bincount(r.ravel(), data.ravel())
        normalization = np.bincount(r.ravel())
        return binned_data / normalization

    kinetic_energy = np.sum(velocity * velocity, axis=3) / 2
    energy_fft_result = np.fft.fftn(kinetic_energy)
    energy_fft_result = np.abs(np.fft.fftshift(energy_fft_result))
    # cut off the zero freq energy
    return radial_profile(energy_fft_result)[1:]


def curl(vector_field):
    """Computes curl (rotation) of a 2D or 3D vector field."""
    dim = len(vector_field.shape) - 1

    def grad(diff_coord, idx):
        return np.gradient(vector_field[..., idx], axis=diff_coord)

    if dim == 2:
        return grad(0, 1) - grad(1, 0)
    elif dim == 3:
        result = np.zeros_like(vector_field)
        result[..., 0] = grad(1, 2) - grad(2, 1)
        result[..., 1] = grad(2, 0) - grad(0, 2)
        result[..., 2] = grad(0, 1) - grad(1, 0)
        return result
    else:
        raise NotImplementedError("curl only implemented for 2D and 3D")


def mean_kinetic_energy(velocity_arr):
    """Computes average kinetic energy in the given 3D velocity array as :math:`1/2 * v^2`. """
    energy_density = np.sum(velocity_arr * velocity_arr, axis=3)
    return 0.5 * np.mean(energy_density)


def mean_enstrophy(velocity_arr):
    """Computes average enstrophy of given 3D velocity array.

    Enstrophy is computed as spatial average of :math:`1/2 (∇ x u)^2`
    """
    w = curl(velocity_arr)
    w_dot_w = np.sum(w*w, axis=3)
    return 0.5 * np.mean(w_dot_w)


def parallel_mean(lb_step, velocity_post_processor, all_reduce=True):
    num_local_blocks = 0
    for b in lb_step.data_handling.iterate(ghost_layers=False, inner_ghost_layers=False):
        num_local_blocks += 1
        velocity = b[lb_step.velocity_data_name]
        local_result = velocity_post_processor(velocity)
    assert num_local_blocks == 1
    reduce_result = lb_step.data_handling.reduce_float_sequence([local_result, 1.0],
                                                                operation='sum', all_reduce=all_reduce)
    if reduce_result is not None:
        return reduce_result[0] / reduce_result[1]
    else:
        return None


def plot_energy_spectrum(velocity_arr):
    import matplotlib.pyplot as plt
    spectrum = energy_density_spectrum(velocity_arr)
    plt.loglog(spectrum)
    plt.title("Energy Spectrum")
    plt.show()


# --------------------------------------------- Main -------------------------------------------------------------------

def run(re=6000, eval_interval=0.05, total_time=3.0, domain_size=100, u_0=0.05,
        initialization_relaxation_rate=None, vtk_output=False, parallel=False, **kwargs):
    """Runs the kida vortex simulation.

    Args:
        re: Reynolds number
        eval_interval: interval in non-dimensional time to evaluate flow properties
        total_time: non-dimensional time of complete simulation
        domain_size: integer (not tuple) since domain is cubic
        u_0: maximum lattice velocity
        initialization_relaxation_rate: if not None, an advanced initialization scheme is run to initialize higher
                                        order moments correctly
        vtk_output: if vtk files are written out
        parallel: MPI parallelization with walberla
        **kwargs: passed to LbStep

    Returns:
        dictionary with simulation results
    """
    domain_shape = (domain_size, domain_size, domain_size)
    relaxation_rate = relaxation_rate_from_reynolds_number(re, u_0, domain_size)
    dh = ps.create_data_handling(domain_shape, periodicity=True, parallel=parallel)
    rr_subs = {'viscosity': relaxation_rate,
               'trt_magic': relaxation_rate_from_magic_number(relaxation_rate),
               'free': sp.Symbol("rr_f")}

    if 'relaxation_rates' in kwargs:
        kwargs['relaxation_rates'] = [rr_subs[r] if isinstance(r, str) else r for r in kwargs['relaxation_rates']]
    else:
        kwargs['relaxation_rates'] = [relaxation_rate]

    dh.log_on_root("Running kida vortex scenario of size {} with {}".format(domain_size, kwargs))
    dh.log_on_root("Compiling method")

    lb_step = LatticeBoltzmannStep(data_handling=dh, name="kida_vortex", **kwargs)

    set_initial_velocity(lb_step, u_0)
    residuum, init_steps = np.nan, 0
    if initialization_relaxation_rate is not None:
        dh.log_on_root("Running iterative initialization", level='PROGRESS')
        residuum, init_steps = lb_step.run_iterative_initialization(initialization_relaxation_rate,
                                                                    convergence_threshold=1e-12, max_steps=100000,
                                                                    check_residuum_after=2 * domain_size)
        dh.log_on_root("Iterative initialization finished after {} steps at residuum {}".format(init_steps, residuum))

    total_time_steps = normalized_time_to_time_step(total_time, domain_size, u_0)
    eval_time_steps = normalized_time_to_time_step(eval_interval, domain_size, u_0)

    initial_energy = parallel_mean(lb_step, mean_kinetic_energy, all_reduce=False)
    times = []
    energy_list = []
    enstrophy_list = []
    mlups_list = []
    energy_spectrum_arr = None

    while lb_step.time_steps_run < total_time_steps:
        mlups = lb_step.benchmark_run(eval_time_steps, number_of_cells=domain_size**3)
        if vtk_output:
            lb_step.write_vtk()

        current_time = time_step_to_normalized_time(lb_step.time_steps_run, domain_size, u_0)
        current_kinetic_energy = parallel_mean(lb_step, mean_kinetic_energy)
        current_enstrophy = parallel_mean(lb_step, mean_enstrophy)

        is_stable = np.isfinite(lb_step.data_handling.max(lb_step.velocity_data_name)) and current_enstrophy < 1e4
        if not is_stable:
            dh.log_on_root("Simulation got unstable - stopping", level='WARNING')
            break

        if current_time >= 0.5 and energy_spectrum_arr is None and domain_size <= 600:
            dh.log_on_root("Calculating energy spectrum")
            gathered_velocity = lb_step.velocity[:, :, :, :]

            if gathered_velocity is not None:
                energy_spectrum_arr = energy_density_spectrum(gathered_velocity)
            else:
                energy_spectrum_arr = False

        if dh.is_root:
            current_kinetic_energy /= initial_energy
            current_enstrophy *= domain_size ** 2

            times.append(current_time)
            energy_list.append(current_kinetic_energy)
            enstrophy_list.append(current_enstrophy)
            mlups_list.append(mlups)

            dh.log_on_root("Progress: {current_time:.02f} / {total_time} at {mlups:.01f} MLUPS\t"
                           "Enstrophy {current_enstrophy:.04f}\t"
                           "KinEnergy {current_kinetic_energy:.06f}".format(**locals()))

    if dh.is_root:
        return {
            'initialization_residuum': residuum,
            'initialization_steps': init_steps,
            'time': times,
            'kinetic_energy': energy_list,
            'enstrophy': enstrophy_list,
            'mlups': np.average(mlups_list),
            'energy_spectrum': list(energy_spectrum_arr),
            'stable': bool(np.isfinite(lb_step.data_handling.max(lb_step.velocity_data_name)))
        }
    else:
        return None


def create_full_parameter_study(gpu=False):
    """Creates a parameter study that can run the Kida vortex flow with entropic, KBC, Smagorinsky and MRT methods."""
    opt_cpu = {'target': ps.Target.CPU, 'openmp': 4}
    opt_gpu = {'target': ps.Target.GPU}

    mrt_one = [{'method': 'mrt3', 'relaxation_rates': ['viscosity', 1, 1], 'stencil': stencil}
               for stencil in ('D3Q19', 'D3Q27')]
    smagorinsky_srt = [{'method': 'srt', 'smagorinsky': cs, 'stencil': stencil, 'compressible': compressible}
                       for cs in (0.8, 0.1, 0.12, 0.16, 2.0)
                       for stencil in ('D3Q19', 'D3Q27')
                       for compressible in (True, False)]
    smagorinsky_trt = [{'method': 'trt', 'smagorinsky': cs, 'stencil': stencil, 'compressible': compressible,
                        'relaxation_rates': ['viscosity', 'trt_magic']}
                       for cs in (0.12, 0.16, 2.0)
                       for stencil in ('D3Q19', 'D3Q27')
                       for compressible in (True, False)]
    entropic_kbc = [{'method': 'trt-kbc-n{}'.format(kbc_nr), 'entropic': True, 'compressible': True}
                    for kbc_nr in (1, 2, 3, 4)]
    entropic_kbc_d3q19 = [{'method': 'mrt3', 'entropic': True, 'compressible': True,
                           'relaxation_rates': ['viscosity', 'free', 'free'], 'stencil': stencil}
                          for stencil in ('D3Q19', 'D3Q27')]
    entropic_pure = [{'method': 'entropic-srt',  'compressible': True, 'stencil': stencil}
                     for stencil in ('D3Q19', 'D3Q27')]

    all_scenarios = mrt_one + smagorinsky_trt + entropic_kbc_d3q19 + smagorinsky_srt + entropic_kbc + entropic_pure
    study = ParameterStudy(run)
    domain_sizes = (50, 100, 200)
    for domain_size in domain_sizes:
        for re in (6000,):
            for scenario in all_scenarios:
                scenario = scenario.copy()
                scenario['re'] = re
                scenario['optimization'] = opt_gpu if gpu else opt_cpu
                scenario['domain_size'] = domain_size
                study.add_run(scenario, weight=(domain_size / min(domain_sizes)) ** 4)
    return study


if __name__ == '__main__':
    parameter_study = create_full_parameter_study(gpu=True)
    parameter_study.run_from_command_line()
