"""Shear wave scenario from
    The cumulant lattice Boltzmann equation in three dimensions: Theory and validation
    by  Geier, Martin; Schönherr, Martin; Pasquali, Andrea; Krafczyk, Manfred (2015)

    :cite:`geier2015` Chapter 5.1

    NOTE: for integration tests, the parameter study is greatly shortened, i.e., the runs are shortened in time and
    not all resolutions and viscosities are considered. Nevertheless, all values used by Geier et al. are still in
    the setup, only commented, and remain ready to be used (check for comments that start with `NOTE`).
"""
import numpy as np
import pytest
import sympy as sp

from lbmpy import LatticeBoltzmannStep, LBStencil
from lbmpy.creationfunctions import LBMConfig, LBMOptimisation
from lbmpy.db import LbmpyJsonSerializer
from lbmpy.enums import Method, Stencil
from lbmpy.relaxationrates import (
    relaxation_rate_from_lattice_viscosity, relaxation_rate_from_magic_number)
from pystencils import Target, create_data_handling, CreateKernelConfig


def get_exponent_term(l, **kwargs):
    pi = np.pi
    return (2 * pi / l) ** 2 + (4 * pi / (3 * l)) ** 2


def get_initial_velocity_field(l, l_0, u_0, v_0, y_size, **kwargs):
    pi = np.pi
    domain_size = (l, y_size, 3 * l // 2)
    vel = np.zeros(domain_size + (3,))
    ranges = [np.arange(n, dtype=float) for n in vel.shape[:-1]]
    x, y, z = np.meshgrid(*ranges, indexing='ij')

    vel[..., 0] = u_0 * l_0 / l
    vel[..., 1] = v_0 * l_0 / l * np.sin(2 * pi * x / l) * np.cos(4 * pi * z / (3 * l))
    vel[..., 2] = 0

    return vel


def get_analytical_solution(l, l_0, u_0, v_0, nu, t):
    pi = np.pi
    domain_size = (l, 3, 3 * l // 2)
    vel = np.zeros(domain_size + (3,))
    ranges = [np.arange(n, dtype=float) for n in vel.shape[:-1]]
    x, y, z = np.meshgrid(*ranges, indexing='ij')

    exponent_term = (2 * pi / l) ** 2 + (4 * pi / (3 * l)) ** 2
    vel[..., 0] = u_0 * l_0 / l
    vel[..., 1] = v_0 * l_0 / l * np.sin(2 * pi * (x + u_0 * t) / l) * \
                  np.cos(4 * pi * z / (3 * l)) * np.exp(-nu * t * exponent_term)
    vel[..., 2] = 0

    return vel


def plot_y_velocity(vel, **kwargs):
    import matplotlib.pyplot as plt
    vel = vel[:, vel.shape[1] // 2, :, 1]
    ranges = [np.arange(n, dtype=float) for n in vel.shape]
    x, y = np.meshgrid(*ranges, indexing='ij')
    fig = plt.gcf()
    ax = fig.gca(projection='3d')

    ax.plot_surface(x, y, vel, cmap='coolwarm', rstride=2, cstride=2,
                    linewidth=0, antialiased=True, **kwargs)


def fit_and_get_slope(x_values, y_values):
    matrix = np.vstack([x_values, np.ones(len(x_values))]).T
    m, _ = np.linalg.lstsq(matrix, y_values, rcond=1e-14)[0]
    return m


def get_phase_error(phases, evaluation_interval):
    x_values = np.arange(len(phases) * evaluation_interval, step=evaluation_interval)
    phase_error = fit_and_get_slope(x_values, phases)
    return phase_error


def get_viscosity(abs_values, evaluation_interval, l):
    y_values = [np.log(v) for v in abs_values]
    x_values = np.arange(0, evaluation_interval * len(y_values), evaluation_interval)
    m = fit_and_get_slope(x_values, y_values)
    exp_term = get_exponent_term(l)
    return - m / exp_term


def get_amplitude_and_phase(vel_slice):
    fft = np.fft.rfft2(vel_slice)
    fft_abs = np.abs(fft)
    fft_phase = np.angle(fft)
    max_index = np.unravel_index(fft_abs.argmax(), fft_abs.shape)
    return fft_abs[max_index], fft_phase[max_index]


def run(l, l_0, u_0, v_0, nu, y_size, lbm_config, lbm_optimisation, config):
    inv_result = {
        'viscosity_error': np.nan,
        'phase_error': np.nan,
        'mlups': np.nan,
    }
    if lbm_config.initial_velocity:
        # no manually specified initial velocity allowed in shear wave setup
        lbm_config.initial_velocity = None

    print(f"Running size l={l} nu={nu}")
    initial_vel_arr = get_initial_velocity_field(l, l_0, u_0, v_0, y_size)
    omega = relaxation_rate_from_lattice_viscosity(nu)

    if lbm_config.fourth_order_correction and omega < 1.75:
        pytest.skip("Fourth-order correction only allowed for omega >= 1.75.")

    lbm_config.relaxation_rates = [sp.sympify(r) for r in lbm_config.relaxation_rates]
    lbm_config.relaxation_rates = [r.subs(sp.Symbol("omega"), omega) for r in lbm_config.relaxation_rates]

    periodicity_in_kernel = (lbm_optimisation.builtin_periodicity != (False, False, False))
    domain_size = initial_vel_arr.shape[:-1]

    data_handling = create_data_handling(domain_size, periodicity=not periodicity_in_kernel,
                                         default_ghost_layers=1, parallel=False)

    scenario = LatticeBoltzmannStep(data_handling=data_handling, name="periodic_scenario", lbm_config=lbm_config,
                                    lbm_optimisation=lbm_optimisation, config=config)
    for b in scenario.data_handling.iterate(ghost_layers=False):
        np.copyto(b[scenario.velocity_data_name], initial_vel_arr[b.global_slice])
    scenario.set_pdf_fields_from_macroscopic_values()

    # NOTE: use those values to limit the runtime in integration tests
    total_time_steps = 2000 * (l // l_0) ** 2
    initial_time_steps = 1100 * (l // l_0) ** 2
    eval_interval = 100 * (l // l_0) ** 2
    # NOTE: for simulating the real shear-wave scenario from Geier et al. use the following values
    # total_time_steps = 20000 * (l // l_0) ** 2
    # initial_time_steps = 11000 * (l // l_0) ** 2
    # eval_interval = 1000 * (l // l_0) ** 2

    scenario.run(initial_time_steps)
    if np.isnan(scenario.velocity_slice()).any():
        print("   Result", inv_result)
        return inv_result

    magnitudes = []
    phases = []
    mlups = []
    while scenario.time_steps_run < total_time_steps:
        mlup_current_run = scenario.benchmark_run(eval_interval)
        if np.isnan(scenario.velocity_slice()).any():
            return inv_result
        magnitude, phase = get_amplitude_and_phase(scenario.velocity[:, y_size // 2, :, 1])
        magnitudes.append(magnitude)
        phases.append(abs(phase))
        mlups.append(mlup_current_run)

    assert scenario.time_steps_run == total_time_steps
    estimated_viscosity = get_viscosity(magnitudes, eval_interval, l)
    viscosity_error = abs(estimated_viscosity - nu) / nu  # called ER_\nu in the paper

    result = {
        'viscosity_error': viscosity_error,
        'phaseError': get_phase_error(phases, eval_interval),
        'mlups': max(mlups),
    }
    print("   Result", result)
    return result


def create_full_parameter_study():
    from pystencils.runhelper import ParameterStudy

    setup_params = {
        'l_0': 32,
        'u_0': 0.096,
        'v_0': 0.1,
        'y_size': 1
    }

    omega, omega_f = sp.symbols("omega, omega_f")

    # NOTE: use those values to limit the runtime in integration tests
    ls = [32]
    nus = [1e-5]
    # NOTE: for simulating the real shear-wave scenario from Geier et al. use the following values
    # ls = [32 * 2 ** i for i in range(0, 5)]
    # nus = [1e-2, 1e-3, 1e-4, 1e-5]

    srt_and_trt_methods = [LBMConfig(method=method,
                                     stencil=LBStencil(stencil),
                                     compressible=comp,
                                     relaxation_rates=[omega, str(relaxation_rate_from_magic_number(omega))],
                                     equilibrium_order=eqOrder,
                                     continuous_equilibrium=mbEq)
                           for method in (Method.SRT, Method.TRT)
                           for stencil in (Stencil.D3Q19, Stencil.D3Q27)
                           for comp in (True, False)
                           for eqOrder in (1, 2, 3)
                           for mbEq in (True, False)]

    optimization_srt_trt = LBMOptimisation(split=True, cse_pdfs=True)

    config = CreateKernelConfig(target=Target.CPU)

    methods = [LBMConfig(stencil=LBStencil(Stencil.D3Q27), method=Method.TRT, relaxation_rates=[omega, 1]),
               LBMConfig(stencil=LBStencil(Stencil.D3Q27), method=Method.MRT, relaxation_rates=[omega],
                         equilibrium_order=2),
               LBMConfig(stencil=LBStencil(Stencil.D3Q27), method=Method.MRT, relaxation_rates=[omega],
                         equilibrium_order=3),
               LBMConfig(stencil=LBStencil(Stencil.D3Q27), method=Method.CUMULANT, relaxation_rates=[omega],  # cumulant
                         compressible=True, continuous_equilibrium=True, equilibrium_order=3),
               LBMConfig(stencil=LBStencil(Stencil.D3Q27), method=Method.CUMULANT, relaxation_rates=[omega],  # cumulant with fourth-order correction
                         compressible=True, continuous_equilibrium=True, fourth_order_correction=0.1),
               LBMConfig(stencil=LBStencil(Stencil.D3Q27), method=Method.TRT_KBC_N4, relaxation_rates=[omega, omega_f],
                         entropic=True, zero_centered=False, continuous_equilibrium=True, equilibrium_order=2),
               LBMConfig(stencil=LBStencil(Stencil.D3Q27), method=Method.TRT_KBC_N4, relaxation_rates=[omega, omega_f],
                         entropic=True, zero_centered=False, continuous_equilibrium=True, equilibrium_order=3),

               # entropic cumulant: not supported for the moment
               # LBMConfig(method=Method.CUMULANT, relaxation_rates=["omega", "omega_f"], entropic=True, zero_centered=False,
               #           compressible=True, continuous_equilibrium=True, equilibrium_order=3)
               ]

    parameter_study = ParameterStudy(run, database_connector="shear_wave_db",
                                     serializer_info=('lbmpy_serializer', LbmpyJsonSerializer))
    for L in ls:
        for nu in nus:
            for methodParams in srt_and_trt_methods:
                simulation_params = setup_params.copy()

                simulation_params['lbm_config'] = methodParams
                simulation_params['lbm_optimisation'] = optimization_srt_trt
                simulation_params['config'] = config

                simulation_params['l'] = L
                simulation_params['nu'] = nu
                l_0 = simulation_params['l_0']
                parameter_study.add_run(simulation_params.copy(), weight=(L / l_0) ** 4)

            for methodParams in methods:
                simulation_params = setup_params.copy()

                simulation_params['lbm_config'] = methodParams
                simulation_params['lbm_optimisation'] = LBMOptimisation()
                simulation_params['config'] = config

                simulation_params['l'] = L
                simulation_params['nu'] = nu
                l_0 = simulation_params['l_0']
                parameter_study.add_run(simulation_params.copy(), weight=(L / l_0) ** 4)
    return parameter_study


def test_shear_wave():
    pytest.importorskip('cupy')
    params = {
        'y_size': 1,
        'l_0': 32,
        'u_0': 0.096,
        'v_0': 0.1,

        'l': 32,
        'nu': 1e-2,
    }

    kernel_config = CreateKernelConfig(target=Target.GPU)
    lbm_config = LBMConfig(method=Method.SRT, relaxation_rates=[sp.Symbol("omega")], stencil=LBStencil(Stencil.D3Q27),
                           compressible=True, continuous_equilibrium=True, equilibrium_order=2)

    run(**params, lbm_config=lbm_config, config=kernel_config, lbm_optimisation=LBMOptimisation())


@pytest.mark.longrun
def test_all_scenarios():
    parameter_study = create_full_parameter_study()
    parameter_study.run()
