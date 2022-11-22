"""Shear wave scenario from
    The cumulant lattice Boltzmann equation in three dimensions: Theory and validation
    by  Geier, Martin; Schönherr, Martin; Pasquali, Andrea; Krafczyk, Manfred (2015)

    Chapter 5.1
"""
import numpy as np
import sympy as sp

import pytest
from pystencils import Target

from lbmpy.creationfunctions import update_with_default_parameters
from lbmpy.relaxationrates import (
    relaxation_rate_from_lattice_viscosity, relaxation_rate_from_magic_number)
from lbmpy.scenarios import create_fully_periodic_flow


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
    from matplotlib import cm
    vel = vel[:, vel.shape[1]//2, :, 1]
    ranges = [np.arange(n, dtype=float) for n in vel.shape]
    x, y = np.meshgrid(*ranges, indexing='ij')
    fig = plt.gcf()
    ax = fig.gca(projection='3d')

    ax.plot_surface(x, y, vel, cmap=cm.coolwarm, rstride=2, cstride=2,
                    linewidth=0, antialiased=True, **kwargs)


def fit_and_get_slope(x_values, y_values):
    matrix = np.vstack([x_values, np.ones(len(x_values))]).T
    m, _ = np.linalg.lstsq(matrix, y_values, rcond=1e-14)[0]
    return m


def get_phase_error(phases, evaluation_interval):
    xValues = np.arange(len(phases) * evaluation_interval, step=evaluation_interval)
    phaseError = fit_and_get_slope(xValues, phases)
    return phaseError


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


def run(l, l_0, u_0, v_0, nu, y_size, periodicity_in_kernel, **kwargs):
    inv_result = {
        'viscosity_error': np.nan,
        'phase_error': np.nan,
        'mlups': np.nan,
    }
    if 'initial_velocity' in kwargs:
        del kwargs['initial_velocity']

    print("Running size l=%d nu=%f " % (l, nu), kwargs)
    initial_vel_arr = get_initial_velocity_field(l, l_0, u_0, v_0, y_size)
    omega = relaxation_rate_from_lattice_viscosity(nu)

    kwargs['relaxation_rates'] = [sp.sympify(r) for r in kwargs['relaxation_rates']]
    if 'entropic' not in kwargs or not kwargs['entropic']:
        kwargs['relaxation_rates'] = [r.subs(sp.Symbol("omega"), omega) for r in kwargs['relaxation_rates']]

    scenario = create_fully_periodic_flow(initial_vel_arr, periodicity_in_kernel=periodicity_in_kernel, **kwargs)

    if 'entropic' in kwargs and kwargs['entropic']:
        scenario.kernelParams['omega'] = kwargs['relaxation_rates'][0].subs(sp.Symbol("omega"), omega)

    total_time_steps = 20000 * (l // l_0) ** 2
    initial_time_steps = 11000 * (l // l_0) ** 2
    eval_interval = 1000 * (l // l_0) ** 2
    scenario.run(initial_time_steps)
    if np.isnan(scenario.velocity_slice()).any():
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

    params = {
        'l_0': 32,
        'u_0': 0.096,
        'v_0': 0.1,
        'ySize': 1,
        'periodicityInKernel': True,
        'stencil': 'D3Q27',
        'compressible': True,
    }
    ls = [32 * 2 ** i for i in range(0, 5)]
    nus = [1e-2, 1e-3, 1e-4, 1e-5]

    srt_and_trt_methods = [{'method': method,
                            'stencil': stencil,
                            'compressible': comp,
                            'relaxation_rates': ["omega", str(relaxation_rate_from_magic_number(sp.Symbol("omega")))],
                            'equilibrium_order': eqOrder,
                            'continuous_equilibrium': mbEq,
                            'optimization': {'target': Target.CPU, 'split': True, 'cse_pdfs': True}}
                           for method in ('srt', 'trt')
                           for stencil in ('D3Q19', 'D3Q27')
                           for comp in (True, False)
                           for eqOrder in (1, 2, 3)
                           for mbEq in (True, False)]

    methods = [{'method': 'trt', 'relaxation_rates': ["omega", 1]},
               {'method': 'mrt3', 'relaxation_rates': ["omega", 1, 1], 'equilibrium_order': 2},
               {'method': 'mrt3', 'relaxation_rates': ["omega", 1, 1], 'equilibrium_order': 3},
               {'method': 'mrt3', 'cumulant': True, 'relaxation_rates': ["omega", 1, 1],  # cumulant
                'continuous_equilibrium': True, 'equilibrium_order': 3,
                'optimization': {'cse_global': True}},
               {'method': 'trt-kbc-n4', 'relaxation_rates': ["omega", "omega_f"], 'entropic': True,  # entropic order 2
                'continuous_equilibrium': True, 'equilibrium_order': 2},
               {'method': 'trt-kbc-n4', 'relaxation_rates': ["omega", "omega_f"], 'entropic': True,  # entropic order 3
                'continuous_equilibrium': True, 'equilibrium_order': 3},

               # entropic cumulant:
               {'method': 'trt-kbc-n4', 'relaxation_rates': ["omega", "omega_f"], 'entropic': True,
                'cumulant': True, 'continuous_equilibrium': True, 'equilibrium_order': 3},
               {'method': 'trt-kbc-n2', 'relaxation_rates': ["omega", "omega_f"], 'entropic': True,
                'cumulant': True, 'continuous_equilibrium': True, 'equilibrium_order': 3},
               {'method': 'mrt3', 'relaxation_rates': ["omega", "omega_f", "omega_f"], 'entropic': True,
                'cumulant': True, 'continuous_equilibrium': True, 'equilibrium_order': 3},
               ]

    parameter_study = ParameterStudy(run, database_connector="shear_wave_db")
    for L in ls:
        for nu in nus:
            for methodParams in methods + srt_and_trt_methods:
                simulation_params = params.copy()
                simulation_params.update(methodParams)
                if 'optimization' not in simulation_params:
                    simulation_params['optimization'] = {}
                new_params, new_opt_params = update_with_default_parameters(simulation_params,
                                                                            simulation_params['optimization'], False)
                simulation_params = new_params
                simulation_params['optimization'] = new_opt_params

                simulation_params['L'] = L
                simulation_params['nu'] = nu
                l_0 = simulation_params['l_0']
                parameter_study.add_run(simulation_params.copy(), weight=(L / l_0) ** 4)
    return parameter_study


def test_shear_wave():
    pytest.importorskip('pycuda')
    params = {
        'l_0': 32,
        'u_0': 0.096,
        'v_0': 0.1,

        'stencil': 'D3Q19',
        'compressible': True,
        "optimization": {"target": Target.GPU}
    }
    run(32, nu=1e-2, equilibrium_order=2, method='srt', y_size=1, periodicity_in_kernel=True,
        relaxation_rates=[sp.Symbol("omega"), 5, 5], continuous_equilibrium=True, **params)
