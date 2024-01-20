"""
Square channel - test of spurious currents normal to flow direction in reduced stencils (D3Q19, D3Q15)
Test case described in:
Truncation errors and the rotational invariance of three-dimensional lattice models in the lattice Boltzmann method
Silva, Semiao


python3 scenario_square_channel.py server
python3 scenario_square_channel.py client --host i10staff41 -P '{ "optimization" : { "target" : Target.GPU} }'
"""

import numpy as np
import sympy as sp

from lbmpy import Stencil, Method, ForceModel, LBStencil
from lbmpy.methods.creationfunctions import relaxation_rate_from_magic_number
from lbmpy.scenarios import create_channel
from pystencils import make_slice

defaultParameters = {
    'stencil': LBStencil(Stencil.D3Q19),
    'method': Method.SRT,

    'lambda_plus_sq': 4 / 25,
    'square_size': 15,
    'quadratic': True,
    're': 10,
    'compressible': True,
    'continuous_equilibrium': False,
    'equilibrium_order': 2,
    'force_model': ForceModel.GUO,
    'c_s_sq': 1 / 3,

    'analytic_initial_velocity': False,
    'convergence_check_interval': 10000,
    'convergence_threshold': 1e-10,

    'reynolds_nr_accuracy': 1e-8,
    'use_mean_for_reynolds': False,
}


def fit_and_get_slope(x_values, y_values):
    matrix = np.vstack([x_values, np.ones(len(x_values))]).T
    m, _ = np.linalg.lstsq(matrix, y_values)[0]
    return m


def get_convergence_order(series):
    return fit_and_get_slope(np.log10(series.index.values), np.log10(series))


def lambda_plus_sq_to_relaxation_rate(l):
    lambda_plus = np.sqrt(l)
    return 1.0 / (lambda_plus + 0.5)


def viscosity(relaxation_rate, viscosity_factor):
    return viscosity_factor * (1 / relaxation_rate - 0.5)


def x_vorticity(velocity_field, dx):
    grad_y_of_z = np.gradient(velocity_field[:, :, :, 2], dx, axis=1, edge_order=2)
    grad_z_of_y = np.gradient(velocity_field[:, :, :, 1], dx, axis=2, edge_order=2)
    return grad_y_of_z - grad_z_of_y


def x_vorticity_rms(velocity_field, dx):
    x_vort = x_vorticity(velocity_field, dx)
    return np.sqrt(np.sum(x_vort * x_vort) / x_vort.size)


def reynolds_number(max_velocity, relaxation_rate, length, viscosity_factor):
    return max_velocity * length / viscosity(relaxation_rate, viscosity_factor)


def force_from_reynolds_number(re, length, relaxation_rate, viscosity_factor, max_velocity_factor=1):
    nu = viscosity(relaxation_rate, viscosity_factor)
    max_velocity = re * nu / length * max_velocity_factor
    force = max_velocity * 4 * nu / ((length / 2) ** 2 * 11815 / 10032)
    return force


def analytical_vel_max(force, relaxation_rate, width, viscosity_factor):
    return force / (4 * viscosity(relaxation_rate, viscosity_factor)) * (width / 2) ** 2 * 11815 / 10032


def create_initial_velocity_field(force, relaxation_rate, domain_size, viscosity_factor):
    y_half = domain_size[1] / 2
    z_half = domain_size[2] / 2

    x_grid, y_grid = np.meshgrid(np.arange(domain_size[1]), np.arange(domain_size[2]), indexing='ij')
    x = (x_grid - y_half) / y_half
    y = (y_grid - z_half) / z_half

    x_velocity = -(x ** 12 - 66 * x ** 10 * y ** 2 + 495 * x ** 8 * y ** 4 - 33 * x ** 8 - 924 * x ** 6 * y ** 6
                   + 924 * x ** 6 * y ** 2 + 495 * x ** 4 * y ** 8 - 2310 * x ** 4 * y ** 4 + 1815 * x ** 4
                   - 66 * x ** 2 * y ** 10 + 924 * x ** 2 * y ** 6 - 10890 * x ** 2 * y ** 2
                   + 10032 * x ** 2 + y ** 12 - 33 * y ** 8 + 1815 * y ** 4 + 10032 * y ** 2 - 11815) / 10032

    x_velocity *= force / (4 * viscosity(relaxation_rate, viscosity_factor)) * (domain_size[1] / 2) ** 2
    x_velocity = np.repeat(x_velocity[np.newaxis, :, :], axis=0, repeats=domain_size[0])
    velocity = np.zeros(tuple(domain_size) + (3,))
    velocity[:, :, :, 0] = x_velocity
    return velocity


def run(convergence_check_interval=1000, convergence_threshold=1e-12, plot_result=False, lambda_plus_sq=4 / 25, re=10,
        square_size=15, quadratic=True, analytic_initial_velocity=False, reynolds_nr_accuracy=1e-8,
        use_mean_for_reynolds=True, **params):
    """
    3D Channel benchmark with rectangular cross-section
    :return: tuple containing
        - size of spurious velocity normal to flow direction normalized to maximum flow velocity
        - number of iterations until convergence
        - the computed reynolds number
    """
    omega = lambda_plus_sq_to_relaxation_rate(lambda_plus_sq)
    params['relaxation_rates'] = [omega, relaxation_rate_from_magic_number(omega, 3 / 16)]

    stencil = params['stencil']
    viscosity_factor = 1 / 2 if stencil == LBStencil(Stencil.D3Q15) and params['continuous_equilibrium'] else 1 / 3

    print("Running size %d quadratic %d analyticInit %d " %
          (square_size, quadratic, analytic_initial_velocity) + str(params))
    domain_size = [3, square_size, square_size]
    if not quadratic:
        domain_size[2] //= 2
        if domain_size[2] % 2 == 0:
            domain_size[2] -= 1

    params['domain_size'] = domain_size
    initial_force_value = force_from_reynolds_number(re, domain_size[1], omega,
                                                     viscosity_factor, 2 if use_mean_for_reynolds else 1)
    if not quadratic:
        initial_force_value *= 2  # analytical solution for force is invalid if not quadratic - a good guess is doubled

    if analytic_initial_velocity:
        initial_field = create_initial_velocity_field(initial_force_value, omega, domain_size, viscosity_factor)
        params['initial_velocity'] = initial_field

    scenario = create_channel(force=sp.Symbol('Force'), kernel_params={'Force': initial_force_value}, **params)

    last_vel_field = None
    iterations = 0

    while True:
        scenario.run(convergence_check_interval)
        iterations += convergence_check_interval
        vel = scenario.velocity_slice(make_slice[:, :, :])
        if last_vel_field is not None:
            change_in_time = float(np.ma.average(np.abs(vel - last_vel_field)))

            max_vel = np.array([np.max(scenario.velocity_slice(make_slice[:, :, :])[..., i]) for i in range(3)])

            vel_for_reynolds = np.mean(
                scenario.velocity_slice(make_slice[1, :, :, ])[..., 0]) if use_mean_for_reynolds else max_vel[0]
            computed_re = reynolds_number(vel_for_reynolds, omega, domain_size[1], viscosity_factor)

            reynolds_number_wrong = False
            if reynolds_nr_accuracy is not None and change_in_time < 1e-5:
                reynolds_number_wrong = abs(computed_re / re - 1) > reynolds_nr_accuracy
                if reynolds_number_wrong:
                    old_force = scenario.kernel_params['Force']
                    scenario.kernel_params['Force'] = old_force * re / computed_re

            ref_square_size = 15
            scale_factor = square_size / ref_square_size
            scaled_velocity = scenario.velocity_slice(make_slice[:, :, :]) * scale_factor
            scaled_vorticity_rms = x_vorticity_rms(scaled_velocity, 1 / scale_factor)

            print("    ", iterations, "ReErr", computed_re / re - 1, " spuriousVel ", max_vel[1] / max_vel[0],
                  " Vort ", scaled_vorticity_rms, " Change ", change_in_time)

            if np.isnan(max_vel).any():
                break

            if change_in_time < convergence_threshold and not reynolds_number_wrong:
                break
        last_vel_field = np.copy(vel)

    if plot_result:
        import lbmpy.plot as plt
        vel_profile = vel[1, params['domain_size'][1] // 2, :, 0]
        plt.subplot(1, 2, 1)
        plt.plot(vel_profile)

        vel_cross_section = vel[1, :, :, 1:]
        plt.subplot(1, 2, 2)
        plt.vector_field(vel_cross_section, step=1)

        print(max_vel)
        print(max_vel / max_vel[0])

        plt.show()

    velocity_profile = list(scenario.velocity[1, :, 0.5, 0].data)

    return {
        'normalized_spurious_vel_max': max_vel[1] / max_vel[0],
        'scaled_vorticity_rms': scaled_vorticity_rms,
        'x_vorticity_rms': x_vorticity_rms(scenario.velocity[:, :, :], 1),
        'iterations': iterations,
        'computed_re': computed_re,
        'velocity_profile': velocity_profile,
    }


def parameter_filter(p):
    if p.cumulant and p.compressible:
        return None
    if p.cumulant and not p.continuous_equilibrium:
        return None
    if p.cumulant and p.stencil == LBStencil(Stencil.D3Q15):
        return None
    if not p.quadratic and not p.reynolds_nr_accuracy:
        # analytical formula not valid for rectangular channel
        # -> rectangular setup should be run with adaptive force only
        return None
    return p


def weight(p):
    return int((p.square_size / 15) ** 4)


def small_study():
    from pystencils.runhelper import ParameterStudy

    parameter_study = ParameterStudy(run, database_connector="square_channel_db")

    common_degrees_of_freedom = [
        ('reynolds_nr_accuracy', [1e-8, None]),
        ('analytic_initial_velocity', [True]),
        ('force_model', [ForceModel.LUO]),
        ('method', [Method.SRT]),
        ('equilibrium_order', [2]),
        ('stencil', [LBStencil(Stencil.D3Q19)]),
        ('compressible', [True]),
        ('quadratic', [True, False]),
        ('continuous_equilibrium', [False, True]),
    ]
    convergence_study = common_degrees_of_freedom.copy()
    convergence_study += [('square_size', [15, 25, 35, 45, 53])]
    convergence_study += [('lambda_plus_sq', [4 / 25])]
    parameter_study.add_combinations(convergence_study, defaultParameters, parameter_filter, weight)
    return parameter_study


def create_full_parameter_study():
    from pystencils.runhelper import ParameterStudy

    parameter_study = ParameterStudy(run, database_connector="mongo://square_channel")

    common_degrees_of_freedom = [
        ('cumulant', [False, True]),
        ('cumulant', [False]),
        ('compressible', [False, True]),
        ('reynolds_nr_accuracy', [None, 1e-8]),
        ('stencil', [LBStencil(Stencil.D3Q19), LBStencil(Stencil.D3Q15)]),
        ('analytic_initial_velocity', [False]),
        ('force_model', [ForceModel.GUO, ForceModel.SIMPLE, ForceModel.SILVA, ForceModel.LUO]),
        ('method', [Method.SRT, Method.TRT]),
        ('equilibrium_order', [2, 3]),
        ('quadratic', [True, False]),
        ('continuous_equilibrium', [False, True]),
        ('use_mean_for_reynolds', [True, False]),
    ]

    convergence_study = common_degrees_of_freedom.copy()
    convergence_study += [('square_size', [15, 25, 35, 45, 53, 85, 135])]

    convergence_study += [('lambda_plus_sq', [4 / 25, 1 / 12])]
    parameter_study.add_combinations(convergence_study, defaultParameters, parameter_filter, weight)

    relaxation_rate_study = common_degrees_of_freedom.copy()
    relaxation_rate_study += [('square_size', [53])]
    values_from_silva_paper = [0.01, 0.04, 0.09, 0.167, 0.18, 0.25, 0.36]
    additional_values_near_004 = [0.02, 0.03, 0.035, 0.045, 0.05, 0.06, 0.07, 0.08]
    relaxation_rate_study += [('lambda_plus_sq', values_from_silva_paper + additional_values_near_004)]
    parameter_study.add_combinations(relaxation_rate_study, defaultParameters, parameter_filter, weight)

    return parameter_study


def d3q15_cs_sq_half_study():
    from pystencils.runhelper import ParameterStudy
    parameter_study = ParameterStudy(run, database_connector="square_channel_db_d3q15study_otherMoments")

    dofs = [
        ('compressible', [False, True]),
        ('reynolds_nr_accuracy', [None, ]),
        ('analytic_initial_velocity', [False]),
        ('force_model', [ForceModel.GUO, ForceModel.SILVA]),
        ('method', [Method.SRT, Method.TRT]),
        ('equilibrium_order', [2, 3]),
        ('stencil', [LBStencil(Stencil.D3Q15)]),
        ('quadratic', [True, ]),
        ('continuous_equilibrium', [True, ]),
        ('c_s_sq', [1 / 3]),
        ('square_size', [45, 85]),
    ]
    parameter_study.add_combinations(dofs, defaultParameters, parameter_filter, weight)
    return parameter_study


def d3q27_study():
    from pystencils.runhelper import ParameterStudy
    parameter_study = ParameterStudy(run, database_connector="mongo://square_channel")

    dofs = [
        ('compressible', [False]),
        ('reynolds_nr_accuracy', [None, ]),
        ('analytic_initial_velocity', [False]),
        ('force_model', [ForceModel.GUO, ForceModel.SILVA]),
        ('method', [Method.SRT]),
        ('equilibrium_order', [2]),
        ('stencil', [LBStencil(Stencil.D3Q27)]),
        ('continuous_equilibrium', [True, ]),
        ('c_s_sq', [1 / 3]),
        ('square_size', [15, 25, 35, 45, 53, 85, 135]),
        ('use_mean_for_reynolds', [False]),
    ]
    parameter_study.add_combinations(dofs, defaultParameters, parameter_filter, weight)
    return parameter_study


def test_square_channel():
    res = run(convergence_check_interval=1000, convergence_threshold=1e-5, plot_result=False, lambda_plus_sq=4 / 25,
              re=10, square_size=53, quadratic=True, analytic_initial_velocity=False, reynolds_nr_accuracy=None,
              force_model=ForceModel.BUICK, stencil=LBStencil(Stencil.D3Q19),
              continuous_equilibrium=False, equilibrium_order=2, compressible=True)

    assert 1e-5 < res['normalized_spurious_vel_max'] < 1.2e-5

    # TODO test again if compressible works when !113 is merged
    res = run(convergence_check_interval=1000, convergence_threshold=1e-5, plot_result=False, lambda_plus_sq=4 / 25,
              re=10, square_size=53, quadratic=True, analytic_initial_velocity=False, reynolds_nr_accuracy=None,
              force_model=ForceModel.BUICK, stencil=LBStencil(Stencil.D3Q19),
              continuous_equilibrium=True, equilibrium_order=2, compressible=False)

    assert res['normalized_spurious_vel_max'] < 1e-14


if __name__ == '__main__':
    create_full_parameter_study().run_from_command_line()
