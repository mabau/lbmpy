from copy import deepcopy

import numpy as np
import pytest

from lbmpy.scenarios import create_channel


def run_equivalence_test(scenario_creator, time_steps=13, **params):
    print("Scenario", params)
    params['optimization']['target'] = 'cpu'
    cpu_scenario = scenario_creator(**params)
    params['optimization']['target'] = 'gpu'
    gpu_scenario = scenario_creator(**params)

    cpu_scenario.run(time_steps)
    gpu_scenario.run(time_steps)

    max_vel_error = np.max(np.abs(cpu_scenario.velocity_slice() - gpu_scenario.velocity_slice()))
    max_rho_error = np.max(np.abs(cpu_scenario.density_slice() - gpu_scenario.density_slice()))

    np.testing.assert_allclose(max_vel_error, 0, atol=1e-14)
    np.testing.assert_allclose(max_rho_error, 0, atol=1e-14)


def test_force_driven_channel_short():
    default = {
        'scenario_creator': create_channel,
        'domain_size': (32, 32),
        'relaxation_rates': [1.95, 1.9, 1.92],
        'method': 'mrt3',
        'pressure_difference': 0.001,
        'optimization': {},
    }
    scenarios = []

    # Different methods
    for ds, method, compressible, block_size, field_layout in [((17, 12), 'srt', False, (12, 4), 'reverse_numpy'),
                                                               ((18, 20), 'mrt3', True, (4, 2), 'zyxf'),
                                                               ((7, 11, 18), 'trt', False, False, 'numpy')]:
        params = deepcopy(default)
        if block_size is not False:
            params['optimization'].update({
                'gpu_indexing_params': {'block_size': block_size}
            })
        else:
            params['optimization']['gpu_indexing'] = 'line'

        params['domain_size'] = ds
        params['method'] = method
        params['compressible'] = compressible
        params['optimization']['field_layout'] = field_layout
        scenarios.append(params)

    for scenario in scenarios:
        run_equivalence_test(**scenario)


@pytest.mark.longrun
def test_force_driven_channel():
    default = {
        'scenario_creator': create_channel,
        'domain_size': (32, 32),
        'relaxation_rates': [1.95, 1.9, 1.92],
        'method': 'mrt3',
        'pressure_difference': 0.001,
        'optimization': {},
    }

    scenarios = []

    # Different methods
    for method in ('srt', 'mrt3'):
        for compressible in (True, False):
            params = deepcopy(default)
            params['optimization'].update({
                'gpu_indexing_params': {'block_size': (16, 16)}
            })
            params['method'] = method
            params['compressible'] = compressible
            scenarios.append(params)

    # Blocked indexing with different block sizes
    for block_size in ((16, 16), (8, 16), (4, 2)):
        params = deepcopy(default)
        params['method'] = 'mrt3'
        params['compressible'] = True
        params['optimization'].update({
            'gpu_indexing': 'block',
            'gpu_indexing_params': {'block_size': block_size}
        })
        scenarios.append(params)

    # Line wise indexing
    params = deepcopy(default)
    params['optimization']['gpu_indexing'] = 'line'
    scenarios.append(params)

    # Different field layouts
    for field_layout in ('numpy', 'reverse_numpy', 'zyxf'):
        for fixed_size in (False, True):
            params = deepcopy(default)
            params['optimization'].update({
                'gpu_indexing_params': {'block_size': (16, 16)}
            })
            if fixed_size:
                params['optimization']['field_size'] = params['domain_size']
            else:
                params['optimization']['field_size'] = None

            params['optimization']['field_layout'] = field_layout
            scenarios.append(params)

    print("Testing %d scenarios" % (len(scenarios),))
    for scenario in scenarios:
        run_equivalence_test(**scenario)
