from dataclasses import replace

import numpy as np
import pytest
from pystencils import Backend, Target, CreateKernelConfig

from lbmpy.creationfunctions import LBMConfig, LBMOptimisation
from lbmpy.scenarios import create_channel
from lbmpy.enums import Method


def run_equivalence_test(domain_size, lbm_config, lbm_opt, config, time_steps=13):
    config = replace(config, target=Target.CPU)
    cpu_scenario = create_channel(domain_size=domain_size, pressure_difference=0.001,
                                  lbm_config=lbm_config, lbm_optimisation=lbm_opt, config=config)
    config = replace(config, target=Target.GPU, backend=Backend.CUDA)
    gpu_scenario = create_channel(domain_size=domain_size, pressure_difference=0.001,
                                  lbm_config=lbm_config, lbm_optimisation=lbm_opt, config=config)

    cpu_scenario.run(time_steps)
    gpu_scenario.run(time_steps)

    max_vel_error = np.max(np.abs(cpu_scenario.velocity_slice() - gpu_scenario.velocity_slice()))
    max_rho_error = np.max(np.abs(cpu_scenario.density_slice() - gpu_scenario.density_slice()))

    np.testing.assert_allclose(max_vel_error, 0, atol=1e-14)
    np.testing.assert_allclose(max_rho_error, 0, atol=1e-14)


@pytest.mark.parametrize('scenario', [((17, 12), Method.SRT, False, (12, 4), 'reverse_numpy'),
                                      ((18, 20), Method.MRT, True, (4, 2), 'zyxf'),
                                      ((7, 11, 18), Method.TRT, False, False, 'numpy')])
def test_force_driven_channel_short(scenario):
    pytest.importorskip("pycuda")
    ds = scenario[0]
    method = scenario[1]
    compressible = scenario[2]
    block_size = scenario[3]
    field_layout = scenario[4]

    lbm_config = LBMConfig(method=method, compressible=compressible, relaxation_rates=[1.95, 1.9, 1.92, 1.92])
    lbm_opt = LBMOptimisation(field_layout=field_layout)

    # Different methods
    if block_size is not False:
        config = CreateKernelConfig(gpu_indexing_params={'block_size': block_size})
    else:
        config = CreateKernelConfig(gpu_indexing='line')

    run_equivalence_test(domain_size=ds, lbm_config=lbm_config, lbm_opt=lbm_opt, config=config)
