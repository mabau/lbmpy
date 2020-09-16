import numpy as np

from lbmpy.scenarios import create_channel


def test_fluctuating_generation_pipeline():
    ch = create_channel((10, 10), stencil='D2Q9', method='mrt', weighted=True, relaxation_rates=[1.5] * 5, force=1e-5,
                        force_model='luo', fluctuating={'temperature': 1e-9}, kernel_params={'time_step': 1, 'seed': 312},
                        optimization={'cse_global': True})

    ch.run(10)
    assert np.max(ch.velocity[:, :]) < 0.1
