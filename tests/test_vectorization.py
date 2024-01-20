import numpy as np
import pytest

import pystencils as ps
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets
from lbmpy.scenarios import create_lid_driven_cavity
from lbmpy.creationfunctions import LBMConfig, LBMOptimisation


@pytest.mark.skipif(not get_supported_instruction_sets(), reason='cannot detect CPU instruction set')
def test_lbm_vectorization_short():
    print("Computing reference solutions")
    size1 = (64, 32)
    relaxation_rate = 1.8

    ldc1_ref = create_lid_driven_cavity(size1, relaxation_rate=relaxation_rate)
    ldc1_ref.run(10)

    lbm_config = LBMConfig(relaxation_rate=relaxation_rate)
    config = ps.CreateKernelConfig(cpu_vectorize_info={'instruction_set': get_supported_instruction_sets()[-1],
                                                       'assume_aligned': True,
                                                       'nontemporal': True,
                                                       'assume_inner_stride_one': True,
                                                       'assume_sufficient_line_padding': False,
                                                       })
    ldc1 = create_lid_driven_cavity(size1, lbm_config=lbm_config, config=config,
                                    fixed_loop_sizes=False)
    ldc1.run(10)


@pytest.mark.parametrize('instruction_set', get_supported_instruction_sets())
@pytest.mark.parametrize('aligned_and_padding', [[False, False], [True, False], [True, True]])
@pytest.mark.parametrize('nontemporal', [False, True])
@pytest.mark.parametrize('double_precision', [False, True])
@pytest.mark.parametrize('fixed_loop_sizes', [False, True])
@pytest.mark.longrun
def test_lbm_vectorization(instruction_set, aligned_and_padding, nontemporal, double_precision, fixed_loop_sizes):
    vectorization_options = {'instruction_set': instruction_set,
                             'assume_aligned': aligned_and_padding[0],
                             'nontemporal': nontemporal,
                             'assume_inner_stride_one': True,
                             'assume_sufficient_line_padding': aligned_and_padding[1]}
    time_steps = 100
    size1 = (64, 32)
    size2 = (666, 34)
    relaxation_rate = 1.8

    print("Computing reference solutions")
    ldc1_ref = create_lid_driven_cavity(size1, relaxation_rate=relaxation_rate)
    ldc1_ref.run(time_steps)
    ldc2_ref = create_lid_driven_cavity(size2, relaxation_rate=relaxation_rate)
    ldc2_ref.run(time_steps)

    lbm_config = LBMConfig(relaxation_rate=relaxation_rate)
    config = ps.CreateKernelConfig(data_type="float64" if double_precision else "float32",
                                   default_number_float="float64" if double_precision else "float32",
                                   cpu_vectorize_info=vectorization_options)
    lbm_opt_split = LBMOptimisation(cse_global=True, split=True)
    lbm_opt = LBMOptimisation(cse_global=True, split=False)

    print(f"Vectorization test, double precision {double_precision}, vectorization {vectorization_options}, "
          f"fixed loop sizes {fixed_loop_sizes}")
    ldc1 = create_lid_driven_cavity(size1, fixed_loop_sizes=fixed_loop_sizes,
                                    lbm_config=lbm_config, lbm_optimisation=lbm_opt, config=config)
    ldc1.run(time_steps)
    np.testing.assert_almost_equal(ldc1_ref.velocity[:, :], ldc1.velocity[:, :])

    ldc2 = create_lid_driven_cavity(size2, fixed_loop_sizes=fixed_loop_sizes,
                                    lbm_config=lbm_config, lbm_optimisation=lbm_opt_split, config=config)
    ldc2.run(time_steps)
    np.testing.assert_almost_equal(ldc2_ref.velocity[:, :], ldc2.velocity[:, :])
