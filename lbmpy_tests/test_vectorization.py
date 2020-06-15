import numpy as np
import pytest

from lbmpy.scenarios import create_lid_driven_cavity


def test_lbm_vectorization_short():
    print("Computing reference solutions")
    size1 = (64, 32)
    relaxation_rate = 1.8

    ldc1_ref = create_lid_driven_cavity(size1, relaxation_rate=relaxation_rate)
    ldc1_ref.run(10)

    ldc1 = create_lid_driven_cavity(size1, relaxation_rate=relaxation_rate,
                                    optimization={'vectorization': {'instruction_set': 'avx',
                                                                    'assume_aligned': True,
                                                                    'nontemporal': True,
                                                                    'assume_inner_stride_one': True,
                                                                    'assume_sufficient_line_padding': False,
                                                                    }},
                                    fixed_loop_sizes=False)
    ldc1.run(10)


@pytest.mark.longrun
def test_lbm_vectorization():
    vectorization_options = [{'instruction_set': instruction_set,
                              'assume_aligned': aa,
                              'nontemporal': nt,
                              'assume_inner_stride_one': True,
                              'assume_sufficient_line_padding': lp,
                              }
                             for instruction_set in ('sse', 'avx')
                             for aa, lp in ([False, False], [True, False], [True, True],)
                             for nt in (False, True)
                             ]
    time_steps = 100
    size1 = (64, 32)
    size2 = (666, 34)
    relaxation_rate = 1.8

    print("Computing reference solutions")
    ldc1_ref = create_lid_driven_cavity(size1, relaxation_rate=relaxation_rate)
    ldc1_ref.run(time_steps)
    ldc2_ref = create_lid_driven_cavity(size2, relaxation_rate=relaxation_rate)
    ldc2_ref.run(time_steps)

    for double_precision in (False, True):
        for vec_opt in vectorization_options:
            for fixed_loop_sizes in (True, False):
                optimization = {'double_precision': double_precision,
                                'vectorization': vec_opt,
                                'cse_global': True,
                                }
                print("Vectorization test, double precision {}, vectorization {}, fixed loop sizes {}".format(
                    double_precision, vec_opt, fixed_loop_sizes))
                ldc1 = create_lid_driven_cavity(size1, relaxation_rate=relaxation_rate, optimization=optimization,
                                                fixed_loop_sizes=fixed_loop_sizes)
                ldc1.run(time_steps)
                np.testing.assert_almost_equal(ldc1_ref.velocity[:, :], ldc1.velocity[:, :])

                optimization['split'] = True
                ldc2 = create_lid_driven_cavity(size2, relaxation_rate=relaxation_rate, optimization=optimization,
                                                fixed_loop_sizes=fixed_loop_sizes)
                ldc2.run(time_steps)
                np.testing.assert_almost_equal(ldc2_ref.velocity[:, :], ldc2.velocity[:, :])


if __name__ == '__main__':
    test_lbm_vectorization()
