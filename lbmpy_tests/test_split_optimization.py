import numpy as np
import pytest

from lbmpy.creationfunctions import create_lb_ast
from lbmpy.scenarios import create_lid_driven_cavity
from pystencils.sympyextensions import count_operations_in_ast
from sympy.core.cache import clear_cache


def test_split_number_of_operations():
    # For the following configurations the number of operations for splitted and un-splitted version are
    # exactly equal. This is not true for D3Q15 and D3Q27 because some sub-expressions are computed in multiple
    # splitted, inner loops.
    for stencil in ['D2Q9', 'D3Q19']:
        for compressible in (True, False):
            for method in ('srt', 'trt'):
                common_params = {'stencil': stencil,
                                 'method': method,
                                 'compressible': compressible,
                                 'force_model': 'luo',
                                 'force': (1e-6, 1e-5, 1e-7)
                                 }
                ast_with_splitting = create_lb_ast(optimization={'split': True}, **common_params)
                ast_without_splitting = create_lb_ast(optimization={'split': False}, **common_params)

                op_with_splitting = count_operations_in_ast(ast_with_splitting)
                op_without_splitting = count_operations_in_ast(ast_without_splitting)
                assert op_without_splitting['muls'] == op_with_splitting['muls']
                assert op_without_splitting['adds'] == op_with_splitting['adds']
                assert op_without_splitting['divs'] == op_with_splitting['divs']


@pytest.mark.parametrize('stencil', ['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'])
@pytest.mark.parametrize('compressible', [True, False])
@pytest.mark.parametrize('method', ['srt', 'mrt'])
@pytest.mark.parametrize('force', [(0, 0, 0), (1e-6, 1e-7, 2e-6)])
@pytest.mark.longrun
def test_equivalence(stencil, compressible, method, force):
    relaxation_rates = [1.8, 1.7, 1.0, 1.0, 1.0, 1.0]
    clear_cache()
    common_params = {'domain_size': (10, 20) if stencil.startswith('D2') else (5, 10, 7),
                     'stencil': stencil,
                     'method': method,
                     'weighted': True,
                     'compressible': compressible,
                     'force': force,
                     'force_model': 'schiller',
                     'relaxation_rates': relaxation_rates}
    print("Running Scenario", common_params)
    with_split = create_lid_driven_cavity(optimization={'split': True}, **common_params)
    without_split = create_lid_driven_cavity(optimization={'split': False}, **common_params)
    with_split.run(100)
    without_split.run(100)
    np.testing.assert_almost_equal(with_split.velocity_slice(), without_split.velocity_slice())


def test_equivalence_short():
    relaxation_rates = [1.8, 1.7, 1.0, 1.0, 1.0, 1.0]
    for stencil, compressible, method, force in [('D2Q9', True, 'srt', 1e-7), ('D3Q19', False, 'mrt', 0)]:
        dim = int(stencil[1])
        common_params = {'domain_size': (20, 30) if stencil.startswith('D2') else (10, 13, 7),
                         'stencil': stencil,
                         'method': method,
                         'weighted': True,
                         'compressible': compressible,
                         'force': (force, 0, 0)[:dim],
                         'relaxation_rates': relaxation_rates}
        print("Running Scenario", common_params)
        with_split = create_lid_driven_cavity(optimization={'split': True}, **common_params)
        without_split = create_lid_driven_cavity(optimization={'split': False}, **common_params)
        with_split.run(100)
        without_split.run(100)
        np.testing.assert_almost_equal(with_split.velocity_slice(), without_split.velocity_slice())
