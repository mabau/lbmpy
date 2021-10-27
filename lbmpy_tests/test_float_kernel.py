import pytest

import pystencils as ps

from lbmpy.creationfunctions import create_lb_function, LBMConfig
from lbmpy.enums import Method
from lbmpy.scenarios import create_lid_driven_cavity


@pytest.mark.parametrize('double_precision', [False, True])
@pytest.mark.parametrize('method_enum', [Method.SRT, Method.CENTRAL_MOMENT, Method.CUMULANT])
def test_creation(method_enum, double_precision):
    """Simple test that makes sure that only float variables are created"""
    lbm_config = LBMConfig(method=method_enum, relaxation_rate=1.5)
    config = ps.CreateKernelConfig(data_type="float64" if double_precision else "float32")
    func = create_lb_function(lbm_config=lbm_config, config=config)
    code = ps.get_code_str(func)

    if double_precision:
        assert 'float' not in code
        assert 'double' in code
    else:
        assert 'double' not in code
        assert 'float' in code


@pytest.mark.parametrize('double_precision', [False, True])
@pytest.mark.parametrize('method_enum', [Method.SRT, Method.CENTRAL_MOMENT, Method.CUMULANT])
def test_scenario(method_enum, double_precision):
    lbm_config = LBMConfig(method=method_enum, relaxation_rate=1.5)
    config = ps.CreateKernelConfig(data_type="double" if double_precision else "float32")
    sc = create_lid_driven_cavity((16, 16, 8), lbm_config=lbm_config, config=config)
    sc.run(1)
    code = ps.get_code_str(sc.ast)

    if double_precision:
        assert 'float' not in code
        assert 'double' in code
    else:
        assert 'double' not in code
        assert 'float' in code
