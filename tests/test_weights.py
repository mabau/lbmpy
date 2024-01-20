import pytest
from lbmpy.creationfunctions import create_lb_method, LBMConfig
from lbmpy.enums import Method, Stencil
from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.stencils import LBStencil


def compare_weights(method, zero_centered, continuous_equilibrium, stencil_name):
    stencil = LBStencil(stencil_name)
    hardcoded_weights = get_weights(stencil)

    method = create_lb_method(LBMConfig(stencil=stencil, method=method, zero_centered=zero_centered, 
                                        continuous_equilibrium=continuous_equilibrium))
    weights = method.weights

    for i in range(len(weights)):
        assert hardcoded_weights[i] == weights[i]


@pytest.mark.parametrize('method', [Method.SRT, Method.TRT])
@pytest.mark.parametrize('zero_centered', [False, True])
@pytest.mark.parametrize('continuous_equilibrium', [False, True])
@pytest.mark.parametrize('stencil_name', [Stencil.D2Q9, Stencil.D3Q7])
def test_weight_calculation(method, zero_centered, continuous_equilibrium, stencil_name):
    compare_weights(method, zero_centered, continuous_equilibrium, stencil_name)


@pytest.mark.parametrize('method', [Method.MRT, Method.CENTRAL_MOMENT])
@pytest.mark.parametrize('continuous_equilibrium', [False, True])
@pytest.mark.parametrize('zero_centered', [False, True])
@pytest.mark.parametrize('stencil_name', [Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.longrun
def test_weight_calculation_longrun(method, zero_centered, continuous_equilibrium, stencil_name):
    compare_weights(method, zero_centered, continuous_equilibrium, stencil_name)