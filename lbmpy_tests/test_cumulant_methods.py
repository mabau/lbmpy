import pytest

from lbmpy.creationfunctions import create_lb_method, create_lb_method_from_existing, LBMConfig
from lbmpy.enums import Method, Stencil
from lbmpy.methods import create_srt
from lbmpy.stencils import LBStencil
from lbmpy.methods.creationfunctions import create_with_default_polynomial_cumulants


@pytest.mark.parametrize('stencil_name', [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
def test_weights(stencil_name):
    stencil = LBStencil(stencil_name)
    cumulant_method = create_with_default_polynomial_cumulants(stencil, [1])
    moment_method = create_srt(stencil, 1, compressible=True, maxwellian_moments=True)
    assert cumulant_method.weights == moment_method.weights


def test_create_cumulant_method_from_existing():
    lbm_config = LBMConfig(stencil=LBStencil(Stencil.D2Q9), method=Method.CUMULANT, relaxation_rate=1.5)
    method = create_lb_method(lbm_config=lbm_config)
    old_relaxation_info_dict = method.relaxation_info_dict

    def modification_func(cumulant, eq, rate):
        if rate == 0:
            return cumulant, eq, 1.0
        return cumulant, eq, rate

    new_method = create_lb_method_from_existing(method, modification_func)
    new_relaxation_info_dict = new_method.relaxation_info_dict

    for i, (o, n) in enumerate(zip(old_relaxation_info_dict.items(), new_relaxation_info_dict.items())):
        assert o[0] == n[0]
        assert o[1].equilibrium_value == n[1].equilibrium_value
        if o[1].relaxation_rate == 0:
            assert n[1].relaxation_rate == 1.0
        else:
            assert o[1].relaxation_rate == n[1].relaxation_rate
