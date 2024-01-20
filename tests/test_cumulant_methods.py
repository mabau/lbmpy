import pytest

from lbmpy.creationfunctions import create_lb_method, LBMConfig
from lbmpy.enums import Method, Stencil
from lbmpy.methods import create_srt
from lbmpy.stencils import LBStencil
from lbmpy.methods.creationfunctions import create_with_default_polynomial_cumulants


@pytest.mark.parametrize('stencil_name', [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
def test_weights(stencil_name):
    stencil = LBStencil(stencil_name)
    cumulant_method = create_with_default_polynomial_cumulants(stencil, [1])
    moment_method = create_srt(stencil, 1, compressible=True, continuous_equilibrium=True)
    assert cumulant_method.weights == moment_method.weights
