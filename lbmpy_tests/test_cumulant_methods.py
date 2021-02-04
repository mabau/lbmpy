import pytest

from lbmpy.methods import create_srt
from lbmpy.stencils import get_stencil
from lbmpy.methods.creationfunctions import create_with_default_polynomial_cumulants

@pytest.mark.parametrize('stencil_name', ['D2Q9', 'D3Q19', 'D3Q27'])
def test_weights(stencil_name):
    stencil = get_stencil(stencil_name)
    cumulant_method = create_with_default_polynomial_cumulants(stencil, [1])
    moment_method = create_srt(stencil, 1, cumulant=False, compressible=True, maxwellian_moments=True)
    assert cumulant_method.weights == moment_method.weights
