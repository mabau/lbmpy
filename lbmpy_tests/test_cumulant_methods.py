import pytest
import numpy as np

from lbmpy.methods import create_srt
from lbmpy.stencils import get_stencil
from lbmpy.methods.creationfunctions import create_with_default_polynomial_cumulants
from lbmpy.scenarios import create_lid_driven_cavity

@pytest.mark.parametrize('stencil_name', ['D2Q9', 'D3Q19', 'D3Q27'])
def test_weights(stencil_name):
    stencil = get_stencil(stencil_name)
    cumulant_method = create_with_default_polynomial_cumulants(stencil, [1])
    moment_method = create_srt(stencil, 1, cumulant=False, compressible=True, maxwellian_moments=True)
    assert cumulant_method.weights == moment_method.weights

def test_cumulant_ldc():
    sc_cumulant = create_lid_driven_cavity((20, 20), method='cumulant', relaxation_rate=1.999999,
                                            compressible=True, force=(-1e-10, 0))

    sc_cumulant_3D = create_lid_driven_cavity((20, 20, 3), method='cumulant', relaxation_rate=1.999999,
                                              compressible=True, force=(-1e-10, 0, 0),
                                              galilean_correction=True)

    sc_cumulant.run(1000)
    sc_cumulant_3D.run(1000)
    assert np.isfinite(np.max(sc_cumulant.velocity[:, :]))
    assert np.isfinite(np.max(sc_cumulant_3D.velocity[:, :, :]))
