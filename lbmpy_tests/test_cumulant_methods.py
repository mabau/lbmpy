import pytest
import numpy as np

from lbmpy.creationfunctions import create_lb_method, create_lb_method_from_existing
from lbmpy.methods import create_srt
from lbmpy.stencils import get_stencil
from lbmpy.methods.creationfunctions import create_with_default_polynomial_cumulants
from lbmpy.scenarios import create_lid_driven_cavity

@pytest.mark.parametrize('stencil_name', ['D2Q9', 'D3Q19', 'D3Q27'])
def test_weights(stencil_name):
    stencil = get_stencil(stencil_name)
    cumulant_method = create_with_default_polynomial_cumulants(stencil, [1])
    moment_method = create_srt(stencil, 1, compressible=True, maxwellian_moments=True)
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

def test_create_cumulant_method_from_existing():
    method = create_lb_method(stencil='D2Q9', method='cumulant', relaxation_rate=1.5)
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


