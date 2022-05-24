import platform

import numpy as np
import sympy as sp
import pytest

from lbmpy.enums import ForceModel, Method
from lbmpy.scenarios import create_lid_driven_cavity


@pytest.mark.parametrize('method', [Method.TRT_KBC_N1, Method.TRT_KBC_N2, Method.TRT_KBC_N3, Method.TRT_KBC_N4])
def test_entropic_methods(method):
    if platform.system().lower() == 'windows' and method == Method.TRT_KBC_N4:
        pytest.skip("For some reason this test does not run on windows", allow_module_level=True)
    
    sc_kbc = create_lid_driven_cavity((20, 20), method=method,
                                      relaxation_rates=[1.9999, sp.Symbol("omega_free")],
                                      entropic_newton_iterations=3, entropic=True, compressible=True,
                                      zero_centered=False, force=(-1e-10, 0), force_model=ForceModel.LUO)

    sc_kbc.run(1000)
    assert np.isfinite(np.max(sc_kbc.velocity[:, :]))
