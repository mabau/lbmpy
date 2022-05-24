import numpy as np
import sympy as sp

import pystencils as ps
from lbmpy.enums import ForceModel, Method, Stencil
from lbmpy.scenarios import create_lid_driven_cavity

from lbmpy_tests.poiseuille import poiseuille_channel


def test_poiseuille_channel_quicktest():
    poiseuille_channel(target=ps.Target.CPU, stencil_name=Stencil.D2Q9)


def test_entropic_methods():
    sc_kbc = create_lid_driven_cavity((40, 40), method=Method.TRT_KBC_N4,
                                      relaxation_rates=[1.9999, sp.Symbol("omega_free")],
                                      entropic_newton_iterations=3, entropic=True, compressible=True,
                                      zero_centered=False, force=(-1e-10, 0), force_model=ForceModel.LUO)

    sc_srt = create_lid_driven_cavity((40, 40), relaxation_rate=1.9999, lid_velocity=0.05, compressible=True,
                                      zero_centered=False, force=(-1e-10, 0), force_model=ForceModel.LUO)

    sc_srt.run(1000)
    sc_kbc.run(1000)
    assert np.isnan(np.max(sc_srt.velocity[:, :]))
    assert np.isfinite(np.max(sc_kbc.velocity[:, :]))


def test_cumulant_ldc():
    sc_cumulant = create_lid_driven_cavity((40, 40), method=Method.CUMULANT, relaxation_rate=1.999999,
                                           compressible=True, force=(-1e-10, 0))

    sc_cumulant.run(100)
    assert np.isfinite(np.max(sc_cumulant.velocity[:, :]))
