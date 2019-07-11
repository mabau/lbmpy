import numpy as np
import sympy as sp

from lbmpy.forcemodels import Guo
from lbmpy.methods.entropic_eq_srt import create_srt_entropic
from lbmpy.scenarios import create_lid_driven_cavity
from lbmpy.stencils import get_stencil


def test_entropic_methods():
    sc_kbc = create_lid_driven_cavity((20, 20), method='trt-kbc-n4', relaxation_rate=1.9999,
                                      entropic_newton_iterations=3, entropic=True, compressible=True,
                                      force=(-1e-10, 0))

    sc_srt = create_lid_driven_cavity((40, 40), relaxation_rate=1.9999, lid_velocity=0.05, compressible=True,
                                      force=(-1e-10, 0))

    sc_entropic = create_lid_driven_cavity((40, 40), method='entropic-srt', relaxation_rate=1.9999,
                                           lid_velocity=0.05, compressible=True, force=(-1e-10, 0))

    sc_srt.run(1000)
    sc_kbc.run(1000)
    sc_entropic.run(1000)
    assert np.isnan(np.max(sc_srt.velocity[:, :]))
    assert np.isfinite(np.max(sc_kbc.velocity[:, :]))
    assert np.isfinite(np.max(sc_entropic.velocity[:, :]))


def test_entropic_srt():
    stencil = get_stencil("D2Q9")
    method = create_srt_entropic(stencil, 1.8, Guo((0, 1e-6)), True)
    assert method.zeroth_order_equilibrium_moment_symbol == sp.symbols("rho")
    assert method.first_order_equilibrium_moment_symbols == sp.symbols("u_:2")
