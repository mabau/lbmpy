import numpy as np
import sympy as sp

from lbmpy.forcemodels import Guo
from lbmpy.methods.momentbased.entropic_eq_srt import create_srt_entropic
from lbmpy.scenarios import create_lid_driven_cavity
from lbmpy.stencils import get_stencil


def test_entropic_methods():
    sc_kbc = create_lid_driven_cavity((20, 20), method='trt_kbc_n4', relaxation_rates=[1.9999, sp.Symbol("omega_free")],
                                      entropic_newton_iterations=3, entropic=True, compressible=True,
                                      force=(-1e-10, 0), force_model="luo")

    sc_srt = create_lid_driven_cavity((40, 40), relaxation_rate=1.9999, lid_velocity=0.05, compressible=True,
                                      force=(-1e-10, 0), force_model="luo")

    sc_entropic = create_lid_driven_cavity((40, 40), method='entropic_srt', relaxation_rate=1.9999,
                                           lid_velocity=0.05, compressible=True, force=(-1e-10, 0), force_model="luo")

    sc_srt.run(1000)
    sc_kbc.run(1000)
    sc_entropic.run(1000)
    assert np.isnan(np.max(sc_srt.velocity[:, :]))
    assert np.isfinite(np.max(sc_kbc.velocity[:, :]))
    assert np.isfinite(np.max(sc_entropic.velocity[:, :]))


def test_entropic_srt():
    stencil = get_stencil("D2Q9")
    relaxation_rate = 1.8
    method = create_srt_entropic(stencil, relaxation_rate, Guo((0, 1e-6)), True)
    assert method.zeroth_order_equilibrium_moment_symbol == sp.symbols("rho")
    assert method.first_order_equilibrium_moment_symbols == sp.symbols("u_:2")

    eq = method.get_equilibrium()
    terms = method.get_equilibrium_terms()
    rel = method.relaxation_rates

    for i in range(len(terms)):
        assert sp.simplify(eq.main_assignments[i].rhs - terms[i]) == 0
        assert rel[i] == 1.8

