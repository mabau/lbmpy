import numpy as np
import sympy as sp
import pytest

from lbmpy.enums import ForceModel, Method, Stencil
from lbmpy.forcemodels import Guo
from lbmpy.methods.momentbased.entropic_eq_srt import create_srt_entropic
from lbmpy.scenarios import create_lid_driven_cavity
from lbmpy.stencils import LBStencil


@pytest.mark.parametrize('method', [Method.TRT_KBC_N1, Method.TRT_KBC_N2, Method.TRT_KBC_N3])
def test_entropic_methods(method):
    sc_kbc = create_lid_driven_cavity((20, 20), method=method,
                                      relaxation_rates=[1.9999, sp.Symbol("omega_free")],
                                      entropic_newton_iterations=3, entropic=True, compressible=True,
                                      force=(-1e-10, 0), force_model=ForceModel.LUO)

    sc_kbc.run(1000)
    assert np.isfinite(np.max(sc_kbc.velocity[:, :]))


def test_entropic_srt():
    stencil = LBStencil(Stencil.D2Q9)
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

