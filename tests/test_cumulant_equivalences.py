import pytest
import sympy as sp

from lbmpy.enums import Stencil
from lbmpy.stencils import LBStencil

from lbmpy.methods.creationfunctions import create_with_monomial_cumulants
from lbmpy.maxwellian_equilibrium import get_weights


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19])
def test_zero_centering_equilibrium_equivalence(stencil):
    stencil = LBStencil(stencil)
    omega = sp.Symbol('omega')
    r_rates = (omega,) * stencil.Q

    weights = sp.Matrix(get_weights(stencil))

    rho = sp.Symbol("rho")
    rho_background = sp.Integer(1)
    delta_rho = sp.Symbol("delta_rho")

    subs = {delta_rho: rho - rho_background}
    eqs = []

    for zero_centered in [False, True]:
        method = create_with_monomial_cumulants(stencil, r_rates, zero_centered=zero_centered)
        eq = method.get_equilibrium_terms()
        eqs.append(eq.subs(subs))

    assert (eqs[0] - eqs[1]).expand() == weights
