import pytest
import sympy as sp
from dataclasses import replace

from lbmpy.enums import Method, ForceModel, Stencil
from lbmpy.moments import (
    extract_monomials, get_default_moment_set_for_stencil, non_aliased_polynomial_raw_moments,
    exponent_tuple_sort_key)
from lbmpy.stencils import LBStencil

from lbmpy.methods.creationfunctions import create_with_monomial_cumulants
from lbmpy.maxwellian_equilibrium import get_weights

from lbmpy.moment_transforms import (
    PdfsToMomentsByMatrixTransform, PdfsToMomentsByChimeraTransform
)


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19])
def test_zero_centering_equilibrium_equivalence(stencil):
    stencil = LBStencil(stencil)
    transform_class = PdfsToMomentsByChimeraTransform
    omega = sp.Symbol('omega')
    r_rates = (omega,) * stencil.Q

    weights = sp.Matrix(get_weights(stencil))

    rho = sp.Symbol("rho")
    rho_background = sp.Integer(1)
    delta_rho = sp.Symbol("delta_rho")
    
    subs = { delta_rho : rho - rho_background }
    eqs = []

    for zero_centered in [False, True]:
        method = create_with_monomial_cumulants(stencil, r_rates, zero_centered=zero_centered)
        eq = method.get_equilibrium_terms()
        eqs.append(eq.subs(subs))

    assert (eqs[0] - eqs[1]).expand() == weights
