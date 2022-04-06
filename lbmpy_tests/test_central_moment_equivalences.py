import pytest
import sympy as sp
from dataclasses import replace

from lbmpy.creationfunctions import create_lb_method, create_lb_collision_rule, LBMConfig, LBMOptimisation
from lbmpy.enums import Method, ForceModel, Stencil
from lbmpy.moments import (
    extract_monomials, get_default_moment_set_for_stencil, non_aliased_polynomial_raw_moments,
    exponent_tuple_sort_key)
from lbmpy.stencils import LBStencil

from lbmpy.methods.creationfunctions import create_central_moment
from lbmpy.methods.default_moment_sets import mrt_orthogonal_modes_literature
from lbmpy.maxwellian_equilibrium import get_weights

from lbmpy.moment_transforms import (
    PdfsToMomentsByMatrixTransform, PdfsToMomentsByChimeraTransform
)


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19])
@pytest.mark.parametrize('compressible', [True, False])
def test_full_and_delta_equilibrium_equivalence(stencil, compressible):
    stencil = LBStencil(stencil)
    zero_centered = True
    omega = sp.Symbol('omega')
    r_rates = (omega,) * stencil.Q

    rho = sp.Symbol("rho")
    rho_background = sp.Integer(1)
    delta_rho = sp.Symbol("delta_rho")
    
    subs = { delta_rho : rho - rho_background }
    eqs = []

    for delta_eq in [False, True]:
        method = create_central_moment(stencil, r_rates, continuous_equilibrium=True, compressible=compressible,
                    zero_centered=zero_centered, delta_equilibrium=delta_eq, equilibrium_order=None)
        eq = method.get_equilibrium_terms()
        eqs.append(eq.subs(subs))

    assert (eqs[0] - eqs[1]).expand() == sp.Matrix((0,) * stencil.Q)
    

@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19])
@pytest.mark.parametrize('compressible', [True, False])
@pytest.mark.parametrize('delta_eq', [True, False])
def test_zero_centering_equilibrium_equivalence(stencil, compressible, delta_eq):
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
        method = create_central_moment(stencil, r_rates, continuous_equilibrium=True, compressible=compressible,
                    zero_centered=zero_centered, delta_equilibrium=delta_eq and zero_centered,
                    equilibrium_order=None)
        eq = method.get_equilibrium_terms()
        eqs.append(eq.subs(subs))

    assert (eqs[0] - eqs[1]).expand() == weights
