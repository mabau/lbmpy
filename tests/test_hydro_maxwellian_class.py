import pytest
import sympy as sp
from itertools import chain

from pystencils.sympyextensions import remove_higher_order_terms

from lbmpy.stencils import Stencil, LBStencil
from lbmpy.equilibrium import ContinuousHydrodynamicMaxwellian, DiscreteHydrodynamicMaxwellian
from lbmpy.moments import moments_up_to_component_order, moments_up_to_order
from lbmpy.maxwellian_equilibrium import get_equilibrium_values_of_maxwell_boltzmann_function

from lbmpy.methods.default_moment_sets import cascaded_moment_sets_literature, mrt_orthogonal_modes_literature


def test_compressible_raw_moment_values():
    stencil = LBStencil("D3Q27")
    equilibrium = ContinuousHydrodynamicMaxwellian(dim=stencil.D, compressible=True, deviation_only=False)

    raw_moments = list(chain.from_iterable(mrt_orthogonal_modes_literature(stencil, False)))

    values_a = equilibrium.moments(raw_moments)

    values_b = get_equilibrium_values_of_maxwell_boltzmann_function(raw_moments, stencil.D, space="moment")

    for m, a, b in zip(raw_moments, values_a, values_b):
        assert (a - b).expand() == sp.Integer(0), f"Mismatch at moment {m}."


def test_compressible_central_moment_values():
    stencil = LBStencil("D3Q27")
    equilibrium = ContinuousHydrodynamicMaxwellian(dim=stencil.D, compressible=True, deviation_only=False)

    central_moments = list(chain.from_iterable(cascaded_moment_sets_literature(stencil)))

    values_a = equilibrium.central_moments(central_moments)

    values_b = get_equilibrium_values_of_maxwell_boltzmann_function(central_moments, stencil.D, space="central moment")

    for m, a, b in zip(central_moments, values_a, values_b):
        assert (a - b).expand() == sp.Integer(0), f"Mismatch at moment {m}."


def test_compressible_cumulant_values():
    stencil = LBStencil("D3Q27")
    equilibrium = ContinuousHydrodynamicMaxwellian(dim=stencil.D, compressible=True, deviation_only=False)

    cumulants = list(chain.from_iterable(cascaded_moment_sets_literature(stencil)))

    values_a = equilibrium.cumulants(cumulants, rescale=False)

    values_b = get_equilibrium_values_of_maxwell_boltzmann_function(cumulants, stencil.D, space="cumulant")

    for m, a, b in zip(cumulants, values_a, values_b):
        assert (a - b).expand() == sp.Integer(0), f"Mismatch at cumulant {m}."


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('compressible', [False, True])
@pytest.mark.parametrize('deviation_only', [False, True])
def test_continuous_discrete_moment_equivalence(stencil, compressible, deviation_only):
    stencil = LBStencil(stencil)
    c_s_sq = sp.Rational(1, 3)
    moments = tuple(moments_up_to_order(3, dim=stencil.D, include_permutations=False))
    cd = ContinuousHydrodynamicMaxwellian(dim=stencil.D, compressible=compressible, deviation_only=deviation_only,
                                          order=2, c_s_sq=c_s_sq)
    cm = sp.Matrix(cd.moments(moments))
    dd = DiscreteHydrodynamicMaxwellian(stencil, compressible=compressible, deviation_only=deviation_only,
                                        order=2, c_s_sq=c_s_sq)
    dm = sp.Matrix(dd.moments(moments))

    rho = cd.density
    delta_rho = cd.density_deviation
    rho_0 = cd.background_density

    subs = { delta_rho : rho - rho_0 }

    assert (cm - dm).subs(subs).expand() == sp.Matrix((0,) * len(moments))


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('compressible', [False, True])
@pytest.mark.parametrize('deviation_only', [False, True])
def test_continuous_discrete_central_moment_equivalence(stencil, compressible, deviation_only):
    stencil = LBStencil(stencil)
    c_s_sq = sp.Rational(1, 3)
    moments = tuple(moments_up_to_order(3, dim=stencil.D, include_permutations=False))
    cd = ContinuousHydrodynamicMaxwellian(dim=stencil.D, compressible=compressible, deviation_only=deviation_only,
                                          order=2, c_s_sq=c_s_sq)
    cm = sp.Matrix(cd.central_moments(moments))
    dd = DiscreteHydrodynamicMaxwellian(stencil, compressible=compressible, deviation_only=deviation_only,
                                        order=2, c_s_sq=c_s_sq)
    dm = sp.Matrix(dd.central_moments(moments))
    dm = sp.Matrix([remove_higher_order_terms(t, dd.velocity, order=2) for t in dm])

    rho = cd.density
    delta_rho = cd.density_deviation
    rho_0 = cd.background_density

    subs = { delta_rho : rho - rho_0 }

    assert (cm - dm).subs(subs).expand() == sp.Matrix((0,) * len(moments))


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
def test_continuous_discrete_cumulant_equivalence(stencil):
    stencil = LBStencil(stencil)
    c_s_sq = sp.Rational(1, 3)
    compressible = True
    deviation_only = False
    moments = tuple(moments_up_to_order(3, dim=stencil.D, include_permutations=False))
    cd = ContinuousHydrodynamicMaxwellian(dim=stencil.D, compressible=compressible, deviation_only=deviation_only,
                                          order=2, c_s_sq=c_s_sq)
    cm = sp.Matrix(cd.cumulants(moments))
    dd = DiscreteHydrodynamicMaxwellian(stencil, compressible=compressible, deviation_only=deviation_only,
                                        order=2, c_s_sq=c_s_sq)
    dm = sp.Matrix(dd.cumulants(moments))
    dm = sp.Matrix([remove_higher_order_terms(t, dd.velocity, order=2) for t in dm])

    rho = cd.density
    delta_rho = cd.density_deviation
    rho_0 = cd.background_density

    subs = { delta_rho : rho - rho_0 }

    assert (cm - dm).subs(subs).expand() == sp.Matrix((0,) * len(moments))
