import pytest

from lbmpy.cumulants import raw_moment_as_function_of_cumulants
from lbmpy.enums import Stencil
from lbmpy.maxwellian_equilibrium import *
from lbmpy.moments import (
    MOMENT_SYMBOLS, exponents_to_polynomial_representations, moment_matrix,
    moments_up_to_component_order, moments_up_to_order)
from lbmpy.stencils import LBStencil
from pystencils.sympyextensions import remove_higher_order_terms


def test_maxwellian_moments():
    """Check moments of continuous Maxwellian"""
    rho = sp.Symbol("rho")
    u = sp.symbols("u_0 u_1 u_2")
    c_s = sp.Symbol("c_s")
    eq_moments = get_equilibrium_values_of_maxwell_boltzmann_function(((0, 0, 0), (0, 0, 1)),
                                                                      dim=3, rho=rho, u=u, c_s_sq=c_s ** 2,
                                                                      space="moment")
    assert eq_moments[0] == rho
    assert eq_moments[1] == rho * u[2]

    x, y, z = MOMENT_SYMBOLS
    one = sp.Rational(1, 1)
    eq_moments = get_equilibrium_values_of_maxwell_boltzmann_function((one, x, x ** 2, x * y),
                                                                      dim=2, rho=rho, u=u[:2], c_s_sq=c_s ** 2,
                                                                      space="moment")
    assert eq_moments[0] == rho
    assert eq_moments[1] == rho * u[0]
    assert eq_moments[2] == rho * (c_s ** 2 + u[0] ** 2)
    assert eq_moments[3] == rho * u[0] * u[1]


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
def test_continuous_discrete_moment_equivalence(stencil):
    """Check that moments up to order 3 agree with moments of the continuous Maxwellian"""
    stencil = LBStencil(stencil)
    c_s_sq = sp.Rational(1, 3)
    moments = tuple(moments_up_to_order(3, dim=stencil.D, include_permutations=False))
    cm = sp.Matrix(get_equilibrium_values_of_maxwell_boltzmann_function(moments, order=2, dim=stencil.D,
                                                                        c_s_sq=c_s_sq, space="moment"))
    dm = sp.Matrix(get_moments_of_discrete_maxwellian_equilibrium(stencil, moments, order=2,
                                                                  compressible=True, c_s_sq=c_s_sq))

    diff = sp.simplify(cm - dm)
    for d in diff:
        assert d == 0


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q27])
def test_moment_cumulant_continuous_equivalence(stencil):
    """Test that discrete equilibrium is the same up to order 3 when obtained with following methods

    * eq1: take moments of continuous Maxwellian and transform back to pdf space
    * eq2: take cumulants of continuous Maxwellian, transform to moments then transform to pdf space
    * eq3: take discrete equilibrium from LBM literature
    * eq4: same as eq1 but with built-in function
    """
    stencil = LBStencil(stencil)
    u = sp.symbols(f"u_:{stencil.D}")
    indices = tuple(moments_up_to_component_order(2, dim=stencil.D))
    c_s_sq = sp.Rational(1, 3)
    eq_moments1 = get_equilibrium_values_of_maxwell_boltzmann_function(indices, dim=stencil.D, u=u, c_s_sq=c_s_sq,
                                                                       space="moment")
    eq_cumulants = get_equilibrium_values_of_maxwell_boltzmann_function(indices, dim=stencil.D, u=u, c_s_sq=c_s_sq,
                                                                        space="cumulant")
    eq_cumulants = {idx: c for idx, c in zip(indices, eq_cumulants)}
    eq_moments2 = [raw_moment_as_function_of_cumulants(idx, eq_cumulants) for idx in indices]
    pdfs_to_moments = moment_matrix(indices, stencil)

    def normalize(expressions):
        return [remove_higher_order_terms(e.expand(), symbols=u, order=3) for e in expressions]

    eq1 = normalize(pdfs_to_moments.inv() * sp.Matrix(eq_moments1))
    eq2 = normalize(pdfs_to_moments.inv() * sp.Matrix(eq_moments2))
    eq3 = normalize(discrete_maxwellian_equilibrium(stencil, order=3, c_s_sq=c_s_sq, compressible=True))
    eq4 = normalize(generate_equilibrium_by_matching_moments(stencil, indices, c_s_sq=c_s_sq))

    assert eq1 == eq2
    assert eq2 == eq3
    assert eq3 == eq4


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q27])
def test_moment_cumulant_continuous_equivalence_polynomial_formulation(stencil):
    """Same as test above, but instead of index tuples, the polynomial formulation is used."""
    stencil = LBStencil(stencil)
    u = sp.symbols(f"u_:{stencil.D}")
    index_tuples = tuple(moments_up_to_component_order(2, dim=stencil.D))
    indices = exponents_to_polynomial_representations(index_tuples)
    c_s_sq = sp.Rational(1, 3)
    eq_moments1 = get_equilibrium_values_of_maxwell_boltzmann_function(indices, dim=stencil.D,
                                                                       u=u, c_s_sq=c_s_sq, space="moment")
    eq_cumulants = get_equilibrium_values_of_maxwell_boltzmann_function(indices, dim=stencil.D,
                                                                        u=u, c_s_sq=c_s_sq, space="cumulant")
    eq_cumulants = {idx: c for idx, c in zip(index_tuples, eq_cumulants)}
    eq_moments2 = [raw_moment_as_function_of_cumulants(idx, eq_cumulants) for idx in index_tuples]
    pdfs_to_moments = moment_matrix(indices, stencil)

    def normalize(expressions):
        return [remove_higher_order_terms(e.expand(), symbols=u, order=3) for e in expressions]

    eq1 = normalize(pdfs_to_moments.inv() * sp.Matrix(eq_moments1))
    eq2 = normalize(pdfs_to_moments.inv() * sp.Matrix(eq_moments2))
    eq3 = normalize(discrete_maxwellian_equilibrium(stencil, order=3, c_s_sq=c_s_sq, compressible=True))
    eq4 = normalize(generate_equilibrium_by_matching_moments(stencil, indices, c_s_sq=c_s_sq))

    assert eq1 == eq2
    assert eq2 == eq3
    assert eq3 == eq4
