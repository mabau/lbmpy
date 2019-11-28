"""
Moment-based methods are created by specifying moments and their equilibrium value.
This test checks if the equilibrium formula obtained by this method is the same as the explicitly
given discrete_maxwellian_equilibrium
"""
import sympy as sp

from lbmpy.creationfunctions import create_lb_method
from lbmpy.maxwellian_equilibrium import discrete_maxwellian_equilibrium
from lbmpy.methods import create_mrt_orthogonal, create_srt, create_trt, mrt_orthogonal_modes_literature
from lbmpy.relaxationrates import get_shear_relaxation_rate
from lbmpy.stencils import get_stencil


def check_for_matching_equilibrium(method_name, stencil, compressibility):
    omega = sp.Symbol("omega")
    if method_name == 'srt':
        method = create_srt(stencil, omega, compressible=compressibility, equilibrium_order=2)
    elif method_name == 'trt':
        method = create_trt(stencil, omega, omega, compressible=compressibility, equilibrium_order=2)
    elif method_name == 'mrt':
        method = create_mrt_orthogonal(stencil, lambda v: omega, compressible=compressibility, equilibrium_order=2)
    else:
        raise ValueError("Unknown method")

    reference_equilibrium = discrete_maxwellian_equilibrium(stencil, order=2,
                                                            c_s_sq=sp.Rational(1, 3), compressible=compressibility)
    equilibrium = method.get_equilibrium()
    equilibrium = equilibrium.new_without_subexpressions(subexpressions_to_keep=sp.symbols("rho u_0 u_1 u_2"))

    diff = sp.Matrix(reference_equilibrium) - sp.Matrix([eq.rhs for eq in equilibrium.main_assignments])
    diff = sp.simplify(diff)
    assert diff.is_zero


def check_for_matching_equilibrium_for_stencil(stencil_name):
    stencil = get_stencil(stencil_name)
    for method in ['srt', 'trt', 'mrt']:
        check_for_matching_equilibrium(method, stencil, True)
        check_for_matching_equilibrium(method, stencil, False)


def test_d2_q9():
    check_for_matching_equilibrium_for_stencil('D2Q9')


def test_d3_q27():
    check_for_matching_equilibrium_for_stencil('D3Q27')


def test_d3_q19():
    check_for_matching_equilibrium_for_stencil('D3Q19')


def test_d3_q15():
    check_for_matching_equilibrium_for_stencil('D3Q15')


def test_relaxation_rate_setter():
    o1, o2, o3 = sp.symbols("o1 o2 o3")
    method = create_lb_method(method='srt', stencil='D2Q9', relaxation_rates=[o3])
    method2 = create_lb_method(method='mrt3', stencil='D2Q9', relaxation_rates=[o3, o3, o3])
    method.set_zeroth_moment_relaxation_rate(o1)
    method.set_first_moment_relaxation_rate(o2)
    assert get_shear_relaxation_rate(method) == o3
    method.set_zeroth_moment_relaxation_rate(o3)
    method.set_first_moment_relaxation_rate(o3)
    assert method.collision_matrix == method2.collision_matrix


def test_mrt_orthogonal():
    m = create_mrt_orthogonal(get_stencil("D2Q9"), maxwellian_moments=True, weighted=False)
    assert m.is_orthogonal

    m = create_mrt_orthogonal(get_stencil("D2Q9"), maxwellian_moments=True, weighted=True)
    assert m.is_weighted_orthogonal

    m = create_mrt_orthogonal(get_stencil("D3Q19"), maxwellian_moments=True, weighted=False)
    assert m.is_orthogonal

    m = create_mrt_orthogonal(get_stencil("D3Q19"), maxwellian_moments=True, weighted=True)
    assert m.is_weighted_orthogonal

    moments = mrt_orthogonal_modes_literature(get_stencil("D3Q15"), True, False)
    m = create_mrt_orthogonal(get_stencil("D3Q15"), maxwellian_moments=True, nested_moments=moments)
    assert m.is_weighted_orthogonal

    moments = mrt_orthogonal_modes_literature(get_stencil("D3Q19"), True, False)
    m = create_mrt_orthogonal(get_stencil("D3Q19"), maxwellian_moments=True, nested_moments=moments)
    assert m.is_weighted_orthogonal

    moments = mrt_orthogonal_modes_literature(get_stencil("D3Q27"), False, False)
    m = create_mrt_orthogonal(get_stencil("D3Q27"), maxwellian_moments=True, nested_moments=moments)
    assert m.is_orthogonal
