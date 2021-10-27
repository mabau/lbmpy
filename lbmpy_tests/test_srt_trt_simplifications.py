"""
This unittest checks the simplification quality of SRT and TRT (compressible and incompressible) against
known acceptable values.
"""
import sympy as sp

from lbmpy.enums import Stencil
from lbmpy.forcemodels import Luo
from lbmpy.methods import create_srt, create_trt, create_trt_with_magic_number
from lbmpy.methods.momentbased.momentbasedsimplifications import cse_in_opposing_directions
from lbmpy.simplificationfactory import create_simplification_strategy
from lbmpy.stencils import LBStencil


def check_method(method, limits_default, limits_cse):
    strategy = create_simplification_strategy(method)
    strategy_with_cse = create_simplification_strategy(method)
    strategy_with_cse.add(cse_in_opposing_directions)
    collision_rule = method.get_collision_rule()

    ops_default = strategy(collision_rule).operation_count
    ops_cse = strategy_with_cse(collision_rule).operation_count

    assert ops_default['adds'] <= limits_default[0]
    assert ops_default['muls'] <= limits_default[1]
    assert ops_default['divs'] <= limits_default[2]

    assert ops_cse['adds'] <= limits_cse[0]
    assert ops_cse['muls'] <= limits_cse[1]
    assert ops_cse['divs'] <= limits_cse[2]


def test_simplifications_srt_d2q9_incompressible():
    omega = sp.symbols('omega')
    method = create_srt(LBStencil(Stencil.D2Q9), omega, compressible=False,
                        equilibrium_order=2, moment_transform_class=None)
    check_method(method, [53, 46, 0], [49, 30, 0])


def test_simplifications_srt_d2q9_compressible():
    omega = sp.symbols('omega')
    method = create_srt(LBStencil(Stencil.D2Q9), omega, compressible=True,
                        equilibrium_order=2, moment_transform_class=None)
    check_method(method, [53, 58, 1], [53, 42, 1])


def test_simplifications_trt_d2q9_incompressible():
    o1, o2 = sp.symbols("omega_1 omega_2")
    method = create_trt(LBStencil(Stencil.D2Q9), o1, o2, compressible=False)
    check_method(method, [77, 86, 0], [65, 38, 0])


def test_simplifications_trt_d2q9_compressible():
    o1, o2 = sp.symbols("omega_1 omega_2")
    method = create_trt(LBStencil(Stencil.D2Q9), o1, o2, compressible=True)
    check_method(method, [77, 106, 1], [65, 56, 1])


def test_simplifications_trt_d3q19_force_incompressible():
    o1, o2 = sp.symbols("omega_1 omega_2")
    force_model = Luo([sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(1, 5)])
    method = create_trt(LBStencil(Stencil.D3Q19), o1, o2, compressible=False, force_model=force_model)
    check_method(method, [268, 281, 0], [241, 175, 1])


def test_simplifications_trt_d3q19_force_compressible():
    o1, o2 = sp.symbols("omega_1 omega_2")
    force_model = Luo([sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(1, 5)])
    method = create_trt_with_magic_number(LBStencil(Stencil.D3Q19), o1, compressible=False, force_model=force_model)
    check_method(method, [270, 284, 1], [243, 178, 1])
