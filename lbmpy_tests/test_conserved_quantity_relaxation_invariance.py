"""
The update equations should not change if a relaxation rate of a conserved quantity (density/velocity)
changes. This test checks that for moment-based methods
"""
from copy import copy

import pytest
import sympy as sp
import math

from lbmpy.enums import Stencil, Method
from lbmpy.methods import create_srt, create_trt, create_trt_kbc, \
    create_with_default_polynomial_cumulants
from lbmpy.methods.momentbased.momentbasedmethod import MomentBasedLbMethod
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod
from lbmpy.moments import MOMENT_SYMBOLS
from lbmpy.simplificationfactory import create_simplification_strategy
from lbmpy.stencils import LBStencil


def __change_relaxation_rate_of_conserved_moments(method, new_relaxation_rate=sp.Symbol("test_omega")):
    conserved_moments = (sp.Rational(1, 1),) + MOMENT_SYMBOLS[:method.dim]

    rr_dict = copy(method.relaxation_rate_dict)
    for conserved_moment in conserved_moments:
        rr_dict[conserved_moment] = new_relaxation_rate

    if isinstance(method, MomentBasedLbMethod):
        changed_method = MomentBasedLbMethod(method.stencil, method.equilibrium_distribution, rr_dict, 
                                             method.conserved_quantity_computation,
                                             force_model=method.force_model)
    elif isinstance(method, CumulantBasedLbMethod):
        changed_method = CumulantBasedLbMethod(method.stencil, method.equilibrium_distribution, rr_dict,
                                                       method.conserved_quantity_computation,
                                                       force_model=method.force_model,
                                                       zero_centered=True)
    else:
        raise ValueError("Not a moment or cumulant-based method")

    return changed_method


def check_for_collision_rule_equivalence(collision_rule1, collision_rule2, use_numeric_subs=False):
    collision_rule1 = collision_rule1.new_without_subexpressions()
    collision_rule2 = collision_rule2.new_without_subexpressions()

    if use_numeric_subs:
        free_symbols = collision_rule1.free_symbols
        free_symbols.update(collision_rule2.free_symbols)

        subs_dict = dict()
        value = 10.0
        for symbol in free_symbols:
            subs_dict.update({symbol: value})
            value += 1.1

        collision_rule1 = collision_rule1.subs(subs_dict)
        collision_rule2 = collision_rule2.subs(subs_dict)

    for eq1, eq2 in zip(collision_rule1.main_assignments, collision_rule2.main_assignments):
        diff = sp.cancel(sp.expand(eq1.rhs - eq2.rhs))
        if use_numeric_subs:
            assert math.isclose(diff, 0, rel_tol=0.0, abs_tol=1e-10)
        else:
            assert diff == 0


def check_method_equivalence(m1, m2, do_simplifications, use_numeric_subs=False):
    cr1 = m1.get_collision_rule()
    cr2 = m2.get_collision_rule()
    if do_simplifications:
        cr1 = create_simplification_strategy(m1)(cr1)
        cr2 = create_simplification_strategy(m2)(cr2)
    check_for_collision_rule_equivalence(cr1, cr2, use_numeric_subs)


@pytest.mark.longrun
def test_cumulant():
    stencil = LBStencil(Stencil.D2Q9)
    original_method = create_with_default_polynomial_cumulants(stencil, [sp.Symbol("omega")], zero_centered=True)
    changed_method = __change_relaxation_rate_of_conserved_moments(original_method)

    check_method_equivalence(original_method, changed_method, True, use_numeric_subs=True)
    check_method_equivalence(original_method, changed_method, False, use_numeric_subs=True)


@pytest.mark.longrun
def test_srt():
    stencil = LBStencil(Stencil.D3Q27)
    original_method = create_srt(stencil, sp.Symbol("omega"), compressible=True,
                                 continuous_equilibrium=True)
    changed_method = __change_relaxation_rate_of_conserved_moments(original_method)

    check_method_equivalence(original_method, changed_method, True, use_numeric_subs=True)
    check_method_equivalence(original_method, changed_method, False, use_numeric_subs=True)


def test_srt_short():
    stencil = LBStencil(Stencil.D2Q9)
    original_method = create_srt(stencil, sp.Symbol("omega"), compressible=True,
                                 continuous_equilibrium=True)
    changed_method = __change_relaxation_rate_of_conserved_moments(original_method)

    check_method_equivalence(original_method, changed_method, True, use_numeric_subs=False)
    check_method_equivalence(original_method, changed_method, False, use_numeric_subs=False)


@pytest.mark.parametrize('stencil_name', [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('continuous_moments', [False, True])
@pytest.mark.longrun
def test_trt(stencil_name, continuous_moments):
    stencil = LBStencil(stencil_name)
    original_method = create_trt(stencil, sp.Symbol("omega1"), sp.Symbol("omega2"),
                                 continuous_equilibrium=continuous_moments)
    changed_method = __change_relaxation_rate_of_conserved_moments(original_method)

    check_method_equivalence(original_method, changed_method, True)
    check_method_equivalence(original_method, changed_method, False)


@pytest.mark.parametrize('method_name', [Method.TRT_KBC_N1, Method.TRT_KBC_N2, Method.TRT_KBC_N3, Method.TRT_KBC_N4])
def test_trt_kbc(method_name):
    dim = 2
    method_nr = method_name.name[-1]
    original_method = create_trt_kbc(dim, sp.Symbol("omega1"), sp.Symbol("omega2"),
                                     method_name='KBC-N' + method_nr,
                                     continuous_equilibrium=False)
    changed_method = __change_relaxation_rate_of_conserved_moments(original_method)
    check_method_equivalence(original_method, changed_method, True)
    check_method_equivalence(original_method, changed_method, False)


@pytest.mark.parametrize('method_name', [Method.TRT_KBC_N1, Method.TRT_KBC_N2, Method.TRT_KBC_N3, Method.TRT_KBC_N4])
@pytest.mark.longrun
def test_trt_kbc_long(method_name):
    dim = 3
    method_nr = method_name.name[-1]
    original_method = create_trt_kbc(dim, sp.Symbol("omega1"), sp.Symbol("omega2"),
                                     method_name='KBC-N' + method_nr,
                                     continuous_equilibrium=False)
    changed_method = __change_relaxation_rate_of_conserved_moments(original_method)
    check_method_equivalence(original_method, changed_method, True)
    check_method_equivalence(original_method, changed_method, False)
