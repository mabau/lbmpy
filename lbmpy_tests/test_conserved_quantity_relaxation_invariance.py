"""
The update equations should not change if a relaxation rate of a conserved quantity (density/velocity)
changes. This test checks that for moment-based methods
"""
from copy import copy

import pytest
import sympy as sp

from lbmpy.methods.creationfunctions import RelaxationInfo, create_srt, create_trt, create_trt_kbc, \
    create_with_default_polynomial_cumulants
from lbmpy.methods.momentbased.momentbasedmethod import MomentBasedLbMethod
from lbmpy.methods.centeredcumulant.centeredcumulantmethod import CenteredCumulantBasedLbMethod
from lbmpy.moments import MOMENT_SYMBOLS
from lbmpy.simplificationfactory import create_simplification_strategy
from lbmpy.stencils import get_stencil


def __change_relaxation_rate_of_conserved_moments(method, new_relaxation_rate=sp.Symbol("test_omega")):
    conserved_moments = (sp.Rational(1, 1),) + MOMENT_SYMBOLS[:method.dim]

    rr_dict = copy(method.relaxation_info_dict)
    for conserved_moment in conserved_moments:
        prev = rr_dict[conserved_moment]
        rr_dict[conserved_moment] = RelaxationInfo(prev.equilibrium_value, new_relaxation_rate)

    if isinstance(method, MomentBasedLbMethod):
        changed_method = MomentBasedLbMethod(method.stencil, rr_dict, method.conserved_quantity_computation,
                                             force_model=method.force_model)
    elif isinstance(method, CenteredCumulantBasedLbMethod):
        changed_method = CenteredCumulantBasedLbMethod(method.stencil, rr_dict, method.conserved_quantity_computation,
                                                       force_model=method.force_model)
    else:
        raise ValueError("Not a moment or cumulant-based method")

    return changed_method


def check_for_collision_rule_equivalence(collision_rule1, collision_rule2):
    collision_rule1 = collision_rule1.new_without_subexpressions()
    collision_rule2 = collision_rule2.new_without_subexpressions()
    for eq1, eq2 in zip(collision_rule1.main_assignments, collision_rule2.main_assignments):
        diff = sp.cancel(sp.expand(eq1.rhs - eq2.rhs))
        assert diff == 0


def check_method_equivalence(m1, m2, do_simplifications):
    cr1 = m1.get_collision_rule()
    cr2 = m2.get_collision_rule()
    if do_simplifications:
        cr1 = create_simplification_strategy(m1)(cr1)
        cr2 = create_simplification_strategy(m2)(cr2)
    check_for_collision_rule_equivalence(cr1, cr2)


@pytest.mark.longrun
def test_cumulant():
    for stencil_name in ('D2Q9',):
        stencil = get_stencil(stencil_name)
        original_method = create_with_default_polynomial_cumulants(stencil, [sp.Symbol("omega")])
        changed_method = __change_relaxation_rate_of_conserved_moments(original_method)

        check_method_equivalence(original_method, changed_method, True)
        check_method_equivalence(original_method, changed_method, False)


@pytest.mark.longrun
def test_srt():
    for stencil_name in ('D2Q9',):
        stencil = get_stencil(stencil_name)
        original_method = create_srt(stencil, sp.Symbol("omega"), compressible=True,
                                     maxwellian_moments=True)
        changed_method = __change_relaxation_rate_of_conserved_moments(original_method)

        check_method_equivalence(original_method, changed_method, True)
        check_method_equivalence(original_method, changed_method, False)


def test_srt_short():
    stencil = get_stencil("D2Q9")
    original_method = create_srt(stencil, sp.Symbol("omega"), compressible=True,
                                 maxwellian_moments=True)
    changed_method = __change_relaxation_rate_of_conserved_moments(original_method)

    check_method_equivalence(original_method, changed_method, True)
    check_method_equivalence(original_method, changed_method, False)


@pytest.mark.longrun
def test_trt():
    for stencil_name in ("D2Q9", "D3Q19", "D3Q27"):
        for continuous_moments in (False, True):
            stencil = get_stencil(stencil_name)
            original_method = create_trt(stencil, sp.Symbol("omega1"), sp.Symbol("omega2"),
                                         maxwellian_moments=continuous_moments)
            changed_method = __change_relaxation_rate_of_conserved_moments(original_method)

            check_method_equivalence(original_method, changed_method, True)
            check_method_equivalence(original_method, changed_method, False)


@pytest.mark.longrun
def test_trt_kbc_long():
    for dim in (2, 3):
        for method_name in ("KBC-N1", "KBC-N2", "KBC-N3", "KBC-N4"):
            original_method = create_trt_kbc(dim, sp.Symbol("omega1"), sp.Symbol("omega2"), method_name=method_name,
                                             maxwellian_moments=False)
            changed_method = __change_relaxation_rate_of_conserved_moments(original_method)
            check_method_equivalence(original_method, changed_method, True)
            check_method_equivalence(original_method, changed_method, False)


def test_trt_kbc_short():
    for dim, method_name in [(2, "KBC-N2")]:
        original_method = create_trt_kbc(dim, sp.Symbol("omega1"), sp.Symbol("omega2"), method_name=method_name,
                                         maxwellian_moments=False)
        changed_method = __change_relaxation_rate_of_conserved_moments(original_method)
        check_method_equivalence(original_method, changed_method, True)
        check_method_equivalence(original_method, changed_method, False)
