import functools

import pytest
import sympy as sp

from lbmpy.chapman_enskog.chapman_enskog import (
    ChapmanEnskogAnalysis, LbMethodEqMoments, chapman_enskog_ansatz,
    get_taylor_expanded_lb_equation, take_moments)
from lbmpy.chapman_enskog.chapman_enskog_higher_order import (
    determine_higher_order_moments, get_solvability_conditions)
from lbmpy.chapman_enskog.chapman_enskog_steady_state import (
    SteadyStateChapmanEnskogAnalysis, SteadyStateChapmanEnskogAnalysisSRT)
from lbmpy.creationfunctions import create_lb_method, LBMConfig
from lbmpy.enums import Method, Stencil
from lbmpy.forcemodels import Guo
from lbmpy.relaxationrates import lattice_viscosity_from_relaxation_rate
from lbmpy.stencils import LBStencil
from pystencils.fd import Diff, normalize_diff_order
from pystencils.sympyextensions import multidimensional_sum


@pytest.mark.parametrize('continuous_eq', [False, True])
@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
def test_srt(continuous_eq, stencil):
    omega = sp.Symbol("omega")
    print(f"Analysing {stencil}, ContMaxwellianConstruction {continuous_eq}")

    lbm_config = LBMConfig(stencil=LBStencil(stencil), method=Method.SRT, compressible=True,
                           zero_centered=False, relaxation_rate=omega, continuous_equilibrium=continuous_eq)
    method = create_lb_method(lbm_config=lbm_config)
    analysis = ChapmanEnskogAnalysis(method)
    omega_value = analysis.relaxation_rate_from_kinematic_viscosity(1)[omega]
    assert omega_value, sp.Rational(2 == 7)


@pytest.mark.longrun
def test_steady_state_silva_paper_comparison():
    eps, tau, lambda_plus, f = sp.symbols("epsilon tau Lambda f")

    lbm_config = LBMConfig(stencil=LBStencil(Stencil.D3Q19), compressible=False, relaxation_rate=1 / tau,
                           continuous_equilibrium=False, zero_centered=False)
    method = create_lb_method(lbm_config=lbm_config)
    analysis = SteadyStateChapmanEnskogAnalysis(method)

    dim = 3
    dt = 1
    expanded_pdf_symbols = analysis.f_syms
    feq = expanded_pdf_symbols[0]
    c = analysis.velocity_syms
    lamb = sp.Symbol("Lambda")

    r = sp.Rational

    def d(arg, *args):
        """Shortcut to create nested derivatives"""
        assert arg is not None
        args = sorted(args, reverse=True, key=lambda e: e.name if isinstance(e, sp.Symbol) else e)
        res = arg
        for i in args:
            res = Diff(res, i)
        return res

    s = functools.partial(multidimensional_sum, dim=dim)

    rho = sp.Symbol("rho")
    u = sp.symbols("u_:3")[:dim]

    ref_fs = [
        # f^0, Eq.17a
        feq,
        # f_1 Eq.17b
        - tau * dt * sum(c[i] * d(feq, i) for i, in s(1)),
        # f_2 Eq.17c
        tau * lambda_plus * dt ** 2 * sum(c[i] * c[j] * d(feq, i, j) for i, j in s(2)),
        # f_3 Eq.17d
        -tau * (lambda_plus ** 2 - r(1, 12)) * dt ** 3 * \
        sum(c[i] * c[j] * c[k] * d(feq, i, j, k) for i, j, k in s(3)),
        # f_4 Eq.17e
        tau * lambda_plus * (lambda_plus ** 2 - r(1, 6)) * dt ** 4 * \
        sum(c[i] * c[j] * c[k] * c[l] * d(feq, i, j, k, l) for i, j, k, l in s(4)),
    ]

    def reference_cont_eq(a1=0, a2=1, order_range=(0, 4)):
        by_order = {
            0: sum(d(u[i], i) for i, in s(1)),
            1: - dt * lambda_plus * (
                sum(d(u[i] * u[j], i, j) for i, j in s(2)) + sum(d(rho, i, i) / 3 for i, in s(1))),
            2: dt ** 2 * (lambda_plus ** 2 - r(1, 12)) * sum(d(u[i], i, j, j) for i, j in s(2)),
            3: -dt ** 3 * lambda_plus * (lambda_plus ** 2 - r(1, 6)) * (
                sum(d(rho, i, i, j, j) / 3 for i, j in s(2)) +
                a1 * sum(d(u[k] * u[k], i, i, j, j) for i, j, k in s(3)) +
                a2 * sum(d(u[j] * u[k], i, i, j, k) for i, j, k in s(3)))

        }
        return sum(by_order[i] for i in range(*order_range))

    def reference_mom_eq(h=0, stencil_name="D3Q19", order_range=(0, 4)):
        coefficients = {
            "D3Q15": (0, 7, -6, r(1, 3), r(8, 3), r(-8, 3)),
            "D3Q19": (0, -r(7, 2), r(9, 2), r(1, 3), -r(4, 3), r(5, 3)),
            "D3Q27": (0, 0, 1, r(1, 3), 0, r(1, 3)),
        }
        b1, b2, b3, c1, c2, c3 = coefficients[stencil_name]
        by_order = {
            0: d(rho, h) / 3 + sum(d(u[i] * u[h], i) for i, in s(1)),
            1: -r(1, 3) * dt * lamb * (sum(d(u[h], i, i) for i, in s(1)) + 2 * sum(d(u[i], i, h) for i, in s(1))),
            2: dt ** 2 * (lamb ** 2 - r(1, 12)) * (sum(d(rho, i, i, h) / 3 for i, h in s(2)) +
                                                   b1 * sum(d(u[k] * u[k], i, i, h) for i, k in s(2)) +
                                                   b2 * sum(d(u[j] * u[h], i, i, j) for i, j in s(2)) +
                                                   b3 * sum(d(u[i] * u[j], i, j, h) for i, j in s(2))),
            3: -dt ** 3 * lamb * (lamb ** 2 - r(1, 6)) * (c1 * sum(d(u[h], i, i, j, j) for i, j in s(2)) +
                                                          c1 * sum(d(u[k], k, j, j, h) for j, k in s(2)) +
                                                          c2 * sum(d(u[j], i, i, j, h) for i, j in s(2)) +
                                                          c3 * sum(d(u[i], i, j, j, h) for i, j in s(2)))

        }
        result = sum(by_order[i] for i in range(*order_range))
        return result

    # Check scale hierarchy - Eq.17 in Silva Paper
    for f_idx in range(1, 5):
        print("Checking f_idx", f_idx)
        ref = ref_fs[f_idx].subs(lamb, tau - r(1, 2))
        diff = analysis.pdf_hierarchy[f_idx].subs(analysis.collision_op_sym, 1 / tau) - ref
        diff = diff.expand().subs(analysis.force_sym, 0)
        diff = normalize_diff_order(diff)
        assert diff == 0

    # Check continuity equation
    for order in range(0, 3):
        print("Checking continuity order", order)
        reference = reference_cont_eq(order_range=(order, order + 1)).subs(lamb, tau - r(1, 2))
        diff = reference + analysis.get_continuity_equation(order) / tau
        diff = normalize_diff_order(diff)
        assert diff.expand() == 0

    # Check momentum transport equation
    for order in range(0, 2):
        print("Checking momentum order", order)
        coord = 0
        reference = reference_mom_eq(coord, order_range=(order, order + 1)).subs(lamb, tau - r(1, 2))
        diff = reference + analysis.get_momentum_equation(only_order=order)[coord] / tau
        diff = normalize_diff_order(diff)
        assert diff.expand() == 0


def test_higher_order_moment_computation():
    """In chapman_enskog_higher_order.py there are some functions to generalize the std Chapman Enskog expansion
    These are not used by the Chapman Enskog class yet."""
    method = create_lb_method(lbm_config=LBMConfig(stencil=LBStencil(Stencil.D2Q9), zero_centered=False,
                                                   method=Method.TRT, compressible=False))
    mom_comp = LbMethodEqMoments(method)
    dim = method.dim
    order = 2

    taylored_lb_eq = get_taylor_expanded_lb_equation(taylor_order=order, dim=dim, shift=True)
    eps_dict = chapman_enskog_ansatz(taylored_lb_eq,
                                     time_derivative_orders=(1, 3),
                                     spatial_derivative_orders=(1, 2),
                                     pdfs=(['f', 0, order + 1], ['\\Omega f', 1, order + 1]))
    higher_order_moments = determine_higher_order_moments(eps_dict, method.relaxation_rates, mom_comp,
                                                          dim, order=order)[2]
    solvability_conditions = get_solvability_conditions(dim=method.dim, order=order)
    continuity_eq = mom_comp.substitute(take_moments(eps_dict[1])).subs(solvability_conditions)

    u = sp.symbols("u_:2")
    rho, t = sp.symbols("rho t")
    assert continuity_eq == Diff(rho, t, 1) + Diff(u[0], 0, 1) + Diff(u[1], 1, 1)

    std_ce_analysis = ChapmanEnskogAnalysis(method)
    for k, v in std_ce_analysis.higher_order_moments.items():
        assert sp.expand(higher_order_moments[k] - v) == 0


def test_steady_state():
    rr = sp.symbols("omega")
    method = create_lb_method(lbm_config=LBMConfig(stencil=LBStencil(Stencil.D2Q9),
                                                   method=Method.SRT, relaxation_rate=rr))
    a1 = SteadyStateChapmanEnskogAnalysis(method, order=2)
    a2 = SteadyStateChapmanEnskogAnalysisSRT(method, order=2)

    a1.get_pdf_hierarchy(0)

    def compare(eq1, eq2):
        eq1 = (eq1 * -rr).subs(sp.symbols("epsilon"), 1)
        assert sp.expand(eq1 - eq2) == 0

    compare(a1.get_continuity_equation(), a2.get_continuity_equation(0) + a2.get_continuity_equation(1))
    for d in range(2):
        compare(a1.get_momentum_equation()[d],
                a2.get_momentum_equation(d, 0) + a2.get_momentum_equation(d, 1))

    viscosities = a2.determine_viscosities(0)
    print("viscosities", viscosities)
    nu = sp.symbols("nu")
    assert sp.cancel(viscosities[nu] - lattice_viscosity_from_relaxation_rate(rr)) == 0

    with_force = SteadyStateChapmanEnskogAnalysis(method, order=2, force_model_class=Guo)
    momentum_eq_with_force = sp.expand(with_force.get_momentum_equation(0)[0] * rr)
    momentum_eq_without_force = sp.expand(a1.get_momentum_equation(0)[0] * rr)
    assert momentum_eq_with_force - sp.symbols("a_0", commutative=False) == momentum_eq_without_force
