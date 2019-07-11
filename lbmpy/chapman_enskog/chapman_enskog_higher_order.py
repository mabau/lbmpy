from collections import OrderedDict, namedtuple

import sympy as sp

from lbmpy.chapman_enskog.chapman_enskog import CeMoment, Diff, expanded_symbol, take_moments
from lbmpy.moments import moments_of_order, moments_up_to_order
from pystencils.fd import expand_diff_full


def poly_moments(order, dim):
    from pystencils.sympyextensions import prod
    c = sp.Matrix([expanded_symbol("c", subscript=i) for i in range(dim)])
    return [prod(c_i ** m_i for c_i, m_i in zip(c, m)) for m in moments_of_order(order, dim=dim)]


def special_linsolve(eqs, degrees_of_freedom):
    """Adapted version of sympy's linsolve function.

    Sympy's linsolve can only solve for symbols, not expressions
    The general solve routine can, but is way slower. This function substitutes the expressions to solve for
    by dummy variables and then uses the fast linsolve from sympy
    """
    dummy_subs = {d: sp.Dummy() for d in degrees_of_freedom}
    dummy_subs_inverse = {dum: val for val, dum in dummy_subs.items()}
    eqs_with_dummies = [e.subs(dummy_subs) for e in eqs]
    eqs_to_solve = [eq for eq in eqs_with_dummies if eq.atoms(sp.Dummy)]
    assert eqs_to_solve
    dummy_list = list(dummy_subs.values())
    solve_result = sp.linsolve(eqs_to_solve, dummy_list)
    assert len(solve_result) == 1, "Solve Result length " + str(len(solve_result))
    solve_result = list(solve_result)[0]
    return {dummy_subs_inverse[dummy_list[i]]: solve_result[i] for i in range(len(solve_result))}


def get_solvability_conditions(dim, order):
    solvability_conditions = {}
    for name in ["\\Pi", "\\Upsilon"]:
        for moment_tuple in moments_up_to_order(1, dim=dim):
            for k in range(order + 1):
                solvability_conditions[CeMoment(name, superscript=k, moment_tuple=moment_tuple)] = 0
    return solvability_conditions


def determine_higher_order_moments(epsilon_hierarchy, relaxation_rates, moment_computation, dim, order=2):
    """Computes values of non-equilibrium moments of order 2 up to passed order.

    Args:
        epsilon_hierarchy: dict mapping epsilon exponent to equation.
                           Can be computed by :func:`chapman_enskog_ansatz`
        relaxation_rates: list of symbolic relaxation rates, which are treated as constants
        moment_computation: instance of LbMethodEqMoments, which computes equilibrium moments of a LB scheme
        dim: dimension
        order: moments up to this order are computed, has to be >= 2

    Returns:
        Tuple with
            - values of expanded time derivative objects
            - higher order moments in raw form, with time derivative terms
            - higher order moments where time derivative objects have been substituted
    """
    assert order >= 2
    solvability_conditions = get_solvability_conditions(dim, order)

    def full_expand(expr):
        return expand_diff_full(expr, constants=relaxation_rates)

    def tm(expr):
        expr = take_moments(expr, use_one_neighborhood_aliasing=True)
        return moment_computation.substitute(expr).subs(solvability_conditions)

    time_diffs = OrderedDict()
    non_eq_moms = OrderedDict()
    for eps_order in range(1, order):
        eps_eq = epsilon_hierarchy[eps_order]

        for order in range(order + 1):
            eqs = sp.Matrix([full_expand(tm(eps_eq * m)) for m in poly_moments(order, dim)])
            unknown_moments = [m for m in eqs.atoms(CeMoment)
                               if m.superscript == eps_order and sum(m.moment_tuple) == order]
            if len(unknown_moments) == 0:
                for eq in eqs:
                    t = sp.Symbol("t")
                    time_diffs_in_expr = [d for d in eq.atoms(Diff)
                                          if (d.target == 't' or d.target == t) and d.superscript == eps_order]
                    if len(time_diffs_in_expr) == 0:
                        continue
                    assert len(time_diffs_in_expr) == 1, \
                        "Time diffs in expr %d %s" % (len(time_diffs_in_expr), time_diffs_in_expr)
                    td = time_diffs_in_expr[0]
                    time_diffs[td] = sp.solve(eq, td)[0]
            else:
                solve_result = special_linsolve(eqs, unknown_moments)
                non_eq_moms.update(solve_result)

    substituted_non_eq_moms = OrderedDict()
    for key, value in non_eq_moms.items():
        value = full_expand(value)
        value = full_expand(value.subs(time_diffs)).expand()
        value = full_expand(value.subs(substituted_non_eq_moms)).expand()
        value = full_expand(value.subs(time_diffs)).expand()
        substituted_non_eq_moms[key] = value.expand()

    Result = namedtuple('HigherOrderMoments', ['time_diffs', 'non_eq_moments_raw', 'non_eq_moments'])
    return Result(time_diffs, non_eq_moms, substituted_non_eq_moms)
