import sympy as sp

from pystencils.fd import Diff


def chapman_enskog_derivative_expansion(expr, label, eps=sp.Symbol("epsilon"), start_order=1, stop_order=4):
    """Substitutes differentials with given target and no superscript by the sum:
    eps**(start_order) * Diff(superscript=start_order)   + ... + eps**(stop_order) * Diff(superscript=stop_order)"""
    diffs = [d for d in expr.atoms(Diff) if d.target == label]
    subs_dict = {d: sum([eps ** i * Diff(d.arg, d.target, i) for i in range(start_order, stop_order)])
                 for d in diffs}
    return expr.subs(subs_dict)


def chapman_enskog_derivative_recombination(expr, label, eps=sp.Symbol("epsilon"), start_order=1, stop_order=4):
    """Inverse operation of 'chapman_enskog_derivative_expansion'"""
    expr = expr.expand()
    diffs = [d for d in expr.atoms(Diff) if d.target == label and d.superscript == stop_order - 1]
    for d in diffs:
        substitution = Diff(d.arg, label)
        substitution -= sum([eps ** i * Diff(d.arg, label, i) for i in range(start_order, stop_order - 1)])
        expr = expr.subs(d, substitution / eps ** (stop_order - 1))
    return expr
