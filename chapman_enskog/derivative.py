from pystencils.derivative import *


def chapmanEnskogDerivativeExpansion(expr, label, eps=sp.Symbol("epsilon"), startOrder=1, stopOrder=4):
    """Substitutes differentials with given target and no superscript by the sum:
    eps**(startOrder) * Diff(superscript=startOrder)   + ... + eps**(stopOrder) * Diff(superscript=stopOrder)"""
    diffs = [d for d in expr.atoms(Diff) if d.target == label]
    subsDict = {d: sum([eps ** i * Diff(d.arg, d.target, i) for i in range(startOrder, stopOrder)])
                for d in diffs}
    return expr.subs(subsDict)


def chapmanEnskogDerivativeRecombination(expr, label, eps=sp.Symbol("epsilon"), startOrder=1, stopOrder=4):
    """Inverse operation of 'chapmanEnskogDerivativeExpansion'"""
    expr = expr.expand()
    diffs = [d for d in expr.atoms(Diff) if d.target == label and d.superscript == stopOrder - 1]
    for d in diffs:
        substitution = Diff(d.arg, label)
        substitution -= sum([eps ** i * Diff(d.arg, label, i) for i in range(startOrder, stopOrder - 1)])
        expr = expr.subs(d, substitution / eps**(stopOrder-1))
    return expr