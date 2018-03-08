from pystencils.derivative import *


def chapmanEnskogDerivativeExpansion(expr, label, eps=sp.Symbol("epsilon"), startOrder=1, stopOrder=4):
    """Substitutes differentials with given label and no ceIdx by the sum:
    eps**(startOrder) * Diff(ceIdx=startOrder)   + ... + eps**(stopOrder) * Diff(ceIdx=stopOrder)"""
    diffs = [d for d in expr.atoms(Diff) if d.label == label]
    subsDict = {d: sum([eps ** i * Diff(d.arg, d.label, i) for i in range(startOrder, stopOrder)])
                for d in diffs}
    return expr.subs(subsDict)


def chapmanEnskogDerivativeRecombination(expr, label, eps=sp.Symbol("epsilon"), startOrder=1, stopOrder=4):
    """Inverse operation of 'chapmanEnskogDerivativeExpansion'"""
    expr = expr.expand()
    diffs = [d for d in expr.atoms(Diff) if d.label == label and d.ceIdx == stopOrder - 1]
    for d in diffs:
        substitution = Diff(d.arg, label)
        substitution -= sum([eps ** i * Diff(d.arg, label, i) for i in range(startOrder, stopOrder - 1)])
        expr = expr.subs(d, substitution / eps**(stopOrder-1))
    return expr