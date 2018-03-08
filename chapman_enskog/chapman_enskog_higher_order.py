import sympy as sp
from lbmpy.chapman_enskog import CeMoment, Diff, takeMoments
from lbmpy.chapman_enskog.derivative import fullDiffExpand
from lbmpy.chapman_enskog.chapman_enskog import expandedSymbol
from lbmpy.moments import momentsUpToOrder, momentsOfOrder
from collections import OrderedDict


def polyMoments(order, dim):
    from pystencils.sympyextensions import prod
    c = sp.Matrix([expandedSymbol("c", subscript=i) for i in range(dim)])
    return [prod(c_i ** m_i for c_i, m_i in zip(c, m)) for m in momentsOfOrder(order, dim=dim)]


def specialLinSolve(eqs, dofs):
    """Sympy's linsolve can only solve for symbols, not expressions
    The general solve routine can, but is way slower. This function substitutes the expressions to solve for
    by dummy variables and the uses the fast linsolve from sympy"""
    dummySubs = {d: sp.Dummy() for d in dofs}
    dummySubsInverse = {dum: val for val, dum in dummySubs.items()}
    eqsWithDummies = [e.subs(dummySubs) for e in eqs]
    eqsToSolve = [eq for eq in eqsWithDummies if eq.atoms(sp.Dummy)]
    assert eqsToSolve
    dummyList = list(dummySubs.values())
    solveResult = sp.linsolve(eqsToSolve, dummyList)
    assert len(solveResult) == 1, "Solve Result length " + str(len(solveResult))
    solveResult = list(solveResult)[0]
    return {dummySubsInverse[dummyList[i]]: solveResult[i] for i in range(len(solveResult))}


def getSolvabilityConditions(dim, order):
    solvabilityConditions = {}
    for name in ["\Pi", "\\Upsilon"]:
        for momentTuple in momentsUpToOrder(1, dim=dim):
            for k in range(order+1):
                solvabilityConditions[CeMoment(name, superscript=k, momentTuple=momentTuple)] = 0
    return solvabilityConditions


def determineHigherOrderMoments(epsilonHierarchy, relaxationRates, momentComputation, dim, order=2):
    solvabilityConditions = getSolvabilityConditions(dim, order)

    def fullExpand(expr):
        return fullDiffExpand(expr, constants=relaxationRates)

    def tm(expr):
        expr = takeMoments(expr, useOneNeighborhoodAliasing=True)
        return momentComputation.substitute(expr).subs(solvabilityConditions)

    timeDiffs = OrderedDict()
    nonEqMoms = OrderedDict()
    for epsOrder in range(1, order):
        epsEq = epsilonHierarchy[epsOrder]

        for order in range(order+1):
            eqs = sp.Matrix([fullExpand(tm(epsEq * m)) for m in polyMoments(order, dim)])
            unknownMoments = [m for m in eqs.atoms(CeMoment) if m.superscript == epsOrder and sum(m.momentTuple) == order]
            print(epsOrder, order)
            if len(unknownMoments) == 0:
                for eq in eqs:
                    timeDiffsInExpr = [d for d in eq.atoms(Diff) if
                                       (d.target == 't' or d.target == sp.Symbol("t")) and d.superscript == epsOrder]
                    if len(timeDiffsInExpr) == 0:
                        continue
                    assert len(timeDiffsInExpr) == 1, "Time diffs in expr %d %s" % (len(timeDiffsInExpr), timeDiffsInExpr)
                    td = timeDiffsInExpr[0]
                    timeDiffs[td] = sp.solve(eq, td)[0]
            else:
                solveResult = specialLinSolve(eqs, unknownMoments)
                nonEqMoms.update(solveResult)

    substitutedNonEqMoms = OrderedDict()
    for key, value in nonEqMoms.items():
        print("Substituting", key)
        value = fullExpand(value)
        value = fullExpand(value.subs(timeDiffs)).expand()
        value = fullExpand(value.subs(substitutedNonEqMoms)).expand()
        value = fullExpand(value.subs(timeDiffs)).expand()
        substitutedNonEqMoms[key] = value.expand()

    return timeDiffs, nonEqMoms, substitutedNonEqMoms
