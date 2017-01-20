import sympy as sp
from collections import defaultdict


def createLbmSplitGroups(lbmCollisionEqs):
    """
    Creates split groups for LBM collision equations. For details about split groups see
    :func:`pystencils.transformation.splitInnerLoop` .
    The split groups are added as simplification hint 'splitGroups'

    Split groups are created in the following way: Opposing directions are put into a single group.
    The velocity subexpressions are pre-computed as well as all subexpressions which are used in all
    non-center collision equations, and depend on at least one pdf.

    Required simplification hints:
        - velocity: sequence of velocity symbols
    """
    sh = lbmCollisionEqs.simplificationHints
    assert 'velocity' in sh, "Needs simplification hint 'velocity': Sequence of velocity symbols"

    pdfSymbols = lbmCollisionEqs.method.preCollisionPdfSymbols
    stencil = lbmCollisionEqs.method.stencil

    importantSubExpressions = {e.lhs for e in lbmCollisionEqs.subexpressions
                               if pdfSymbols.intersection(lbmCollisionEqs.getDependentSymbols([e.lhs]))}
    for eq in lbmCollisionEqs.mainEquations[1:]:
        importantSubExpressions.intersection_update(eq.rhs.atoms(sp.Symbol))

    subexpressionsToPreCompute = list(sh['velocity']) + list(importantSubExpressions)
    splitGroups = [subexpressionsToPreCompute, ]

    directionGroups = defaultdict(list)
    dim = len(stencil[0])

    for direction, eq in zip(stencil, lbmCollisionEqs.mainEquations):
        if direction == tuple([0]*dim):
            splitGroups[0].append(eq.lhs)
            continue

        inverseDir = tuple([-i for i in direction])

        if inverseDir in directionGroups:
            directionGroups[inverseDir].append(eq.lhs)
        else:
            directionGroups[direction].append(eq.lhs)
    splitGroups += directionGroups.values()

    return splitGroups
