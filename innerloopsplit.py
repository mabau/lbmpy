import sympy as sp
from collections import defaultdict
from pystencils import Field


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

    preCollisionSymbols = set(lbmCollisionEqs.method.preCollisionPdfSymbols)
    nonCenterPostCollisionSymbols = set(lbmCollisionEqs.method.postCollisionPdfSymbols[1:])
    postCollisionSymbols = set(lbmCollisionEqs.method.postCollisionPdfSymbols)

    stencil = lbmCollisionEqs.method.stencil

    importantSubExpressions = {e.lhs for e in lbmCollisionEqs.subexpressions
                               if preCollisionSymbols.intersection(lbmCollisionEqs.getDependentSymbols([e.lhs]))}

    otherWrittenFields = []
    for eq in lbmCollisionEqs.mainAssignments:
        if eq.lhs not in postCollisionSymbols and isinstance(eq.lhs, Field.Access):
            otherWrittenFields.append(eq.lhs)
        if eq.lhs not in nonCenterPostCollisionSymbols:
            continue
        importantSubExpressions.intersection_update(eq.rhs.atoms(sp.Symbol))

    importantSubExpressions.update(sh['velocity'])

    subexpressionsToPreCompute = list(importantSubExpressions)
    splitGroups = [subexpressionsToPreCompute + otherWrittenFields, ]

    directionGroups = defaultdict(list)
    dim = len(stencil[0])

    for direction, eq in zip(stencil, lbmCollisionEqs.mainAssignments):
        if direction == tuple([0]*dim):
            splitGroups[0].append(eq.lhs)
            continue

        inverseDir = tuple([-i for i in direction])

        if inverseDir in directionGroups:
            directionGroups[inverseDir].append(eq.lhs)
        else:
            directionGroups[direction].append(eq.lhs)
    splitGroups += directionGroups.values()

    lbmCollisionEqs.simplificationHints['splitGroups'] = splitGroups
    return lbmCollisionEqs
