"""
This module holds special transformations for simplifying the collision equations of moment-based methods.
All of these transformations operate on :class:`pystencils.equationcollection.EquationCollection` and need special
simplification hints, which are set by the MomentBasedLbmMethod.
"""
import sympy as sp
from pystencils.sympyextensions import replaceAdditive, replaceSecondOrderProducts, extractMostCommonFactor


def replaceSecondOrderVelocityProducts(lbmCollisionEqs):
    """
    Replaces mixed quadratic velocity terms like :math:`u_0 * u_1` by :math:`(u_0+u+1)^2 - u_0^2 - u_1^2`
    Required simplification hints:
        - velocity: sequence of velocity symbols
    """
    sh = lbmCollisionEqs.simplificationHints
    assert 'velocity' in sh, "Needs simplification hint 'velocity': Sequence of velocity symbols"

    result = []
    substitutions = []
    u = sh['velocity']
    for i, s in enumerate(lbmCollisionEqs.mainEquations):
        newRhs = replaceSecondOrderProducts(s.rhs, u, positive=None, replaceMixed=substitutions)
        result.append(sp.Eq(s.lhs, newRhs))
    res = lbmCollisionEqs.copy(result)
    res.subexpressions += substitutions
    return res


def factorRelaxationRates(lbmCollisionEqs):
    """
    Factors collision equations by relaxation rates.
    Required simplification hints:
        - relaxationRates: Sequence of relaxation rate symbols
    """
    sh = lbmCollisionEqs.simplificationHints
    assert 'relaxationRates' in sh, "Needs simplification hint 'relaxationRates': Sequence of relaxation rate symbols"

    result = []
    for s in lbmCollisionEqs.mainEquations:
        newRhs = s.rhs
        for rp in sh['relaxationRates']:
            newRhs = newRhs.collect(rp)
        result.append(sp.Eq(s.lhs, newRhs))
    return lbmCollisionEqs.copy(result)


def factorDensityAfterFactoringRelaxationTimes(lbmCollisionEqs):
    """
    Tries to factor out the density. This only works if previously
    :func:`lbmpy.methods.momentbasedsimplifications.factorRelaxationTimes` was run.

    This transformations makes only sense for compressible models - for incompressible models this does nothing

    Required simplification hints:
        - density: density symbol which is factored out
        - relaxationRates: set of symbolic relaxation rates in which the terms are assumed to be already factored
    """
    sh = lbmCollisionEqs.simplificationHints
    assert 'density' in sh, "Needs simplification hint 'density': Symbol for density"
    assert 'relaxationRates' in sh, "Needs simplification hint 'relaxationRates': Set of symbolic relaxation rates"
    result = []
    rho = sh['density']
    for s in lbmCollisionEqs.mainEquations:
        newRhs = s.rhs
        for rp in sh['relaxationRates']:
            coeff = newRhs.coeff(rp)
            newRhs = newRhs.subs(coeff, coeff.collect(rho))
        result.append(sp.Eq(s.lhs, newRhs))
    return lbmCollisionEqs.copy(result)


def replaceDensityAndVelocity(lbmCollisionEqs):
    """
    Looks for terms that can be replaced by the density or by one of the velocities
        Required simplification hints:
        - density: density symbol
        - velocity: sequence of velocity symbols
    """
    sh = lbmCollisionEqs.simplificationHints
    assert 'density' in sh, "Needs simplification hint 'density': Symbol for density"
    assert 'velocity' in sh, "Needs simplification hint 'velocity': Sequence of velocity symbols"
    rho = sh['density']
    u = sh['velocity']

    substitutions = lbmCollisionEqs.extract([rho] + list(u)).insertSubexpressions().mainEquations
    result = []
    for s in lbmCollisionEqs.mainEquations:
        newRhs = s.rhs
        for replacement in substitutions:
            newRhs = replaceAdditive(newRhs, replacement.lhs, replacement.rhs, requiredMatchReplacement=0.5)
        result.append(sp.Eq(s.lhs, newRhs))
    return lbmCollisionEqs.copy(result)


def replaceCommonQuadraticAndConstantTerm(lbmCollisionEqs):
    """
    A common quadratic term (f_eq_common) is extracted from the collision equation for center
    and substituted in all equations

    Required simplification hints:
        - density: density symbol
        - velocity: sequence of velocity symbols
        - relaxationRates: set of symbolic relaxation rates
        - stencil:
    """
    sh = lbmCollisionEqs.simplificationHints
    assert 'density' in sh, "Needs simplification hint 'density': Symbol for density"
    assert 'velocity' in sh, "Needs simplification hint 'velocity': Sequence of velocity symbols"
    assert 'relaxationRates' in sh, "Needs simplification hint 'relaxationRates': Set of symbolic relaxation rates"

    stencil = lbmCollisionEqs.method.stencil
    assert sum([abs(e) for e in stencil[0]]) == 0, "Works only if first stencil entry is the center direction"
    f_eq_common = __getCommonQuadraticAndConstantTerms(lbmCollisionEqs)

    if len(f_eq_common.args) > 1:
        f_eq_common = sp.Eq(sp.Symbol('f_eq_common'), f_eq_common)
        result = []
        for s in lbmCollisionEqs.mainEquations:
            newRhs = replaceAdditive(s.rhs, f_eq_common.lhs, f_eq_common.rhs, requiredMatchReplacement=0.5)
            result.append(sp.Eq(s.lhs, newRhs))
        res = lbmCollisionEqs.copy(result)
        res.subexpressions.append(f_eq_common)
        return res
    else:
        return lbmCollisionEqs


def cseInOpposingDirections(lbmCollisionEqs):
    """
    Looks for common subexpressions in terms for opposing directions (e.g. north & south, top & bottom )

    Required simplification hints:
        - relaxationRates: set of symbolic relaxation rates
        - postCollisionPdfSymbols: sequence of symbols
    """
    sh = lbmCollisionEqs.simplificationHints
    assert 'relaxationRates' in sh, "Needs simplification hint 'relaxationRates': Set of symbolic relaxation rates"

    updateRules = lbmCollisionEqs.mainEquations
    stencil = lbmCollisionEqs.method.stencil
    relaxationRates = sh['relaxationRates']

    replacementSymbolGenerator = lbmCollisionEqs.subexpressionSymbolNameGenerator

    directionToUpdateRule = {direction: updateRule for updateRule, direction in zip(updateRules, stencil)}
    result = []
    substitutions = []
    newCoefficientSubstitutions = dict()
    for updateRule, direction in zip(updateRules, stencil):
        if direction not in directionToUpdateRule:
            continue  # already handled the inverse direction
        inverseDir = tuple([-i for i in direction])
        inverseRule = directionToUpdateRule[inverseDir]
        if inverseDir == direction:
            result.append(updateRule)  # center is not modified
            continue
        del directionToUpdateRule[inverseDir]
        del directionToUpdateRule[direction]

        updateRules = [updateRule, inverseRule]

        if len(relaxationRates) == 0:
            foundSubexpressions, newTerms = sp.cse(updateRules, symbols=replacementSymbolGenerator,
                                                   order='None', optimizations=[])
            substitutions += [sp.Eq(f[0], f[1]) for f in foundSubexpressions]

            updateRules = newTerms
        else:
            for relaxationRate in relaxationRates:
                terms = [updateRule.rhs.coeff(relaxationRate) for updateRule in updateRules]
                resultOfCommonFactor = [extractMostCommonFactor(t) for t in terms]
                commonFactors = [r[0] for r in resultOfCommonFactor]
                termsWithoutFactor = [r[1] for r in resultOfCommonFactor]

                if commonFactors[0] == commonFactors[1] and commonFactors[0] != 1:
                    newCoefficient = commonFactors[0] * relaxationRate
                    if newCoefficient not in newCoefficientSubstitutions:
                        newCoefficientSubstitutions[newCoefficient] = next(replacementSymbolGenerator)
                    newCoefficient = newCoefficientSubstitutions[newCoefficient]
                    handledTerms = termsWithoutFactor
                else:
                    newCoefficient = relaxationRate
                    handledTerms = terms

                foundSubexpressions, newTerms = sp.cse(handledTerms, symbols=replacementSymbolGenerator,
                                                       order='None', optimizations=[])
                substitutions += [sp.Eq(f[0], f[1]) for f in foundSubexpressions]

                updateRules = [sp.Eq(ur.lhs, ur.rhs.subs(relaxationRate*oldTerm, newCoefficient*newTerm))
                               for ur, newTerm, oldTerm in zip(updateRules, newTerms, terms)]

        result += updateRules

    for term, substitutedVar in newCoefficientSubstitutions.items():
        substitutions.append(sp.Eq(substitutedVar, term))

    result.sort(key=lambda e: lbmCollisionEqs.method.postCollisionPdfSymbols.index(e.lhs))
    res = lbmCollisionEqs.copy(result)
    res.subexpressions += substitutions
    return res


# -------------------------------------- Helper Functions --------------------------------------------------------------

def __getCommonQuadraticAndConstantTerms(lbmCollisionEqs):
    """Determines a common subexpression useful for most LBM model often called f_eq_common.
    It contains the quadratic and constant terms of the center update rule."""
    sh = lbmCollisionEqs.simplificationHints
    stencil = lbmCollisionEqs.method.stencil
    relaxationRates = sh['relaxationRates']

    dim = len(stencil[0])

    pdfSymbols = lbmCollisionEqs.freeSymbols - relaxationRates

    center = tuple([0]*dim)
    t = lbmCollisionEqs.mainEquations[stencil.index(center)].rhs
    for rp in relaxationRates:
        t = t.subs(rp, 1)

    for fa in pdfSymbols:
        t = t.subs(fa, 0)

    if 'forceTerms' in sh:
        t = t.subs({ft: 0 for ft in sh['forceTerms']})

    weight = t

    for u in sh['velocity']:
        weight = weight.subs(u, 0)
    weight = weight / sh['density']
    return t / weight

