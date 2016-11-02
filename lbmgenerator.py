import operator
from collections import Counter, defaultdict

import sympy as sp
from joblib import Memory

import lbmpy.transformations as trafos
import lbmpy.util as util
from lbmpy.densityVelocityExpressions import getDensityVelocityExpressions
from pystencils.field import Field

memory = Memory(cachedir="/tmp/lbmpy", verbose=False)


def getCommonQuadraticAndConstantTerms(simplifiedUpdateRuleForCenter, latticeModel):
    """Determines a common subexpression useful for most LBM model often called f_eq_common.
    It contains the quadratic and constant terms of the center update rule."""
    t = simplifiedUpdateRuleForCenter
    for rp in latticeModel.relaxationRates:
        t = t.subs(rp, 1)

    for fa in t.atoms(Field.Access):
        t = t.subs(fa, 0)

    weight = t
    for u in util.getSymbolicVelocityVector(latticeModel.dim):
        weight = weight.subs(u, 0)
    weight = weight / util.getSymbolicDensity()
    return t / weight


def pullCommonFactorOut(term):
    coeffDict = term.as_coefficients_dict()
    counter = Counter(coeffDict.values())
    commonFactor, occurances = max(counter.items(), key=operator.itemgetter(1))
    if occurances == 1 and (1 in counter):
        commonFactor = 1
    return commonFactor, term / commonFactor


def cseInOpposingDirections(updateRules, stencil, relaxationParameters):
    def ReplacementSymbolGenerator(name="xi"):
        counter = 0
        while True:
            yield sp.Symbol("%s_%d" % (name, counter))
            counter += 1

    replacementSymbolGenerator = ReplacementSymbolGenerator()

    directionToUpdateRule = {direction: updateRule for updateRule, direction in zip(updateRules, stencil)}
    result = []
    substitutions = []
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
        for rp in relaxationParameters:
            t1 = inverseRule.rhs.coeff(rp)
            t2 = updateRule.rhs.coeff(rp)
            # f1, t1 = pullCommonFactorOut(t1)
            # f2, t2 = pullCommonFactorOut(t2)

            foundSubexpressions, (newT1, newT2) = sp.cse([t1, t2], symbols=replacementSymbolGenerator,
                                                         order='None', optimizations=[])

            substitutions += [sp.Eq(f[0], f[1]) for f in foundSubexpressions]

            inverseRule = sp.Eq(inverseRule.lhs, inverseRule.rhs.subs(t1, newT1))
            updateRule = sp.Eq(updateRule.lhs, updateRule.rhs.subs(t2, newT2))
        result.append(updateRule)
        result.append(inverseRule)
    return substitutions, result


def createCollisionEquations(lm, pdfSymbols, dstField, doCSE):
    replacements = []
    result = []

    terms = lm.getCollideTerms(pdfSymbols)

    rho = util.getSymbolicDensity()
    u = util.getSymbolicVelocityVector(lm.dim, "u")

    rhoDefinition = [sp.Eq(rho, sum(pdfSymbols))]
    uDefinition = [sp.Eq(u_i, u_term_i) for u_i, u_term_i in zip(u, lm.getVelocityTerms(pdfSymbols))]

    updateEquations = []
    velocitySumReplacements = []
    for i, s in enumerate(terms):
        s = s.expand()
        s = trafos.replaceSecondOrderProducts(s, u, positive=None, replaceMixed=velocitySumReplacements)
        s = s.expand(s)

        for rp in lm.relaxationRates:
            s = s.collect(rp)
        for replacement in rhoDefinition + uDefinition:
            s = trafos.replaceAdditive(s, replacement.lhs, replacement.rhs, len(replacement.rhs.args) // 2)

        noOffset = tuple([0] * lm.dim)
        updateEquations.append(sp.Eq(dstField[noOffset](i), pdfSymbols[i] + s))

    replacements += velocitySumReplacements

    assert sum([abs(e) for e in lm.stencil[0]]) == 0, "Works only if first stencil entry is the center direction"
    f_eq_common = getCommonQuadraticAndConstantTerms(updateEquations[0].rhs, lm)
    if f_eq_common != rho:
        f_eq_common = sp.Eq(sp.Symbol('f_eq_common'), f_eq_common)
        replacements.append(f_eq_common)
    else:
        f_eq_common = None
    updateRulesTransformed = []
    for s in updateEquations:
        sRhs = s.rhs
        if f_eq_common is not None:
            sRhs = trafos.replaceAdditive(sRhs, f_eq_common.lhs, f_eq_common.rhs, len(f_eq_common.rhs.args) // 2)
        for velSumEq in velocitySumReplacements:
            sRhs = trafos.replaceAdditive(sRhs, velSumEq.lhs, velSumEq.rhs, len(velSumEq.rhs.args))
        updateRulesTransformed.append(sp.Eq(s.lhs, sRhs))

    if doCSE:
        repl, updateRulesTransformed = cseInOpposingDirections(updateRulesTransformed, lm.stencil, lm.relaxationRates)
        replacements += repl
    result += updateRulesTransformed

    return result, replacements


def createLbmEquations(lm, numpyField=None, srcFieldName="src", dstFieldName="dst", doCSE=False):
    if numpyField is None:
        src = Field.createGeneric(srcFieldName, lm.dim, indexDimensions=1)
        dst = Field.createGeneric(dstFieldName, lm.dim, indexDimensions=1)
    else:
        src = Field.createFromNumpyArray(srcFieldName, numpyField, indexDimensions=1)
        dst = Field.createFromNumpyArray(dstFieldName, numpyField, indexDimensions=1)

    streamedPdfs = []
    for ownIdx, offset in enumerate(lm.stencil):
        inverseIdx = tuple([-d for d in offset])
        streamedPdfs.append(src[inverseIdx](ownIdx))

    densityVelocityDefinition = getDensityVelocityExpressions(lm.stencil, streamedPdfs, lm.compressible)

    collideEqs, subExpressions = createCollisionEquations(lm, streamedPdfs, dst, doCSE)

    return densityVelocityDefinition + subExpressions + collideEqs


def createLbmSplitGroups(lm, equations):
    f_eq_common = sp.Symbol('f_eq_common')
    rho = sp.Symbol('rho')

    result = [
        list(util.getSymbolicVelocityVector(lm.dim)),
    ]
    directionGroups = defaultdict(list)

    for eq in equations:
        if f_eq_common in eq.rhs.atoms(sp.Symbol) and f_eq_common not in result[0]:
            result[0].append(f_eq_common)

        if not type(eq.lhs) is Field.Access:
            continue

        idx = eq.lhs.index[0]
        if idx == 0:
            result[0].append(eq.lhs)
            continue

        dir = lm.stencil[idx]
        inverseDir = tuple([-i for i in dir])

        if inverseDir in directionGroups:
            directionGroups[inverseDir].append(eq.lhs)
        else:
            directionGroups[dir].append(eq.lhs)
    result += directionGroups.values()

    if f_eq_common not in result[0]:
        result[0].append(rho)

    return result


if __name__ == "__main__":
    from lbmpy.collisionoperator import makeSRT
    from lbmpy.stencils import getStencil

    latticeModel = makeSRT(getStencil("D3Q19"), order=2, compressible=False)
    r = createLbmEquations(latticeModel)
