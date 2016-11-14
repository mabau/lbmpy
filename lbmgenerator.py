import operator
from collections import Counter, defaultdict

import sympy as sp
import lbmpy.transformations as trafos
import lbmpy.util as util
from lbmpy.densityVelocityExpressions import getDensityVelocityExpressions
from pystencils.field import Field, getLayoutFromNumpyArray


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


def createCollisionEquations(lm, pdfSymbols, dstField, densityOutputField=None, velocityOutputField=None, doCSE=False):
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
        forceTerm = 0 if lm.forceModel is None else lm.forceModel(latticeModel=lm)[i]
        updateEquations.append(sp.Eq(dstField[noOffset](i), pdfSymbols[i] + s + forceTerm))

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

    if densityOutputField is not None:
        result.append(sp.Eq(densityOutputField(0), rho))
    if velocityOutputField is not None:
        if hasattr(lm.forceModel, "macroscopicVelocity"):
            macroscopicVelocity = lm.forceModel.macroscopicVelocity(lm, u, rho)
        else:
            macroscopicVelocity = u
        result += [sp.Eq(velocityOutputField(i), u_i) for i, u_i in enumerate(macroscopicVelocity)]

    return result, replacements


def createLbmEquations(lm, numpyField=None, srcFieldName="src", dstFieldName="dst",
                       velocityOutputField=None, densityOutputField=None,
                       doCSE=False):
    """
    Creates a list of LBM update equations
    :param lm: instance of lattice model
    :param numpyField: optional numpy field for PDFs. Used to create a kernel of fixed loop bounds and strides
                       if None, a generic kernel is created
    :param srcFieldName: name of the pdf source field
    :param dstFieldName: name of the pdf destination field
    :param velocityOutputField: can be either None in which case the velocity is not written to field
                                if it is a string, velocity is written to a generic velocity field with that name
                                if it is a tuple (name, numpyArray), the numpyArray is used to determine size and stride
                                of the output field
    :param densityOutputField: similar to velocityOutputField
    :param doCSE: if True, common subexpression elimination is done for pdfs in opposing directions
    :return: list of sympy equations
    """
    if numpyField is not None:
        assert len(numpyField.shape) == lm.dim + 1

    velOutField = None
    densityOutField = None

    layout = tuple(getLayoutFromNumpyArray(numpyField)[:lm.dim]) if numpyField is not None else None
    if velocityOutputField:
        if isinstance(velocityOutputField, tuple):
            velOutField = Field.createFromNumpyArray(velocityOutputField[0], velocityOutputField[1], indexDimensions=1)
        else:
            velOutField = Field.createGeneric(velocityOutputField, lm.dim, indexDimensions=1, layout=layout)
    if densityOutputField:
        if isinstance(densityOutputField, tuple):
            densityOutField = Field.createFromNumpyArray(densityOutputField[0], densityOutputField[1])
        else:
            densityOutField = Field.createGeneric(densityOutputField, lm.dim, indexDimensions=0, layout=layout)

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

    rhoSubexprs, rhoEq, uSubexrp, uEqs = getDensityVelocityExpressions(lm.stencil, streamedPdfs, lm.compressible)

    if hasattr(lm.forceModel, "equilibriumVelocity"):
        uSymbols = [e.lhs for e in uEqs]
        uRhs = [e.rhs for e in uEqs]
        correctedVel = lm.forceModel.macroscopicVelocity(lm, uRhs, rhoEq.lhs)
        uEqs = [sp.Eq(u_i, correctedVel_i) for u_i, correctedVel_i in zip(uSymbols, correctedVel)]

    densityVelocityDefinition = rhoSubexprs + [rhoEq] + uSubexrp + uEqs

    collideEqs, subExpressions = createCollisionEquations(lm, streamedPdfs, dst, velocityOutputField=velOutField,
                                                          densityOutputField=densityOutField, doCSE=doCSE)

    return densityVelocityDefinition + subExpressions + collideEqs


def createLbmSplitGroups(lm, equations):
    f_eq_common = sp.Symbol('f_eq_common')
    rho = sp.Symbol('rho')

    result = [
        list(util.getSymbolicVelocityVector(lm.dim)),
    ]

    if lm.compressible:
        result[0].append(rho)

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

        direction = lm.stencil[idx]
        inverseDir = tuple([-i for i in direction])

        if inverseDir in directionGroups:
            directionGroups[inverseDir].append(eq.lhs)
        else:
            directionGroups[direction].append(eq.lhs)
    result += directionGroups.values()

    if f_eq_common not in result[0]:
        result[0].append(rho)

    return result
