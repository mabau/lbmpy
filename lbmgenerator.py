import lbmpy.generator as generator
import lbmpy.util as util
import sympy as sp
import lbmpy.transformations as trafos
from lbmpy.densityVelocityExpressions import getDensityVelocityExpressions
from lbmpy.generator import Field


def getCommonQuadraticAndConstantTerms(simplifiedUpdateRuleForCenter, latticeModel):
    """Determines a common subexpression useful for most LBM model often called f_eq_common.
    It contains the quadratic and constant terms of the center update rule."""
    t = simplifiedUpdateRuleForCenter
    for rp in latticeModel.collisionDOFs:
        t = t.subs(rp, 1)

    for fa in t.atoms(Field.Access):
        t = t.subs(fa, 0)

    weight = t
    for u in util.getSymbolicVelocityVector(latticeModel.dim):
        weight = weight.subs(u, 0)
    weight = weight / util.getSymbolicDensity()
    return t / weight


def processCollideTerms(lm, pdfSymbols):
    replacements = []
    result = []

    terms = lm.getCollideTerms(pdfSymbols)

    rho = util.getSymbolicDensity()
    u = util.getSymbolicVelocityVector(lm.dim, "u")

    rhoDefinition = [sp.Eq(rho, sum(pdfSymbols))]
    uDefinition = [sp.Eq(u_i, u_term_i) for u_i, u_term_i in zip(u, lm.getVelocityTerms(pdfSymbols))]

    simplifiedUpdateRules = []
    for s in terms:
        s = s.expand()
        s = sp.simplify(trafos.replaceSecondOrderProducts(s, u, positive=None, replaceMixed=replacements).expand())

        for rp in lm.collisionDOFs:
            s = s.collect(rp)
        for replacement in rhoDefinition + uDefinition:
            s = trafos.replaceAdditive(s, replacement.lhs, replacement.rhs, len(replacement.rhs.args) // 2)
        simplifiedUpdateRules.append(s)

    assert sum([abs(e) for e in lm.stencil[0]]) == 0, "Works only if first stencil entry is the center direction"
    f_eq_common = getCommonQuadraticAndConstantTerms(simplifiedUpdateRules[0], lm)
    f_eq_common = sp.Eq(sp.Symbol('f_eq_common'), f_eq_common)
    replacements.append(f_eq_common)
    for s in simplifiedUpdateRules:
        s = trafos.replaceAdditive(s, f_eq_common.lhs, f_eq_common.rhs, len(f_eq_common.rhs.args) // 2)
        result.append(s)

    return result, replacements


def createLbmEquations(lm, numpyField=None, srcFieldName="src", dstFieldName="dst"):
    if numpyField is None:
        src = generator.Field.createGeneric(srcFieldName, lm.dim, indexDimensions=1)
        dst = generator.Field.createGeneric(dstFieldName, lm.dim, indexDimensions=1)
    else:
        src = generator.Field.createFromNumpyArray(srcFieldName, numpyField, indexDimensions=1)
        dst = generator.Field.createFromNumpyArray(dstFieldName, numpyField, indexDimensions=1)

    streamedPdfs = []
    for ownIdx, offset in enumerate(lm.stencil):
        inverseIdx = tuple([-d for d in offset])
        streamedPdfs.append(src[inverseIdx](ownIdx))

    densityVelocityDefinition = getDensityVelocityExpressions(lm.stencil, streamedPdfs)

    collideTerms, subExpressions = processCollideTerms(lm, streamedPdfs)

    noOffset = tuple([0]*lm.dim)
    collideEqs = [sp.Eq(dst[noOffset](i), streamedPdfs[i] + term) for i, term in enumerate(collideTerms)]

    return densityVelocityDefinition + subExpressions + collideEqs


