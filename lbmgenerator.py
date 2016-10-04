import lbmpy.collisionoperator as coll
import lbmpy.generator as generator

import lbmpy.util as util
import sympy as sp
import lbmpy.transformations as trafos
from lbmpy.stencils import getStencil
from lbmpy.densityVelocityExpressions import getDensityVelocityExpressions


def processCollideTerms(lm, pdfSymbols):
    replacements = []
    result = []

    terms = lm.getCollideTerms(pdfSymbols)

    rho = util.getSymbolicDensity()
    u = util.getSymbolicVelocityVector(lm.dim, "u")

    rhoDefinition = [sp.Eq(rho, sum(pdfSymbols))]
    uDefinition = [sp.Eq(u_i, u_term_i) for u_i, u_term_i in zip(u, lm.getVelocityTerms(pdfSymbols))]

    sum_u_sq = sum([u_i ** 2 for u_i in u])
    f_eq_common = [sp.Eq(sp.Symbol('f_eq_common'), rho - rho * sp.Rational(3, 2) * sum_u_sq).expand()]
    replacements.append(f_eq_common[0].factor())
    for s in terms:
        s = s.expand()
        s = sp.simplify(trafos.replaceSecondOrderProducts(s, u, positive=None, replaceMixed=replacements).expand())

        for rp in lm.collisionDOFs:
            s = s.collect(rp)
        for replacement in rhoDefinition + uDefinition + f_eq_common:
            s = trafos.replaceAdditive(s, replacement.lhs, replacement.rhs, len(replacement.rhs.args) // 2)
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


