from functools import partial
import sympy as sp
from pystencils.equationcollection.simplifications import applyOnAllEquations, \
    subexpressionSubstitutionInMainEquations, sympyCSE, addSubexpressionsForDivisions


def createSimplificationStrategy(lbmMethod, doCseInOpposingDirections=False, doOverallCse=False):
    from pystencils.equationcollection import SimplificationStrategy
    from lbmpy.methods import MomentBasedLbmMethod
    from lbmpy.methods.momentbasedsimplifications import replaceSecondOrderVelocityProducts, \
        factorDensityAfterFactoringRelaxationTimes, factorRelaxationRates, cseInOpposingDirections, \
        replaceCommonQuadraticAndConstantTerm, replaceDensityAndVelocity

    s = SimplificationStrategy()

    if isinstance(lbmMethod, MomentBasedLbmMethod):
        expand = partial(applyOnAllEquations, operation=sp.expand)
        expand.__name__ = "expand"

        s.add(expand)
        s.add(replaceSecondOrderVelocityProducts)
        s.add(expand)
        s.add(factorRelaxationRates)
        s.add(replaceDensityAndVelocity)
        s.add(replaceCommonQuadraticAndConstantTerm)
        s.add(factorDensityAfterFactoringRelaxationTimes)
        s.add(subexpressionSubstitutionInMainEquations)

    if doCseInOpposingDirections:
        s.add(cseInOpposingDirections)
    if doOverallCse:
        s.add(sympyCSE)

    s.add(addSubexpressionsForDivisions)

    return s


if __name__ == '__main__':
    from lbmpy.stencils import getStencil
    from lbmpy.methods.momentbased import createOrthogonalMRT

    stencil = getStencil("D2Q9")
    m = createOrthogonalMRT(stencil, compressible=True)
    cr = m.getCollisionRule()

    simp = createSimplificationStrategy(m)
    simp(cr)
