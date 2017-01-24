from functools import partial
import sympy as sp

from lbmpy.innerloopsplit import createLbmSplitGroups
from pystencils.equationcollection.simplifications import applyOnAllEquations, \
    subexpressionSubstitutionInMainEquations, sympyCSE, addSubexpressionsForDivisions


def createSimplificationStrategy(lbmMethod, doCseInOpposingDirections=False, doOverallCse=False, splitInnerLoop=False):
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
        if splitInnerLoop:
            s.add(createLbmSplitGroups)

    if doCseInOpposingDirections:
        s.add(cseInOpposingDirections)
    if doOverallCse:
        s.add(sympyCSE)

    s.add(addSubexpressionsForDivisions)

    return s
