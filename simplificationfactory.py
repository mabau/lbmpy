from functools import partial
import sympy as sp

from lbmpy.innerloopsplit import createLbmSplitGroups
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod
from pystencils.assignment_collection.simplifications import applyOnAllEquations, \
    subexpressionSubstitutionInmainAssignments, sympyCSE, addSubexpressionsForDivisions


def createSimplificationStrategy(lbmMethod, doCseInOpposingDirections=False, doOverallCse=False, splitInnerLoop=False):
    from pystencils.assignment_collection import SimplificationStrategy
    from lbmpy.methods import MomentBasedLbMethod
    from lbmpy.methods.momentbasedsimplifications import replaceSecondOrderVelocityProducts, \
        factorDensityAfterFactoringRelaxationTimes, factorRelaxationRates, cseInOpposingDirections, \
        replaceCommonQuadraticAndConstantTerm, replaceDensityAndVelocity

    s = SimplificationStrategy()

    expand = partial(applyOnAllEquations, operation=sp.expand)
    expand.__name__ = "expand"

    if isinstance(lbmMethod, MomentBasedLbMethod):
        s.add(expand)
        s.add(replaceSecondOrderVelocityProducts)
        s.add(expand)
        s.add(factorRelaxationRates)
        s.add(replaceDensityAndVelocity)
        s.add(replaceCommonQuadraticAndConstantTerm)
        s.add(factorDensityAfterFactoringRelaxationTimes)
        s.add(subexpressionSubstitutionInmainAssignments)
        if splitInnerLoop:
            s.add(createLbmSplitGroups)
    elif isinstance(lbmMethod, CumulantBasedLbMethod):
        s.add(expand)
        s.add(factorRelaxationRates)

    s.add(addSubexpressionsForDivisions)

    if doCseInOpposingDirections:
        s.add(cseInOpposingDirections)
    if doOverallCse:
        s.add(sympyCSE)

    return s
