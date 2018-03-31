from functools import partial
import sympy as sp

from lbmpy.innerloopsplit import createLbmSplitGroups
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod
from pystencils.assignment_collection.simplifications import apply_to_all_assignments, \
    subexpression_substitution_in_main_assignments, sympy_cse, add_subexpressions_for_divisions


def createSimplificationStrategy(lbmMethod, doCseInOpposingDirections=False, doOverallCse=False, splitInnerLoop=False):
    from pystencils.assignment_collection import SimplificationStrategy
    from lbmpy.methods import MomentBasedLbMethod
    from lbmpy.methods.momentbasedsimplifications import replaceSecondOrderVelocityProducts, \
        factorDensityAfterFactoringRelaxationTimes, factorRelaxationRates, cseInOpposingDirections, \
        replaceCommonQuadraticAndConstantTerm, replaceDensityAndVelocity

    s = SimplificationStrategy()

    expand = partial(apply_to_all_assignments, operation=sp.expand)
    expand.__name__ = "expand"

    if isinstance(lbmMethod, MomentBasedLbMethod):
        s.add(expand)
        s.add(replaceSecondOrderVelocityProducts)
        s.add(expand)
        s.add(factorRelaxationRates)
        s.add(replaceDensityAndVelocity)
        s.add(replaceCommonQuadraticAndConstantTerm)
        s.add(factorDensityAfterFactoringRelaxationTimes)
        s.add(subexpression_substitution_in_main_assignments)
        if splitInnerLoop:
            s.add(createLbmSplitGroups)
    elif isinstance(lbmMethod, CumulantBasedLbMethod):
        s.add(expand)
        s.add(factorRelaxationRates)

    s.add(add_subexpressions_for_divisions)

    if doCseInOpposingDirections:
        s.add(cseInOpposingDirections)
    if doOverallCse:
        s.add(sympy_cse)

    return s
