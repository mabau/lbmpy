import sympy as sp
import lbmpy.transformations as trafos
from lbmpy.latticemodel import LbmCollisionRule


class Strategy:
    def __init__(self):
        self._transformations = []

    def addSimplificationRule(self, transformation):
        self._transformations.append(transformation)

    def __call__(self,*args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, updateRule):
        for t in self._transformations:
            updateRule = t(updateRule)
        return updateRule

    def applyWithReport(self, updateRule):
        import time
        report = []
        op = updateRule.countNumberOfOperations()
        report.append(["OriginalTerm", op['adds'], op['muls'], op['divs'], '-'])
        for t in self._transformations:
            startTime = time.perf_counter()
            updateRule = t(updateRule)
            endTime = time.perf_counter()
            op = updateRule.countNumberOfOperations()
            report.append([t.__name__, op['adds'], op['muls'], op['divs'], "%.2f ms" % ((endTime-startTime)*1000,)])
        return updateRule, report

    def applyAndPrintIntermediateResults(self, updateRule, printFunction=print, directions=None):
        printFunction("Initial version")
        updateRule.displayRepresentative(printFunction, directions)

        for t in self._transformations:
            updateRule = t(updateRule)
            printFunction(t.__name__)
            updateRule.displayRepresentative(printFunction, directions)

        return updateRule

    def __repr__(self):
        result = "Simplification Strategy:\n"
        for t in self._transformations:
            result += " - %s\n" % (t.__name__,)
        return result


def createDefaultMomentSpaceSimplificationStrategy():
    s = Strategy()
    s.addSimplificationRule(expand)
    s.addSimplificationRule(replaceSecondOrderProducts)
    s.addSimplificationRule(expand)
    s.addSimplificationRule(factorRelaxationTimes)
    s.addSimplificationRule(replaceDensityAndVelocity)
    s.addSimplificationRule(replaceCommonQuadraticAndConstantTerm)
    s.addSimplificationRule(factorRhoAfterFactoringRelaxationTimes)
    s.addSimplificationRule(subexpressionSubstitutionInUpdateEquations)
    return s


# ------------------------------------  Simplification Rules    --------------------------------------------------------

def sympyCSE(lbmUpdateRule):
    replacements, newEq = sp.cse(lbmUpdateRule.subexpressions + lbmUpdateRule.updateEquations)
    replacementEqs = [sp.Eq(*r) for r in replacements]

    modifiedSubexpressions = newEq[:len(lbmUpdateRule.subexpressions)]
    modifiedUpdateEquations = newEq[len(lbmUpdateRule.subexpressions):]

    newSubexpressions = replacementEqs + modifiedSubexpressions
    topologicallySortedPairs = sp.cse_main.reps_toposort([[e.lhs, e.rhs] for e in newSubexpressions])
    newSubexpressions = [sp.Eq(a[0], a[1]) for a in topologicallySortedPairs]

    return LbmCollisionRule(modifiedUpdateEquations, newSubexpressions,
                            lbmUpdateRule.latticeModel, lbmUpdateRule.updateEquationDirections)


def factorRhoAfterFactoringRelaxationTimes(lbmUpdateRule):
    """Important for compressible models only"""
    result = []
    rho = lbmUpdateRule.latticeModel.symbolicDensity
    for s in lbmUpdateRule.updateEquations:
        newRhs = s.rhs
        for rp in lbmUpdateRule.latticeModel.relaxationRates:
            coeff = newRhs.coeff(rp)
            newRhs = newRhs.subs(coeff, coeff.collect(rho))
        result.append(sp.Eq(s.lhs, newRhs))
    return lbmUpdateRule.newWithSubexpressions(result, [])


def subexpressionSubstitutionInExistingSubexpressions(lbmUpdateRule):
    result = []
    for outerCtr, s in enumerate(lbmUpdateRule.subexpressions):
        newRhs = s.rhs
        for innerCtr in range(outerCtr):
            subExpr = lbmUpdateRule.subexpressions[innerCtr]
            newRhs = trafos.replaceAdditive(newRhs, subExpr.lhs, subExpr.rhs, requiredMatchReplacement=1.0)
            newRhs = newRhs.subs(subExpr.rhs, subExpr.lhs)
        result.append(sp.Eq(s.lhs, newRhs))

    return LbmCollisionRule(lbmUpdateRule.updateEquations, result,
                            lbmUpdateRule.latticeModel, lbmUpdateRule.updateEquationDirections)


def subexpressionSubstitutionInUpdateEquations(lbmUpdateRule):
    result = []
    for s in lbmUpdateRule.updateEquations:
        newRhs = s.rhs
        for subExpr in lbmUpdateRule.subexpressions:
            newRhs = trafos.replaceAdditive(newRhs, subExpr.lhs, subExpr.rhs, requiredMatchReplacement=1.0)
        result.append(sp.Eq(s.lhs, newRhs))
    return lbmUpdateRule.newWithSubexpressions(result, [])


def replaceSecondOrderProducts(lbmUpdateRule):
    result = []
    substitutions = []
    u = lbmUpdateRule.latticeModel.symbolicVelocity
    for i, s in enumerate(lbmUpdateRule.updateEquations):
        newRhs = trafos.replaceSecondOrderProducts(s.rhs, u, positive=None, replaceMixed=substitutions)
        result.append(sp.Eq(s.lhs, newRhs))
    return lbmUpdateRule.newWithSubexpressions(result, substitutions)


def expand(lbmUpdateRule):
    result = [s.expand() for s in lbmUpdateRule.updateEquations]
    return lbmUpdateRule.newWithSubexpressions(result, [])


def factorRelaxationTimes(lbmUpdateRule):
    lm = lbmUpdateRule.latticeModel
    result = []
    for s in lbmUpdateRule.updateEquations:
        newRhs = s.rhs
        for rp in lm.relaxationRates:
            newRhs = newRhs.collect(rp)
        result.append(sp.Eq(s.lhs, newRhs))
    return lbmUpdateRule.newWithSubexpressions(result, [])


def replaceDensityAndVelocity(lbmUpdateRule):
    lm = lbmUpdateRule.latticeModel
    rho = lm.symbolicDensity
    u = lm.symbolicVelocity

    velocity = []
    for i in range(lm.dim):
        velocity.append(sum([st[i] * f for st, f in zip(lm.stencil, lm.pdfSymbols)]))

    rhoDefinition = [sp.Eq(rho, sum(lm.pdfSymbols))]
    uDefinition = [sp.Eq(u_i, u_term_i) for u_i, u_term_i in zip(u, velocity)]

    substitutions = rhoDefinition + uDefinition
    result = []
    for s in lbmUpdateRule.updateEquations:
        newRhs = s.rhs
        for replacement in substitutions:
            newRhs = trafos.replaceAdditive(newRhs, replacement.lhs, replacement.rhs, requiredMatchReplacement=0.5)
        result.append(sp.Eq(s.lhs, newRhs))
    return lbmUpdateRule.newWithSubexpressions(result, [])


def replaceCommonQuadraticAndConstantTerm(lbmUpdateRule):
    lm = lbmUpdateRule.latticeModel

    assert sum([abs(e) for e in lm.stencil[0]]) == 0, "Works only if first stencil entry is the center direction"
    f_eq_common = __getCommonQuadraticAndConstantTerms(lbmUpdateRule)

    if len(f_eq_common.args) > 1:
        f_eq_common = sp.Eq(sp.Symbol('f_eq_common'), f_eq_common)
        result = []
        for s in lbmUpdateRule.updateEquations:
            newRhs = trafos.replaceAdditive(s.rhs, f_eq_common.lhs, f_eq_common.rhs, requiredMatchReplacement=0.5)
            result.append(sp.Eq(s.lhs, newRhs))
        return lbmUpdateRule.newWithSubexpressions(result, [f_eq_common])
    else:
        return lbmUpdateRule


def cseInOpposingDirections(lbmUpdateRule):
    """
    Looks for common subexpressions in terms for opposing directions (e.g. north & south, top & bottom )
    """
    latticeModel = lbmUpdateRule.latticeModel
    updateRules = lbmUpdateRule.updateEquations
    stencil = latticeModel.stencil
    relaxationRates = latticeModel.relaxationRates

    def ReplacementSymbolGenerator(name="xi"):
        counter = 0
        while True:
            yield sp.Symbol("%s_%d" % (name, counter))
            counter += 1

    replacementSymbolGenerator = ReplacementSymbolGenerator()

    directionToUpdateRule = {direction: updateRule for updateRule, direction in zip(updateRules, stencil)}
    result = []
    substitutions = []
    newCoefficientSubstitutions = dict()
    handledDirections = []
    for updateRule, direction in zip(updateRules, stencil):
        if direction not in directionToUpdateRule:
            continue  # already handled the inverse direction
        inverseDir = tuple([-i for i in direction])
        inverseRule = directionToUpdateRule[inverseDir]
        if inverseDir == direction:
            result.append(updateRule)  # center is not modified
            handledDirections.append(direction)
            continue
        del directionToUpdateRule[inverseDir]
        del directionToUpdateRule[direction]

        updateRules = [updateRule, inverseRule]

        if latticeModel.allRelaxationRatesFixed:
            foundSubexpressions, newTerms = sp.cse(updateRules, symbols=replacementSymbolGenerator,
                                                   order='None', optimizations=[])
            substitutions += [sp.Eq(f[0], f[1]) for f in foundSubexpressions]

            updateRules = newTerms
        else:
            for relaxationRate in relaxationRates:
                terms = [updateRule.rhs.coeff(relaxationRate) for updateRule in updateRules]
                resultOfCommonFactor = [trafos.extractMostCommonFactor(t) for t in terms]
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
        handledDirections += [direction, inverseDir]

    for term, substitutedVar in newCoefficientSubstitutions.items():
        substitutions.append(sp.Eq(substitutedVar, term))

    return lbmUpdateRule.newWithSubexpressions(result, substitutions, handledDirections)


# -------------------------------------- Helper Functions --------------------------------------------------------------

def __getCommonQuadraticAndConstantTerms(lbmUpdateRule):
    """Determines a common subexpression useful for most LBM model often called f_eq_common.
    It contains the quadratic and constant terms of the center update rule."""
    latticeModel = lbmUpdateRule.latticeModel
    center = tuple([0]*latticeModel.dim)
    t = lbmUpdateRule.updateEquations[latticeModel.stencil.index(center)].rhs
    for rp in latticeModel.relaxationRates:
        t = t.subs(rp, 1)

    for fa in latticeModel.pdfSymbols:
        t = t.subs(fa, 0)

    weight = t

    for u in latticeModel.symbolicVelocity:
        weight = weight.subs(u, 0)
    weight = weight / latticeModel.symbolicDensity
    return t / weight

