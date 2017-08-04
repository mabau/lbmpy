import warnings
from collections import namedtuple
import numpy as np
import sympy as sp
import functools
from sympy.core.cache import cacheit

from lbmpy.cache import diskcache
from lbmpy.chapman_enskog import DiffOperator, normalizeDiffOrder, chapmanEnskogDerivativeExpansion, \
    chapmanEnskogDerivativeRecombination
from lbmpy.chapman_enskog.derivative import collectDerivatives, createNestedDiff

from pystencils.sympyextensions import normalizeProduct, multidimensionalSummation, kroneckerDelta
from lbmpy.chapman_enskog import Diff, expandUsingLinearity, expandUsingProductRule
from lbmpy.moments import discreteMoment, momentMatrix, polynomialToExponentRepresentation, getMomentIndices
from pystencils.sympyextensions import productSymmetric


# --------------------------------------------- Helper Functions -------------------------------------------------------


def getExpandedName(originalObject, number):
    import warnings
    warnings.warn("Deprecated!")
    name = originalObject.name
    newName = name + "^{(%i)}" % (number,)
    return originalObject.func(newName)


def expandedSymbol(name, subscript=None, superscript=None):
    if subscript is not None:
        name += "_{%s}" % (subscript,)
    if superscript is not None:
        name += "^{(%s)}" % (superscript,)
    return sp.Symbol(name)

# --------------------------------   Summation Convention  -------------------------------------------------------------


def getOccurrenceCountOfIndex(term, index):
    if isinstance(term, Diff):
        return getOccurrenceCountOfIndex(term.arg, index) + (1 if term.label == index else 0)
    elif isinstance(term, sp.Symbol):
        return 1 if term.name.endswith("_" + str(index)) else 0
    else:
        return 0


def replaceIndex(term, oldIndex, newIndex):
    if isinstance(term, Diff):
        newArg = replaceIndex(term.arg, oldIndex, newIndex)
        newLabel = newIndex if term.label == oldIndex else term.label
        return Diff(newArg, newLabel, term.ceIdx)
    elif isinstance(term, sp.Symbol):
        if term.name.endswith("_" + str(oldIndex)):
            baseName = term.name[:-(len(str(oldIndex))+1)]
            return sp.Symbol(baseName + "_" + str(newIndex))
        else:
            return term
    else:
        newArgs = [replaceIndex(a, oldIndex, newIndex) for a in term.args]
        return term.func(*newArgs) if newArgs else term

# Problem: when there are more than two repeated indices... which one to replace?

# ----------------------------------------------------------------------------------------------------------------------


def momentAliasing(momentTuple):
    moment = list(momentTuple)
    result = []
    for element in moment:
        if element > 2:
            result.append(2 - (element % 2))
        else:
            result.append(element)
    return tuple(result)


class CeMoment(sp.Symbol):
    def __new__(cls, name, *args, **kwds):
        obj = CeMoment.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, momentTuple, ceIdx=-1):
        obj = super(CeMoment, cls).__xnew__(cls, name)
        obj.momentTuple = momentTuple
        while len(obj.momentTuple) < 3:
            obj.momentTuple = obj.momentTuple + (0,)
        obj.ceIdx = ceIdx
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def _hashable_content(self):
        superClassContents = list(super(CeMoment, self)._hashable_content())
        return tuple(superClassContents + [hash(repr(self.momentTuple)), hash(repr(self.ceIdx))])

    @property
    def indices(self):
        return getMomentIndices(self.momentTuple)

    def __getnewargs__(self):
        return self.name, self.momentTuple, self.ceIdx

    def _latex(self, printer, *args):
        coordNames = ['x', 'y', 'z']
        coordStr = []
        for i, comp in enumerate(self.momentTuple):
            coordStr += [coordNames[i]] * comp
        coordStr = "".join(coordStr)
        result = "{%s_{%s}" % (self.name, coordStr)
        if self.ceIdx >= 0:
            result += "^{(%d)}}" % (self.ceIdx,)
        else:
            result += "}"
        return result

    def __repr__(self):
        return "%s_(%d)_%s" % (self.name, self.ceIdx, self.momentTuple)

    def __str__(self):
        return "%s_(%d)_%s" % (self.name, self.ceIdx, self.momentTuple)


class LbMethodEqMoments:
    def __init__(self, lbMethod):
        self._eq = tuple(e.rhs for e in lbMethod.getEquilibrium().mainEquations)
        self._momentCache = dict()
        self._postCollisionMomentCache = dict()
        self._stencil = lbMethod.stencil
        self._inverseMomentMatrix = momentMatrix(lbMethod.moments, lbMethod.stencil).inv()
        self._method = lbMethod

    def __call__(self, ceMoment):
        return self.getPreCollisionMoment(ceMoment)

    def getPreCollisionMoment(self, ceMoment):
        assert ceMoment.ceIdx == 0, "Only equilibrium moments can be obtained with this function"
        if ceMoment not in self._momentCache:
            self._momentCache[ceMoment] = discreteMoment(self._eq, ceMoment.momentTuple, self._stencil)
        return self._momentCache[ceMoment]

    def getPostCollisionMoment(self, ceMoment, exponent=1, preCollisionMomentName="\\Pi"):
        if (ceMoment, exponent) in self._postCollisionMomentCache:
            return self._postCollisionMomentCache[(ceMoment, exponent)]

        stencil = self._method.stencil
        Minv = self._inverseMomentMatrix

        momentTuple = ceMoment.momentTuple

        momentSymbols = []
        for moment, (eqValue, rr) in self._method.relaxationInfoDict.items():
            if isinstance(moment, tuple):
                momentSymbols.append(-rr**exponent * CeMoment(preCollisionMomentName, moment, ceMoment.ceIdx))
            else:
                momentSymbols.append(-rr**exponent * sum(coeff * CeMoment(preCollisionMomentName, momentTuple, ceMoment.ceIdx)
                                     for coeff, momentTuple in polynomialToExponentRepresentation(moment)))
        momentSymbols = sp.Matrix(momentSymbols)
        postCollisionValue = discreteMoment(tuple(Minv * momentSymbols), momentTuple, stencil)
        self._postCollisionMomentCache[(ceMoment, exponent)] = postCollisionValue

        return postCollisionValue

    def substitutePreCollisionMoments(self, expr, preCollisionMomentName="\\Pi"):
        substitutions = {m: self.getPreCollisionMoment(m) for m in expr.atoms(CeMoment)
                         if m.ceIdx == 0 and m.name == preCollisionMomentName}
        return expr.subs(substitutions)

    def substitutePostCollisionMoments(self, expr, preCollisionMomentName="\\Pi", postCollisionMomentName="\\Upsilon"):
        """
        Substitutes post-collision equilibrium moments 
        :param expr: expression with fully expanded derivatives
        :param preCollisionMomentName: post-collision moments are replaced by CeMoments with this name
        :param postCollisionMomentName: name of post-collision CeMoments
        :return: expressions where equilibrium post-collision moments have been replaced
        """
        expr = sp.expand(expr)

        def visit(node, exponent):
            if node.func == sp.Pow:
                base, exp = node.args
                return visit(base, exp)
            elif isinstance(node, CeMoment) and node.name == postCollisionMomentName:
                return self.getPostCollisionMoment(node, exponent, preCollisionMomentName)
            else:
                return node**exponent if not node.args else node.func(*[visit(k, 1) for k in node.args])
        return visit(expr, 1)

    def substitute(self, expr,  preCollisionMomentName="\\Pi", postCollisionMomentName="\\Upsilon"):
        result = self.substitutePostCollisionMoments(expr, preCollisionMomentName, postCollisionMomentName)
        result = self.substitutePreCollisionMoments(result, preCollisionMomentName)
        return result


def insertMoments(eqn, lbMethodMoments, momentName="\\Pi", useSolvabilityConditions=True):
    subsDict = {}
    if useSolvabilityConditions:
        condition = lambda m:  m.ceIdx > 0 and sum(m.momentTuple) <= 1 and m.name == momentName
        subsDict.update({m: 0 for m in eqn.atoms(CeMoment) if condition(m)})

    condition = lambda m:  m.ceIdx == 0 and m.name == momentName
    subsDict.update({m: lbMethodMoments(m) for m in eqn.atoms(CeMoment) if condition(m)})
    return eqn.subs(subsDict)


def substituteCollisionOperatorMoments(expr, lbMomentComputation, collisionOpMomentName='\\Upsilon',
                                       preCollisionMomentName="\\Pi"):
    momentsToReplace = [m for m in expr.atoms(CeMoment) if m.name == collisionOpMomentName]
    subsDict = {}
    for ceMoment in momentsToReplace:
        subsDict[ceMoment] = lbMomentComputation.getPostCollisionMoment(ceMoment, 1, preCollisionMomentName)

    return expr.subs(subsDict)


def takeMoments(eqn, pdfToMomentName=(('f', '\Pi'), ('\Omega f', '\\Upsilon')), velocityName='c', maxExpansion=5,
                useOneNeighborhoodAliasing=False):

    pdfSymbols = [tuple(expandedSymbol(name, superscript=i) for i in range(maxExpansion))
                  for name, _ in pdfToMomentName]

    velocityTerms = tuple(expandedSymbol(velocityName, subscript=i) for i in range(3))

    def determineFIndex(factor):
        FIndex = namedtuple("FIndex", ['momentName', 'ceIdx'])
        for symbolListId, pdfSymbolsElement in enumerate(pdfSymbols):
            try:
                return FIndex(pdfToMomentName[symbolListId][1], pdfSymbolsElement.index(factor))
            except ValueError:
                pass
        return None

    def handleProduct(productTerm):
        fIndex = None
        derivativeTerm = None
        cIndices = []
        rest = 1
        for factor in normalizeProduct(productTerm):
            if isinstance(factor, Diff):
                assert fIndex is None
                fIndex = determineFIndex(factor.getArgRecursive())
                derivativeTerm = factor
            elif factor in velocityTerms:
                cIndices += [velocityTerms.index(factor)]
            else:
                newFIndex = determineFIndex(factor)
                if newFIndex is None:
                    rest *= factor
                else:
                    assert not(newFIndex and fIndex)
                    fIndex = newFIndex

        momentTuple = [0] * len(velocityTerms)
        for cIdx in cIndices:
            momentTuple[cIdx] += 1
        momentTuple = tuple(momentTuple)

        if useOneNeighborhoodAliasing:
            momentTuple = momentAliasing(momentTuple)
        result = CeMoment(fIndex.momentName, momentTuple, fIndex.ceIdx)
        if derivativeTerm is not None:
            result = derivativeTerm.changeArgRecursive(result)
        result *= rest
        return result

    functions = sum(pdfSymbols, ())
    eqn = expandUsingLinearity(eqn, functions).expand()

    if eqn.func == sp.Mul:
        return handleProduct(eqn)
    else:
        assert eqn.func == sp.Add
        return sum(handleProduct(t) for t in eqn.args)


def timeDiffSelector(eq):
    return [d for d in eq.atoms(Diff) if d.label == sp.Symbol("t")]


def momentSelector(eq):
    return list(eq.atoms(CeMoment))


def diffExpandNormalizer(eq):
    return expandUsingProductRule(eq).expand()


def chainSolveAndSubstitute(eqSequence, unknownSelector, normalizingFunc=diffExpandNormalizer):
    """Takes a list (hierarchy) of equations and does the following:
       Loops over given equations and for every equation:
        - normalizes the equation with the provided normalizingFunc
        - substitute symbols that have been already solved for
        - calls the unknownSelector function with an equation. This function should return a list of unknown symbols,
          and has to have length 0 or 1
        - if unknown was returned, the equation is solved for, and the pair (unknown-> solution) is entered into the dict
    """
    resultEquations = []
    subsDict = {}
    for i, eq in enumerate(eqSequence):
        eq = normalizingFunc(eq)
        eq = eq.subs(subsDict)
        eq = normalizingFunc(eq)
        resultEquations.append(eq)

        symbolsToSolveFor = unknownSelector(eq)
        if len(symbolsToSolveFor) == 0:
            continue
        assert len(symbolsToSolveFor) <= 1, "Unknown Selector return multiple unknowns - expected <=1\n" + str(
            symbolsToSolveFor)
        symbolToSolveFor = symbolsToSolveFor[0]
        solveRes = sp.solve(eq, symbolToSolveFor)
        assert len(solveRes) == 1, "Could not solve uniquely for unknown" + str(symbolToSolveFor)
        subsDict[symbolToSolveFor] = normalizingFunc(solveRes[0])
    return resultEquations, subsDict


def countVars(expr, variables):
    factorList = normalizeProduct(expr)
    diffsToUnpack = [e for e in factorList if isinstance(e, Diff)]
    factorList = [e for e in factorList if not isinstance(e, Diff)]

    while diffsToUnpack:
        d = diffsToUnpack.pop()
        args = normalizeProduct(d.arg)
        for a in args:
            if isinstance(a, Diff):
                diffsToUnpack.append(a)
            else:
                factorList.append(a)

    result = 0
    for v in variables:
        result += factorList.count(v)
    return result


def removeHigherOrderU(expr, order=1, u=sp.symbols("u_:3")):
    return sum(a for a in expr.args if countVars(a, u) <= order)


def removeErrorTerms(expr):
    rhoDiffsToZero = {Diff(sp.Symbol("rho"), i): 0 for i in range(3)}
    expr = expr.subs(rhoDiffsToZero)
    if isinstance(expr, sp.Matrix):
        expr = expr.applyfunc(removeHigherOrderU)
    else:
        expr = removeHigherOrderU(expr.expand())
    return sp.cancel(expr.expand())

# ----------------------------------------------------------------------------------------------------------------------


def getTaylorExpandedLbEquation(pdfSymbolName="f", pdfsAfterCollisionOperator="\Omega f", velocityName="c",
                                dim=3, taylorOrder=2, shift=True):
    dimLabels = [sp.Rational(i, 1) for i in range(dim)]

    c = sp.Matrix([expandedSymbol(velocityName, subscript=label) for label in dimLabels])
    dt, t = sp.symbols("Delta_t t")
    pdf = sp.Symbol(pdfSymbolName)
    collidedPdf = sp.Symbol(pdfsAfterCollisionOperator)

    Dt = DiffOperator(label=t)
    Dx = sp.Matrix([DiffOperator(label=l) for l in dimLabels])

    taylorOperator = sum(dt ** k * (Dt + c.dot(Dx)) ** k / sp.functions.factorial(k)
                         for k in range(1, taylorOrder + 1))

    functions = [pdf, collidedPdf]
    eq_4_5 = taylorOperator - dt * collidedPdf
    applied_eq_4_5 = expandUsingLinearity(DiffOperator.apply(eq_4_5, pdf), functions)

    if shift:
        operator = ((dt / 2) * (Dt + c.dot(Dx))).expand()
        opTimesEq_4_5 = expandUsingLinearity(DiffOperator.apply(operator, applied_eq_4_5), functions).expand()
        opTimesEq_4_5 = normalizeDiffOrder(opTimesEq_4_5, functions)
        eq_4_7 = (applied_eq_4_5 - opTimesEq_4_5).subs(dt ** (taylorOrder+1), 0)
    else:
        eq_4_7 = applied_eq_4_5.subs(dt ** (taylorOrder + 1), 0)

    eq_4_7 = eq_4_7.subs(dt, 1)
    return eq_4_7.expand()


def useChapmanEnskogAnsatz(equation, timeDerivativeOrders=(1, 3), spatialDerivativeOrders=(1, 2),
                           pdfs=(['f', 0, 3], ['\Omega f', 1, 3])):

    t, eps = sp.symbols("t epsilon")

    # expand time derivatives
    if timeDerivativeOrders:
        equation = chapmanEnskogDerivativeExpansion(equation, t, eps, *timeDerivativeOrders)

    # expand spatial derivatives
    if spatialDerivativeOrders:
        spatialDerivatives = [a for a in equation.atoms(Diff) if str(a.label) != 't']
        labels = set(a.label for a in spatialDerivatives)
        for label in labels:
            equation = chapmanEnskogDerivativeExpansion(equation, label, eps, *spatialDerivativeOrders)

    # expand pdfs
    subsDict = {}
    expandedPdfSymbols = []

    maxExpansionOrder = spatialDerivativeOrders[1] if spatialDerivativeOrders else 10
    for pdfName, startOrder, stopOrder in pdfs:
        if isinstance(pdfName, sp.Symbol):
            pdfName = pdfName.name
        expandedPdfSymbols += [expandedSymbol(pdfName, superscript=i) for i in range(startOrder, stopOrder)]
        subsDict[sp.Symbol(pdfName)] = sum(eps ** i * expandedSymbol(pdfName, superscript=i)
                                           for i in range(startOrder, stopOrder))
        maxExpansionOrder = max(maxExpansionOrder, stopOrder)
    equation = equation.subs(subsDict)
    equation = expandUsingLinearity(equation, functions=expandedPdfSymbols).expand().collect(eps)
    result = {epsOrder: equation.coeff(eps ** epsOrder) for epsOrder in range(1, 2*maxExpansionOrder)}
    result[0] = equation.subs(eps, 0)
    return result


def matchEquationsToNavierStokes(conservationEquations, rho=sp.Symbol("rho"), u=sp.symbols("u_:3"), t=sp.Symbol("t")):
    dim = len(conservationEquations) - 1
    u = u[:dim]
    funcs = u + (rho,)

    def diffSimplify(eq):
        variables = eq.atoms(CeMoment)
        variables.update(funcs)
        return expandUsingProductRule(expandUsingLinearity(eq, variables)).expand()

    def matchContinuityEq(continuityEq):
        continuityEq = diffSimplify(continuityEq)
        compressible = u[0] * Diff(rho, 0) in continuityEq.args
        factor = rho if compressible else 1
        refContinuityEq = diffSimplify(Diff(rho, t) + sum(Diff(factor * u[i], i) for i in range(dim)))
        return refContinuityEq - continuityEq, compressible

    def matchMomentEqs(momentEqs, compressible):
        shearAndPressureEqs = []
        for i, momEq in enumerate(momentEqs):
            factor = rho if compressible else 1
            ref = diffSimplify(Diff(factor * u[i], t) + sum(Diff(factor * u[i] * u[j], j) for j in range(dim)))
            shearAndPressureEqs.append(diffSimplify(momentEqs[i]) - ref)

        # extract pressure term
        coefficentArgSets = []
        for i, eq in enumerate(shearAndPressureEqs):
            coefficentArgSets.append(set())
            eq = eq.expand()
            assert eq.func == sp.Add
            for term in eq.args:
                if term.atoms(CeMoment):
                    continue
                candidateList = [e for e in term.atoms(Diff) if e.label == i]
                if len(candidateList) != 1:
                    continue
                coefficentArgSets[i].add((term / candidateList[0], candidateList[0].arg))
        pressureTerms = set.intersection(*coefficentArgSets)

        sigma = sp.zeros(dim)
        errorTerms = []
        for i, shearAndPressureEq in enumerate(shearAndPressureEqs):
            eqWithoutPressure = shearAndPressureEq - sum(coeff * Diff(arg, i) for coeff, arg in pressureTerms)
            for d in eqWithoutPressure.atoms(Diff):
                eqWithoutPressure = eqWithoutPressure.collect(d)
                sigma[i, d.label] += eqWithoutPressure.coeff(d) * d.arg
                eqWithoutPressure = eqWithoutPressure.subs(d, 0)

            errorTerms.append(eqWithoutPressure)
        pressure = [coeff * arg for coeff, arg in pressureTerms]

        return pressure, sigma, errorTerms

    continuityErrorTerms, compressible = matchContinuityEq(conservationEquations[0])
    pressure, sigma, momentErrorTerms = matchMomentEqs(conservationEquations[1:], compressible)

    errorTerms = [continuityErrorTerms] + momentErrorTerms
    for et in errorTerms:
        assert et == 0

    return compressible, pressure, sigma


@diskcache
def computeHigherOrderMomentSubsDict(momentEquations):
    oEpsWithoutTimeDiffs, timeDiffSubstitutions = chainSolveAndSubstitute(momentEquations, timeDiffSelector)
    momentsToSolveFor = set()
    pi_ab_equations = []
    for eq in oEpsWithoutTimeDiffs:
        foundMoments = momentSelector(eq)
        if foundMoments:
            momentsToSolveFor.update(foundMoments)
            pi_ab_equations.append(eq)
    return sp.solve(pi_ab_equations, momentsToSolveFor)


class ChapmanEnskogAnalysis(object):

    def __init__(self, method, constants=None):
        cqc = method.conservedQuantityComputation
        self._method = method
        self._momentCache = LbMethodEqMoments(method)
        self.rho = cqc.definedSymbols(order=0)[1]
        self.u = cqc.definedSymbols(order=1)[1]
        self.t = sp.Symbol("t")
        self.epsilon = sp.Symbol("epsilon")

        tayloredLbEq = getTaylorExpandedLbEquation(dim=self._method.dim)
        self.equationsGroupedByOrder = useChapmanEnskogAnsatz(tayloredLbEq)

        # Taking moments
        c = sp.Matrix([expandedSymbol("c", subscript=i) for i in range(self._method.dim)])
        momentsUntilOrder1 = [1] + list(c)
        momentsOrder2 = [c_i * c_j for c_i, c_j in productSymmetric(c, c)]

        symbolicRelaxationRates = [rr for rr in method.relaxationRates if isinstance(rr, sp.Symbol)]
        if constants is None:
            constants = set(symbolicRelaxationRates)
        else:
            constants.update(symbolicRelaxationRates)

        oEpsMoments1 = [expandUsingLinearity(self._takeAndInsertMoments(self.equationsGroupedByOrder[1] * moment),
                                             constants=constants)
                        for moment in momentsUntilOrder1]
        oEpsMoments2 = [expandUsingLinearity(self._takeAndInsertMoments(self.equationsGroupedByOrder[1] * moment),
                                             constants=constants)
                        for moment in momentsOrder2]
        oEpsSqMoments1 = [expandUsingLinearity(self._takeAndInsertMoments(self.equationsGroupedByOrder[2] * moment),
                                               constants=constants)
                          for moment in momentsUntilOrder1]

        self._equationsWithHigherOrderMoments = [self._ceRecombine(ord1 * self.epsilon + ord2 * self.epsilon ** 2)
                                                 for ord1, ord2 in zip(oEpsMoments1, oEpsSqMoments1)]

        self._higherOrderMomentSubsDict = computeHigherOrderMomentSubsDict(tuple(oEpsMoments1 + oEpsMoments2))

        # Match to Navier stokes
        compressible, pressure, sigma = matchEquationsToNavierStokes(self._equationsWithHigherOrderMoments)
        self.compressible = compressible
        self.pressureEquation = pressure
        self._sigmaWithHigherOrderMoments = sigma
        self._sigma = sigma.subs(self._higherOrderMomentSubsDict).expand().applyfunc(self._ceRecombine)
        self._sigmaWithoutErrorTerms = removeErrorTerms(self._sigma)

    def getMacroscopicEquations(self, substituteHigherOrderMoments=False):
        if substituteHigherOrderMoments:
            return self._equationsWithHigherOrderMoments.subs(self._higherOrderMomentSubsDict)
        else:
            return self._equationsWithHigherOrderMoments

    def getViscousStressTensor(self, substituteHigherOrderMoments=True):
        if substituteHigherOrderMoments:
            return self._sigma
        else:
            return self._sigmaWithHigherOrderMoments

    def _takeAndInsertMoments(self, eq):
        eq = takeMoments(eq)
        eq = substituteCollisionOperatorMoments(eq, self._momentCache)
        return insertMoments(eq, self._momentCache).expand()

    def _ceRecombine(self, expr):
        expr = chapmanEnskogDerivativeRecombination(expr, self.t, stopOrder=3)
        for l in range(self._method.dim):
            expr = chapmanEnskogDerivativeRecombination(expr, l, stopOrder=2)
        return expr

    def getDynamicViscosity(self):
        candidates = self.getShearViscosityCandidates()
        if len(candidates) != 1:
            raise ValueError("Could not find expression for kinematic viscosity. "
                             "Probably method does not approximate Navier Stokes.")
        return candidates.pop()

    def getKinematicViscosity(self):
        if self.compressible:
            return (self.getDynamicViscosity() / self.rho).expand()
        else:
            return self.getDynamicViscosity()

    def getShearViscosityCandidates(self):
        result = set()
        dim = self._method.dim
        for i, j in productSymmetric(range(dim), range(dim), withDiagonal=False):
            result.add(-sp.cancel(self._sigmaWithoutErrorTerms[i, j] / (Diff(self.u[i], j) + Diff(self.u[j], i))))
        return result

    def doesApproximateNavierStokes(self):
        """Returns a set of equations that are required in order for the method to approximate Navier Stokes equations
        up to second order"""
        conditions = set([0])
        dim = self._method.dim
        assert dim > 1
        # Check that shear viscosity does not depend on any u derivatives - create conditions (equations) that
        # have to be fulfilled for this to be the case
        viscosityReference = self._sigmaWithoutErrorTerms[0, 1].expand().coeff(Diff(self.u[0], 1))
        for i, j in productSymmetric(range(dim), range(dim), withDiagonal=False):
            term = self._sigmaWithoutErrorTerms[i, j]
            equalCrossTermCondition = sp.expand(term.coeff(Diff(self.u[i], j)) - viscosityReference)
            term = term.subs({Diff(self.u[i], j): 0,
                              Diff(self.u[j], i): 0})

            conditions.add(equalCrossTermCondition)
            for k in range(dim):
                symmetricTermCondition = term.coeff(Diff(self.u[k], k))
                conditions.add(symmetricTermCondition)
            term = term.subs({Diff(self.u[k], k): 0 for k in range(dim)})
            conditions.add(term)

        bulkCandidates = list(self.getBulkViscosityCandidates(-viscosityReference))
        if len(bulkCandidates) > 0:
            for i in range(1, len(bulkCandidates)):
                conditions.add(bulkCandidates[0] - bulkCandidates[i])

        return conditions

    def getBulkViscosityCandidates(self, viscosity=None):
        sigma = self._sigmaWithoutErrorTerms
        assert self._sigmaWithHigherOrderMoments.is_square
        result = set()
        if viscosity is None:
            viscosity = self.getDynamicViscosity()
        for i in range(sigma.shape[0]):
            bulkTerm = sigma[i, i] + 2 * viscosity * Diff(self.u[i], i)
            bulkTerm = bulkTerm.expand()
            for d in bulkTerm.atoms(Diff):
                bulkTerm = bulkTerm.collect(d)
                result.add(bulkTerm.coeff(d))
                bulkTerm = bulkTerm.subs(d, 0)
            if bulkTerm != 0:
                return set()
        if len(result) == 0:
            result.add(0)
        return result

    def getBulkViscosity(self):
        candidates = self.getBulkViscosityCandidates()
        if len(candidates) != 1:
            raise ValueError("Could not find expression for bulk viscosity. "
                             "Probably method does not approximate Navier Stokes.")

        viscosity = self.getDynamicViscosity()
        return (candidates.pop() + 2 * viscosity / 3).expand()

    def relaxationRateFromKinematicViscosity(self, nu):
        kinematicViscosity = self.getKinematicViscosity()
        solveRes = sp.solve(kinematicViscosity - nu, kinematicViscosity.atoms(sp.Symbol), dict=True)
        return solveRes[0]


class SteadyStateChapmanEnskogAnalysis(object):
    """Experimental:
        -> does not yield correct results for MRT, only SRT
        -> does predict different value for bulk viscosity?! 
    """

    def __init__(self, method, order=4):
        self.method = method
        dim = method.dim
        momentComputation = LbMethodEqMoments(method)

        eps, B, f, dt = sp.symbols("epsilon B f Delta_t")
        self.dt = dt
        expandedPdfSymbols = [expandedSymbol("f", superscript=i) for i in range(0, order + 1)]
        feq = expandedPdfSymbols[0]
        c = sp.Matrix([expandedSymbol("c", subscript=i) for i in range(dim)])
        Dx = sp.Matrix([DiffOperator(label=l) for l in range(dim)])
        differentialOperator = sum((dt * eps * c.dot(Dx)) ** n / sp.factorial(n) for n in range(1, order + 1))
        taylorExpansion = DiffOperator.apply(differentialOperator.expand(), f)
        epsDict = useChapmanEnskogAnsatz(taylorExpansion,
                                         spatialDerivativeOrders=None,  # do not expand the differential operator itself
                                         pdfs=(['f', 0, order + 1],))  # expand only the 'f' terms

        self.scaleHierarchy = [-B * epsDict[i] for i in range(0, order+1)]
        self.scaleHierarchyRaw = self.scaleHierarchy.copy()

        expandedPdfs = [feq, self.scaleHierarchy[1]]
        subsDict = {expandedPdfSymbols[1]: self.scaleHierarchy[1]}
        for i in range(2, 5):
            eq = self.scaleHierarchy[i].subs(subsDict)
            eq = expandUsingLinearity(eq, functions=expandedPdfSymbols)
            eq = normalizeDiffOrder(eq, functions=expandedPdfSymbols)
            subsDict[expandedPdfSymbols[i]] = eq
            expandedPdfs.append(eq)
        self.scaleHierarchy = expandedPdfs

        constants = sp.Matrix(method.relaxationRates).atoms(sp.Symbol)
        recombined = -sum(self.scaleHierarchy[n] for n in range(1, order + 1))  # Eq 18a
        recombined = sp.cancel(recombined / (dt * B)).expand()  # cancel common factors

        def handlePostcollisionValues(eq):
            eq = eq.expand()
            assert isinstance(eq, sp.Add)
            result = 0
            for summand in eq.args:

                moment = summand.atoms(CeMoment)
                moment = moment.pop()
                collisionOperatorExponent = normalizeProduct(summand).count(B)
                if collisionOperatorExponent == 0:
                    result += summand
                else:
                    substitutions = {
                        B: 1,
                        moment: -momentComputation.getPostCollisionMoment(moment, -collisionOperatorExponent),
                    }
                    result += summand.subs(substitutions)

            return result

        # Continuity equation (mass transport)
        contEq = takeMoments(recombined, maxExpansion=(order + 1) * 2)
        contEq = handlePostcollisionValues(contEq)
        contEq = expandUsingLinearity(contEq, constants=constants).expand().collect(dt)
        self.continuityEquationWithMoments = contEq
        contEq = insertMoments(contEq, momentComputation, useSolvabilityConditions=False)
        contEq = expandUsingLinearity(contEq, constants=constants).expand().collect(dt)
        self.continuityEquation = contEq

        # Momentum equation (momentum transport)
        self.momentumEquationsWithMoments = []
        self.momentumEquations = []
        for h in range(dim):
            momEq = takeMoments(recombined * c[h], maxExpansion=(order + 1) * 2)
            momEq = handlePostcollisionValues(momEq)
            momEq = expandUsingLinearity(momEq, constants=constants).expand().collect(dt)
            self.momentumEquationsWithMoments.append(momEq)
            momEq = insertMoments(momEq, momentComputation, useSolvabilityConditions=False)
            momEq = expandUsingLinearity(momEq, constants=constants).expand().collect(dt)
            self.momentumEquations.append(momEq)

    def getMassTransportEquation(self, order):
        if order == 0:
            result = self.continuityEquation.subs(self.dt, 0)
        else:
            result = self.continuityEquation.coeff(self.dt ** order)
        return collectDerivatives(result)

    def getMomentumTransportEquation(self, coordinate, order):
        if order == 0:
            result = self.momentumEquations[coordinate].subs(self.dt, 0)
        else:
            result = self.momentumEquations[coordinate].coeff(self.dt ** order)
        return collectDerivatives(result)

    def determineViscosities(self, coordinate):
        """
        Matches the first order term of the momentum equation to Navier stokes
        Automatically neglects higher order velocity terms and rho derivatives

        The bulk viscosity is predicted differently than by the normal Navier Stokes analysis...why??

        :param coordinate: which momentum equation to use i.e. x,y or z, to approximate Navier Stokes
                           all have to return the same result
        """
        dim = self.method.dim

        D = createNestedDiff
        s = functools.partial(multidimensionalSummation, dim=dim)
        kd = kroneckerDelta

        eta, eta_b = sp.symbols("nu nu_B")
        u = sp.symbols("u_:3")[:dim]
        a = coordinate
        navierStokesRef = eta * sum(D(b, b, arg=u[a]) + D(b, a, arg=u[b]) for b, in s(1)) + \
                          (eta_b - 2 * eta / 3) * sum(D(b, g, arg=u[g]) * kd(a, b) for b, g in s(2))
        navierStokesRef = -navierStokesRef.expand()

        firstOrderTerms = self.getMomentumTransportEquation(coordinate, order=1)
        firstOrderTerms = removeHigherOrderU(firstOrderTerms)
        firstOrderTerms = expandUsingLinearity(firstOrderTerms, constants=[sp.Symbol("rho")])

        matchCoeffEquations = []
        for diff in navierStokesRef.atoms(Diff):
            matchCoeffEquations.append(navierStokesRef.coeff(diff) - firstOrderTerms.coeff(diff))
        return sp.solve(matchCoeffEquations, [eta, eta_b])


if __name__ == '__main__':
    from lbmpy.creationfunctions import createLatticeBoltzmannMethod
    m = createLatticeBoltzmannMethod(stencil='D2Q9')
    ce = ChapmanEnskogAnalysis(m)