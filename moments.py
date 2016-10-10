import sympy as sp
import itertools
import math
from collections import Counter
import functools
from lbmpy.util import getSymbolicVelocityVector, uniqueList
from lbmpy.transformations import makeExponentialFuncArgumentSquares

MOMENT_SYMBOLS = sp.symbols("x y z")


# ---------------------------------  Continuous ------------------------------------------------------------------------


@functools.lru_cache()
def __createMomentGeneratingFunction(function, symbols, newSymbols):
    assert len(symbols) == len(newSymbols)

    for t_i, v_i in zip(newSymbols, symbols):
        function *= sp.exp(t_i*v_i)

    # This is a custom transformation that speeds up the integrating process
    # of a MaxwellBoltzmann distribution
    # without this transformation the symbolic integration is sometimes not possible (e.g. in 2D without assumptions)
    # or is really slow
    # other functions should not be affected by this transformation
    # Without this transformation the following assumptions are required for the u and v variables of Maxwell Boltzmann
    #  2D: real=True ( without assumption it will not work)
    #  3D: no assumption ( with assumption it will not work :) )
    function = makeExponentialFuncArgumentSquares(function, symbols)

    bounds = [(s_i, -sp.oo, sp.oo) for s_i in symbols]
    result = sp.integrate(function, *bounds)

    return sp.simplify(result)


def __continuousCumulantOrMoment(function, exponents, symbols, cumulant):
    dim = len(exponents)

    if not symbols:
        symbols = getSymbolicVelocityVector(dim, "v")

    assert len(exponents) == len(symbols)

    t = getSymbolicVelocityVector(dim, "t")
    generatingFunction = __createMomentGeneratingFunction(function, symbols, t)

    if cumulant:
        generatingFunction = sp.ln(generatingFunction)

    diffArgs = []
    for order, t_i in zip(exponents, t):
        for i in range(order):
            diffArgs.append(t_i)

    if len(diffArgs) > 0:
        r = sp.diff(generatingFunction, *diffArgs)
    else:
        r = generatingFunction

    for t_i in t:
        r = r.subs(t_i, 0)

    return sp.simplify(r)


@functools.lru_cache(maxsize=512)
def continuousMoment(function, exponents, symbols=None):
    """
    Computes moment of given function

    to speed up the computation first a moment generating function is calculated and stored in a cache
    :param function: function to compute moments of
    :param exponents: power of the symbols
    :param symbols: tuple of symbols: computes moments w.r.t. these variables
                     defaults to v_0, v_1, ...
    """
    return __continuousCumulantOrMoment(function, exponents, symbols, False)


@functools.lru_cache(maxsize=512)
def continuousCumulant(function, exponents, symbols=None):
    """
    Computes cumulant of given function ( for parameters see function continuousMoment )
    """
    return __continuousCumulantOrMoment(function, exponents, symbols, True)


# ------------------------------ Discrete (Exponent Tuples) ------------------------------------------------------------


def __unique_permutations(elements):
    if len(elements) == 1:
        yield (elements[0],)
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in __unique_permutations(remaining_elements):
                yield (first_element,) + sub_permutation


def __generateFixedSumTuples(tupleLength, tupleSum, allowedValues=None, ordered=False):
    if not allowedValues:
        allowedValues = list(range(0, tupleSum+1))

    assert (0 in allowedValues)

    def recursive_helper(currentList, position, totalSum):
        newPosition = position + 1
        if newPosition < len(currentList):
            for i in allowedValues:
                currentList[position] = i
                newSum = totalSum - i
                if newSum < 0:
                    continue
                yield from recursive_helper(currentList, newPosition, newSum)
        else:
            if totalSum in allowedValues:
                currentList[-1] = totalSum
                if not ordered:
                    yield tuple(currentList)
                if ordered and currentList == sorted(currentList, reverse=True):
                    yield tuple(currentList)

    return recursive_helper([0] * tupleLength, 0, tupleSum)


def momentMultiplicity(exponentTuple):
    """Returns number of permutations of this moment"""
    c = Counter(exponentTuple)
    result = math.factorial(len(exponentTuple))
    for d in c.values():
        result //= math.factorial(d)
    return result


def momentPermutations(exponentTuple):
    return __unique_permutations(exponentTuple)


def momentsOfOrder(order, dim=3, includePermutations=True):
    """All tuples of length 'dim' which sum equals 'order'"""
    yield from __generateFixedSumTuples(dim, order, ordered=not includePermutations)


def momentsUpToOrder(order, dim=3, includePermutations=True):
    """All tuples of length 'dim' which sum is smaller than 'order' """
    singleMomentIterators = [momentsOfOrder(o, dim, includePermutations) for o in range(order+1)]
    return itertools.chain(*singleMomentIterators)


def momentsUpToComponentOrder(order, dim=3):
    """All tuples of length 'dim' where each entry is smaller or equal to 'order' """
    return itertools.product(*[range(order + 1)] * dim)


def extendMomentsWithPermutations(exponentTuples):
    """Returns all permutations of the given exponent tuples"""
    allMoments = []
    for i in exponentTuples:
        allMoments += list(momentPermutations(i))
    return uniqueList(allMoments)


def exponentTuplesToPolynomials(exponentTuples):
    """Converts an iterable of exponent tuples to a list of corresponding polynomial moments"""
    result = []
    for tup in exponentTuples:
        poly = 1
        for sym, tupleEntry in zip( MOMENT_SYMBOLS[:len(tup)], tup):
            poly *= sym ** tupleEntry
        result.append(poly)
    return result


# -------------------- Common Function working with exponent tuples and polynomial moments -----------------------------


def isEven(moment):
    """A moment is considered even when under sign reversal nothing changes i.e.
    m(-x,-y,-z) == m(x,y,z)
    For the exponent tuple representation that means that the exponent sum is even
    e.g. x**2 * y**2 is even
         x**2 * y    is odd
         x           is odd
         1           is even
    """
    if type(moment) is tuple:
        return sum(moment) % 2 == 0
    else:
        opposite = moment
        for s in MOMENT_SYMBOLS:
            opposite = opposite.subs(s, -s)
        return sp.simplify(moment-opposite) == 0


def isConservedMoment(moment):
    return moment == sp.Rational(1, 1) or moment in MOMENT_SYMBOLS


def getOrder(moment):
    """Computes polynomial order of given moment
    Examples: x**2 * y + x   ->  3
              z**4 * x**2    ->  6"""
    if len(moment.atoms(sp.Symbol)) == 0:
        return 0
    leadingCoefficient = sp.polys.polytools.LM(moment)
    symbolsInLeadingCoefficient = leadingCoefficient.atoms(sp.Symbol)
    return sum([sp.degree(leadingCoefficient, gen=m) for m in symbolsInLeadingCoefficient])


def discreteMoment(function, moment, stencil):
    """Computes discrete moment of given distribution function
    :param function     list of distribution functions for each direction
    :param moment       can either be a exponent tuple, or a sympy polynomial expression
                        e.g. first velocity moment can be either (1,0,0) or x
                        or a third order moment: (0,1,2) or y * z**2
    :param stencil      list of directions
    """
    assert len(stencil) == len(function)
    res = 0
    for factor, e in zip(function, stencil):
        if type(moment) is tuple:
            for vel, exponent in zip(e, moment):
                factor *= vel**exponent
            res += factor
        else:
            weight = moment
            for variable, e_i in zip(MOMENT_SYMBOLS, e):
                weight = weight.subs(variable, e_i)
            res += weight * factor

    return sp.simplify(res)


def momentMatrix(moments, stencil):
    """Returns transformation matrix to moment space"""

    if type(moments[0]) is tuple:
        def generator(row, column):
            result = sp.Rational(1, 1)
            for exponent, stencilEntry in zip(moments[row], stencil[column]):
                result *= int(stencilEntry**exponent)
            return result
    else:
        def generator(row, column):
            evaluated = moments[row]
            for var, stencilEntry in zip(MOMENT_SYMBOLS, stencil[column]):
                evaluated = evaluated.subs(var, stencilEntry)
            return evaluated

    return sp.Matrix(len(moments), len(stencil), generator)


def gramSchmidt(moments, stencil, weights=None):
    if weights is None:
        weights = sp.eye(len(stencil))
    if type(weights) is list:
        assert len(weights) == len(stencil)
        weights = sp.diag(*weights)

    if type(moments[0]) is tuple:
        moments = exponentTuplesToPolynomials(moments)
    else:
        from copy import copy
        moments = copy(moments)

    M = momentMatrix(moments, stencil).transpose()
    columnsOfM = [M.col(i) for i in range(M.cols)]
    result = []
    for i in range(len(columnsOfM)):
        currentElement = columnsOfM[i]
        for j in range(i):
            prevElement = result[j]
            denom = prevElement.dot(weights*prevElement)
            if denom == 0:
                raise ValueError("Not an independent set of vectors given: "
                                 "vector %d is dependent on previous vectors" %(i,))
            overlap = currentElement.dot(weights*prevElement) / denom
            currentElement -= overlap * prevElement
            moments[i] -= overlap * moments[j]
        result.append(currentElement)
    return moments, result


def momentEqualityTable(stencil, discreteEq, continuousEq, maxOrder=4, truncateOrder=2):
    """
    Creates a table showing which moments of a discrete stencil/equilibrium coincide with the
    corresponding continuous moments
    :param stencil: list of stencil velocities
    :param discreteEq: list of sympy expr to compute discrete equilibrium for each direction
    :param continuousEq: sympy expression for continuous equilibrium (usually this is a Maxwellian)
    :param maxOrder: compare moments up to this order
    :param truncateOrder: remove terms from continuous moments containing power of velocity higher than
                          this given order
    :return: Object to display in an IPython notebook
    """
    import ipy_table
    from lbmpy.transformations import removeHigherOrderTerms
    dim = len(stencil[0])

    table = []
    matchedMoments = 0
    nonMatchedMoments = 0

    momentsList = [list(momentsOfOrder(o, dim, includePermutations=False)) for o in range(maxOrder+1)]

    colors = dict()
    nrOfColumns = max([len(v) for v in momentsList]) + 1

    headerRow = [' '] * nrOfColumns
    headerRow[0] = 'order'
    table.append(headerRow)

    for order, moments in enumerate(momentsList):
        row = [' '] * nrOfColumns
        row[0] = '%d' % (order,)
        for moment, colIdx in zip(moments, range(1, len(row))):
            multiplicity = momentMultiplicity(moment)
            dm = discreteMoment(discreteEq, moment, stencil)
            cm = removeHigherOrderTerms(continuousMoment(continuousEq, moment), order=truncateOrder)
            difference = sp.simplify(dm - cm)
            if difference != 0:
                colors[(order + 1, colIdx)] = 'Orange'
                nonMatchedMoments += multiplicity
            else:
                colors[(order + 1, colIdx)] = 'lightGreen'
                matchedMoments += multiplicity

            row[colIdx] = '%s  x %d' % (moment, momentMultiplicity(moment))

        table.append(row)

    tableDisplay = ipy_table.make_table(table)
    ipy_table.set_row_style(0, color='#ddd')
    for cellIdx, color in colors.items():
        ipy_table.set_cell_style(cellIdx[0], cellIdx[1], color=color)

    print("Matched moments %d - non matched moments %d - total %d" %
          (matchedMoments, nonMatchedMoments, matchedMoments + nonMatchedMoments))

    return tableDisplay


# ---------------------------- Functions to get default set of moments -------------------------------------------------


def getDefaultMoments(number):
    x, y, z = MOMENT_SYMBOLS
    if number == 9:
        return [x**i * y**j for i in range(3) for j in range(3)]
    elif number == 27:
        return [x ** i * y ** j * z ** k for i in range(3) for j in range(3) for k in range(3)]

    raise Exception("No set of moments available")


class MomentSystem:

    def __init__(self, allMoments, momentIdGroups):
        self._allMoments = allMoments
        self._momentIdGroups = momentIdGroups

        if momentIdGroups is not None:
            # Extract and check conserved moments
            conservedMomentIds = momentIdGroups[0]
            for i in conservedMomentIds:
                assert isConservedMoment(allMoments[i])

            if len(allMoments) <= 9:
                assert len(self.conservedMomentIds) == 3
            else:
                assert len(self.conservedMomentIds) == 4

            # Check non-conserved moments: in one group all moments should have the same order
            for momentIdGroup in momentIdGroups[1:]:
                assert len(set([getOrder(allMoments[i]) for i in momentIdGroup])) == 1

    @property
    def hasMomentGroups(self):
        return self._momentIdGroups is not None

    @property
    def allMoments(self):
        return self._allMoments

    @property
    def conservedMomentIds(self):
        return self._momentIdGroups[0]

    @property
    def nonConservedMomentGroupIds(self):
        return self._momentIdGroups[1:]

    def getSymbolicRelaxationRates(self, ordering='lowestMomentId', namePrefix="s_"):
        assert self.hasMomentGroups
        result = [0] * len(self._allMoments)

        if ordering == 'lowestMomentId':
            for momentIdGroup in self.nonConservedMomentGroupIds:
                rate = sp.Symbol("%s%d" % (namePrefix, momentIdGroup[0]))
                for momentId in momentIdGroup:
                    result[momentId] = rate
        elif ordering == 'ascending':
            nextParam = 1
            for momentIdGroup in self.nonConservedMomentGroupIds:
                rate = sp.Symbol("%s%d" % (namePrefix, nextParam))
                nextParam += 1
                for momentId in momentIdGroup:
                    result[momentId] = rate
        return result


def getDefaultOrthogonalMoments(stencil):
    number = len(stencil)
    x, y, z = MOMENT_SYMBOLS

    if number == 9:
        import lbmpy.util as util
        import lbmpy.equilibria as eq
        defaultMoments = getDefaultMoments(number)
        orthoMoments, orthoVecs = gramSchmidt(defaultMoments, stencil, weights=eq.getWeights(stencil))
        orthoMomentsScaled = [e * util.commonDenominator(e) for e in orthoMoments]
        return MomentSystem(orthoMomentsScaled, None)
    elif number == 15:
        # from Khirevich, Ginzburg, Tallarek 2015: Coarse- and fine-grid numerical ...
        # took the D3Q19 moments and delete 16,17,18 and 10, 12 - renumbered - and added the last one
        # should be same as in above paper :)
        sq = x ** 2 + y ** 2 + z ** 2
        allMoments = [
            sp.Rational(1, 1),  # 0
            sq - 1,  # 1
            3 * sq ** 2 - 6 * sq + 1,  # 2
            x, (3 * sq - 5) * x,  # 3,4
            y, (3 * sq - 5) * y,  # 5,6
            z, (3 * sq - 5) * z,  # 7,8
            3 * x ** 2 - sq,  # 9
            y ** 2 - z ** 2,  # 10
            x * y, y * z, x * z,  # 11,12,13
            x*y*z,  # 14
        ]
        momentGroups = [[0, 3, 5, 7],
                        [9, 10, 11, 12, 13],
                        [1],
                        [2],
                        [4, 6, 8],
                        [14],
                        ]
        return MomentSystem(allMoments, momentGroups)
    elif number == 19:
        # from: Toelke, Freudiger, Krafczyk 2006: An adaptive scheme using hierarchical grids
        # and :
        sq = x ** 2 + y ** 2 + z ** 2
        allMoments = [
            sp.Rational(1, 1),  # 0
            sq - 1,  # 1
            3 * sq ** 2 - 6 * sq + 1,  # 2
            x, (3 * sq - 5) * x,  # 3,4
            y, (3 * sq - 5) * y,  # 5,6
            z, (3 * sq - 5) * z,  # 7,8
            3 * x ** 2 - sq,  # 9
            (2 * sq - 3) * (3 * x ** 2 - sq),  # 10
            y ** 2 - z ** 2,  # 11
            (2 * sq - 3) * (y ** 2 - z ** 2),  # 12
            x * y, y * z, x * z,  # 13,14,15
            (y ** 2 - z ** 2) * x,  # 16
            (z ** 2 - x ** 2) * y,  # 17
            (x ** 2 - y ** 2) * z,  # 18
        ]
        momentGroups = [[0, 3, 5, 7],
                        [9, 11, 13, 14, 15],
                        [1],
                        [2],
                        [4, 6, 8],
                        [10, 12],
                        [16, 17, 18]]
        return MomentSystem(allMoments, momentGroups)

    elif number == 27:
        # from Premnath, Banerjee 2012: On the three dimensional central moment LBM
        xsq, ysq, zsq = x**2, y**2, z**2
        a = [
            sp.Rational(1, 1),  # 0
            x, y, z,  # 1,2,3
            x * y, x * z, y * z,  # 4,5,6
            xsq - ysq, xsq - zsq, xsq + ysq + zsq,  # 7,8,9   # In original paper: ,  # 7,8,9
            x * ysq + x * zsq,  # 10
            xsq * y + y * zsq,  # 11
            xsq * z + ysq * z,  # 12
            x * ysq - x * zsq,  # 13
            xsq * y - y * zsq,  # 14
            xsq * z - ysq * z,  # 15
            x * y * z,  # 16
            xsq * ysq + xsq * zsq + ysq * zsq,  # 17
            xsq * ysq + xsq * zsq - 2 * ysq * zsq,  # 18   in original paper: xsq * ysq + xsq * zsq - ysq * zsq,  # 18
            xsq * ysq - xsq * zsq,  # 19
            xsq * y * z,  # 20
            x * ysq * z,  # 21
            z * y * zsq,  # 22
            x * ysq * zsq,  # 23
            xsq * y * zsq,  # 24
            xsq * ysq * z,  # 25
            xsq * ysq * zsq,  # 26
        ]
        allMoments = [
            sp.Rational(1, 1),  # 0
            x, y, z,  # 1, 2, 3
            x * y, x * z, y * z,  # 4, 5, 6
            xsq - ysq,  # 7
            (xsq + ysq + zsq) - 3 * zsq,  # 8
            (xsq + ysq + zsq) - 2,  # 9
            3 * (x * ysq + x * zsq) - 4 * x,  # 10
            3 * (xsq * y + y * zsq) - 4 * y,  # 11
            3 * (xsq * z + ysq * z) - 4 * z,  # 12
            x * ysq - x * zsq,  # 13
            xsq * y - y * zsq,  # 14
            xsq * z - ysq * z,  # 15
            x * y * z,  # 16
            3 * (xsq * ysq + xsq * zsq + ysq * zsq) - 4 * (xsq + ysq + zsq) + 4,  # 17
            3 * (xsq * ysq + xsq * zsq - 2 * ysq * zsq) - 2 * (2 * xsq - ysq - zsq),  # 18
            3 * (xsq * ysq - xsq * zsq) - 2 * (ysq - zsq),  # 19
            3 * (xsq*y*z) - 2 * (y*z), # 20
            3 * (x*ysq*z) - 2 * (x*z),  # 21
            3 * (x*y*zsq) - 2 * (x*y),  # 22
            9 * (x * ysq * zsq) - 6 * (x * ysq + x * zsq) + 4 * x,  # 23
            9 * (xsq * y * zsq) - 6 * (xsq * y + y * zsq) + 4 * y,  # 24
            9 * (xsq * ysq * z) - 6 * (xsq * z + ysq * z) + 4 * z,  # 25
            27 * (xsq * ysq * zsq) - 18 * (xsq * ysq + xsq * zsq + ysq * zsq) + 12 * (xsq + ysq + zsq) - 8,  # 26
        ]
        return MomentSystem(allMoments, None)

    raise Exception("No set of moments available")
