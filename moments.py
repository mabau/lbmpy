"""

Module Overview
~~~~~~~~~~~~~~~

This module provides functions to

- generate moments according to certain patterns
- compute moments of discrete probability distribution functions
- create transformation rules into moment space
- orthogonalize moment bases


Moment Representation
~~~~~~~~~~~~~~~~~~~~~

Moments can be represented in two ways:

- by an index :math:`i,j`: defined as :math:`m_{ij} := \sum_{\mathbf{d} \in stencil} <\mathbf{d}, \mathbf{x}> f_i`
- or by a polynomial in the variables x,y and z. For example the polynomial :math:`x^2 y^1 z^3 + x + 1` is
  describing the linear combination of moments: :math:`m_{213} + m_{100} + m_{000}`

The polynomial description is more powerful, since one polynomial can express a linear combination of single moments.
All moment polynomials have to use ``MOMENT_SYMBOLS`` (which is a module variable) as degrees of freedom.

Example ::

    from lbmpy.moments import MOMENT_SYMBOLS
    x, y, z = MOMENT_SYMBOLS
    secondOrderMoment = x*y + y*z


Functions
~~~~~~~~~

"""
import sympy as sp
import math
import itertools
import functools
from copy import copy
from collections import Counter

from lbmpy.continuous_distribution_measures import continuousMoment
from lbmpy_old.transformations import removeHigherOrderTerms

MOMENT_SYMBOLS = sp.symbols("x y z")


def __uniqueList(seq):
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


def __uniquePermutations(elements):
    if len(elements) == 1:
        yield (elements[0],)
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in __uniquePermutations(remaining_elements):
                yield (first_element,) + sub_permutation


def __generateFixedSumTuples(tupleLength, tupleSum, allowedValues=None, ordered=False):
    if not allowedValues:
        allowedValues = list(range(0, tupleSum + 1))

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


# ------------------------------ Discrete (Exponent Tuples) ------------------------------------------------------------


def momentMultiplicity(exponentTuple):
    """
    Returns number of permutations of the given moment tuple

    Example:
    >>> momentMultiplicity((2,0,0))
    3
    >>> list(momentPermutations((2,0,0)))
    [(0, 0, 2), (0, 2, 0), (2, 0, 0)]
    """
    c = Counter(exponentTuple)
    result = math.factorial(len(exponentTuple))
    for d in c.values():
        result //= math.factorial(d)
    return result


def momentPermutations(exponentTuple):
    """Returns all (unique) permutations of the given tuple"""
    return __uniquePermutations(exponentTuple)


def momentsOfOrder(order, dim=3, includePermutations=True):
    """All tuples of length 'dim' which sum equals 'order'"""
    yield from __generateFixedSumTuples(dim, order, ordered=not includePermutations)


def momentsUpToOrder(order, dim=3, includePermutations=True):
    """All tuples of length 'dim' which sum is smaller than 'order' """
    singleMomentIterators = [momentsOfOrder(o, dim, includePermutations) for o in range(order + 1)]
    return tuple(itertools.chain(*singleMomentIterators))


def momentsUpToComponentOrder(order, dim=3):
    """All tuples of length 'dim' where each entry is smaller or equal to 'order' """
    return tuple(itertools.product(*[range(order + 1)] * dim))


def extendMomentsWithPermutations(exponentTuples):
    """Returns all permutations of the given exponent tuples"""
    allMoments = []
    for i in exponentTuples:
        allMoments += list(momentPermutations(i))
    return __uniqueList(allMoments)


# ------------------------------ Representation Conversions ------------------------------------------------------------


def exponentToPolynomialRepresentation(exponentTuple):
    """
    Converts an exponent tuple to corresponding polynomial representation

    Example:
        >>> exponentToPolynomialRepresentation( (2,1,3) )
        x**2*y*z**3
    """
    poly = 1
    for sym, tupleEntry in zip(MOMENT_SYMBOLS[:len(exponentTuple)], exponentTuple):
        poly *= sym ** tupleEntry
    return poly


def exponentsToPolynomialRepresentations(sequenceOfExponentTuples):
    """Applies :func:`exponentToPolynomialRepresentation` to given sequence"""
    return tuple([exponentToPolynomialRepresentation(t) for t in sequenceOfExponentTuples])


def polynomialToExponentRepresentation(polynomial, dim=3):
    """
    Converts a linear combination of moments in polynomial representation into exponent representation

    :returns list of tuples where the first element is the coefficient and the second element is the exponent tuple

    Example:
        >>> x , y, z = MOMENT_SYMBOLS
        >>> polynomialToExponentRepresentation(1 + (42 * x**2 * y**2 * z) )
        [(1, (0, 0, 0)), (42, (2, 2, 1))]
    """
    assert dim <= 3
    x, y, z = MOMENT_SYMBOLS
    polynomial = polynomial.expand()
    coeffExpTupleRepresentation = []
    for expr, coefficient in polynomial.as_coefficients_dict().items():
        if len(expr.atoms(sp.Symbol) - set(MOMENT_SYMBOLS)) > 0:
            raise ValueError("Invalid moment polynomial: " + str(expr))
        x_exp, y_exp, z_exp = sp.Wild('xexp'), sp.Wild('yexp'), sp.Wild('zc')
        matchRes = expr.match(x**x_exp * y**y_exp * z**z_exp)
        assert matchRes[x_exp].is_integer and matchRes[y_exp].is_integer and matchRes[z_exp].is_integer
        expTuple = (int(matchRes[x_exp]), int(matchRes[y_exp]), int(matchRes[z_exp]),)
        if dim < 3:
            for i in range(dim, 3):
                assert expTuple[i] == 0, "Used symbols in polynomial are not representable in that dimension"
            expTuple = expTuple[:dim]
        coeffExpTupleRepresentation.append((coefficient, expTuple))
    return coeffExpTupleRepresentation


# -------------------- Common Function working with exponent tuples and polynomial moments -----------------------------


def isEven(moment):
    """
    A moment is considered even when under sign reversal nothing changes i.e. :math:`m(-x,-y,-z) = m(x,y,z)`

    For the exponent tuple representation that means that the exponent sum is even  e.g.
        >>> x , y, z = MOMENT_SYMBOLS
        >>> isEven(x**2 * y**2)
        True
        >>> isEven(x**2 * y)
        False
        >>> isEven((1,0,0))
        False
        >>> isEven(1)
        True
    """
    if type(moment) is tuple:
        return sum(moment) % 2 == 0
    else:
        moment = sp.sympify(moment)
        opposite = moment
        for s in MOMENT_SYMBOLS:
            opposite = opposite.subs(s, -s)
        return sp.expand(moment - opposite) == 0


def getOrder(moment):
    """
    Computes polynomial order of given moment

    Examples:
        >>> x , y, z = MOMENT_SYMBOLS
        >>> getOrder(x**2 * y + x)
        3
        >>> getOrder(z**4 * x**2)
        6
        >>> getOrder((2,1,0))
        3
    """
    if isinstance(moment, tuple):
        return sum(moment)
    if len(moment.atoms(sp.Symbol)) == 0:
        return 0
    leadingCoefficient = sp.polys.polytools.LM(moment)
    symbolsInLeadingCoefficient = leadingCoefficient.atoms(sp.Symbol)
    return sum([sp.degree(leadingCoefficient, gen=m) for m in symbolsInLeadingCoefficient])


def isShearMoment(moment):
    """Shear moments in 3D are: x*y, x*z and y*z - in 2D its only x*y"""
    if type(moment) is tuple:
        moment = exponentToPolynomialRepresentation(moment)
    return moment in isShearMoment.shearMoments
isShearMoment.shearMoments = set([c[0] * c[1] for c in itertools.combinations(MOMENT_SYMBOLS, 2)])


@functools.lru_cache(maxsize=512)
def discreteMoment(function, moment, stencil):
    """
    Computes discrete moment of given distribution function

    .. math ::
        \sum_{d \in stencil} p(d) f_i

    where :math:`p(d)` is the moment polynomial where :math:`x, y, z` have been replaced with the components of the
    stencil direction, and :math:`f_i` is the i'th entry in the passed function sequence

    :param function: list of distribution functions for each direction
    :param moment: can either be a exponent tuple, or a sympy polynomial expression
    :param stencil: sequence of directions
    """
    assert len(stencil) == len(function)
    res = 0
    for factor, e in zip(function, stencil):
        if type(moment) is tuple:
            for vel, exponent in zip(e, moment):
                factor *= vel ** exponent
            res += factor
        else:
            weight = moment
            for variable, e_i in zip(MOMENT_SYMBOLS, e):
                weight = weight.subs(variable, e_i)
            res += weight * factor

    return res


def momentMatrix(moments, stencil):
    """
    Returns transformation matrix to moment space

    each row corresponds to a moment, each column to a direction of the stencil
    The entry i,j is the i'th moment polynomial evaluated at direction j
    """

    if type(moments[0]) is tuple:
        def generator(row, column):
            result = sp.Rational(1, 1)
            for exponent, stencilEntry in zip(moments[row], stencil[column]):
                result *= int(stencilEntry ** exponent)
            return result
    else:
        def generator(row, column):
            evaluated = moments[row]
            for var, stencilEntry in zip(MOMENT_SYMBOLS, stencil[column]):
                evaluated = evaluated.subs(var, stencilEntry)
            return evaluated

    return sp.Matrix(len(moments), len(stencil), generator)


def gramSchmidt(moments, stencil, weights=None):
    """
    Computes orthogonal set of moments using the method by Gram-Schmidt

    :param moments: sequence of moments, either in tuple or polynomial form
    :param stencil: stencil as sequence of directions
    :param weights: optional weights, that define the scalar product which is used for normalization.
                    Scalar product :math:`< a,b > = \sum a_i b_i w_i` with weights :math:`w_i`.
                    Passing no weights sets all weights to 1.
    :return: set of orthogonal moments in polynomial form
    """
    if weights is None:
        weights = sp.eye(len(stencil))
    if type(weights) is list:
        assert len(weights) == len(stencil)
        weights = sp.diag(*weights)

    if type(moments[0]) is tuple:
        moments = list(exponentsToPolynomialRepresentations(moments))
    else:
        moments = list(copy(moments))

    M = momentMatrix(moments, stencil).transpose()
    columnsOfM = [M.col(i) for i in range(M.cols)]
    orthogonalizedVectors = []
    for i in range(len(columnsOfM)):
        currentElement = columnsOfM[i]
        for j in range(i):
            prevElement = orthogonalizedVectors[j]
            denom = prevElement.dot(weights * prevElement)
            if denom == 0:
                raise ValueError("Not an independent set of vectors given: "
                                 "vector %d is dependent on previous vectors" % (i,))
            overlap = currentElement.dot(weights * prevElement) / denom
            currentElement -= overlap * prevElement
            moments[i] -= overlap * moments[j]
        orthogonalizedVectors.append(currentElement)

    return moments


def momentEqualityTable(stencil, discreteEquilibrium=None, continuousEquilibrium=None, maxOrder=4, truncateOrder=None):
    """
    Creates a table showing which moments of a discrete stencil/equilibrium coincide with the
    corresponding continuous moments

    :param stencil: list of stencil velocities
    :param discreteEquilibrium: list of sympy expr to compute discrete equilibrium for each direction, if left
                                to default the standard discrete Maxwellian equilibrium is used
    :param continuousEquilibrium: continuous equilibrium, if left to default, the continuous Maxwellian is used
    :param maxOrder: compare moments up to this order (number of rows in table)
    :param truncateOrder: moments are considered equal if they match up to this order
    :return: Object to display in an Jupyter notebook
    """
    import ipy_table
    dim = len(stencil[0])

    if discreteEquilibrium is None:
        from lbmpy.maxwellian_equilibrium import discreteMaxwellianEquilibrium
        discreteEquilibrium = discreteMaxwellianEquilibrium(stencil, c_s_sq=sp.Rational(1, 3), compressible=True)
    if continuousEquilibrium is None:
        from lbmpy.maxwellian_equilibrium import continuousMaxwellianEquilibrium
        continuousEquilibrium = continuousMaxwellianEquilibrium(dim=dim, c_s_sq=sp.Rational(1, 3))

    table = []
    matchedMoments = 0
    nonMatchedMoments = 0

    momentsList = [list(momentsOfOrder(o, dim, includePermutations=False)) for o in range(maxOrder + 1)]

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
            dm = discreteMoment(discreteEquilibrium, moment, stencil)
            cm = continuousMoment(continuousEquilibrium, moment, symbols=sp.symbols("v_0 v_1 v_2")[:dim])
            difference = sp.simplify(dm - cm)
            if truncateOrder:
                difference = sp.simplify(removeHigherOrderTerms(difference, order=truncateOrder))
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
