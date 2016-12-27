"""
This module provides functions to compute cumulants of discrete functions.
Additionally functions are provided to compute cumulants from moments and vice versa
"""

import functools
import sympy as sp

from lbmpy.moments import momentsUpToComponentOrder
from lbmpy.continuous_distribution_measures import multiDifferentiation


# ------------------------------------------- Internal Functions -------------------------------------------------------
from pystencils.sympyextensions import fastSubs


def __getIndexedSymbols(passedSymbols, prefix, indices):
    """If passed symbols is not None, they are returned, if they are None
    indexed symbols of the form {prefix}_{index} are returned"""
    try:
        dim = len(indices[0])
    except TypeError:
        dim = 1

    if passedSymbols is not None:
        return passedSymbols
    else:
        formatString = "%s_" + "_".join(["%d"]*dim)
        tupleIndices = []
        for i in indices:
            tupleIndices.append(i if type(i) is tuple else (i,))
        return [sp.Symbol(formatString % ((prefix,) + i)) for i in tupleIndices]


def __partition(collection):
    """Yields all set partitions of a given sequence"""
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in __partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        yield [[first]] + smaller


def __cumulantRawMomentTransform(index, dependentVarDict, outerFunction, defaultPrefix, centralized):
    """
    Function to express cumulants as function of moments as vice versa.
    Uses multivariate version of Faa di Bruno's formula.

    :param index: tuple describing the index of the cumulant/moment to express as function of moments/cumulants
    :param dependentVarDict: a dictionary from index tuple to moments/cumulants symbols, or None to use default symbols
    :param outerFunction: logarithm to transform from moments->cumulants, exp for inverse direction
    :param defaultPrefix: if dependentVarDict is None, this is used to construct symbols of the form prefix_i_j_k
    :param centralized: if True the first order moments/cumulants are set to zero
    """
    dim = len(index)
    subsDict = {}

    def createMomentSymbol(idx):
        nonlocal subsDict
        idx = tuple(idx)
        resultSymbol = sp.Symbol(defaultPrefix + "_" + "_".join(["%d"] * len(idx)) % idx)
        if dependentVarDict is not None and idx in dependentVarDict:
            subsDict[resultSymbol] = dependentVarDict[idx]
        return resultSymbol

    zerothMoment = createMomentSymbol((0,)*dim)

    def outerFunctionDerivative(n):
        x = zerothMoment
        return sp.diff(outerFunction(x), *tuple([x]*n))

    # index (2,1,0) means differentiate twice w.r.t to first variable, and once w.r.t to second variable
    # this is transformed here into representation [0,0,1] such that each entry is one diff operation
    partitionList = []
    for i, indexComponent in enumerate(index):
        for j in range(indexComponent):
            partitionList.append(i)

    if len(partitionList) == 0:  # special case for zero index
        return fastSubs(outerFunction(zerothMoment), subsDict)

    # implementation of Faa di Bruno's formula:
    result = 0
    for partition in __partition(partitionList):
        factor = outerFunctionDerivative(len(partition))
        for elements in partition:
            momentIndex = [0, ] * dim
            for i in elements:
                momentIndex[i] += 1
            factor *= createMomentSymbol(momentIndex)
        result += factor

    if centralized:
        for i in range(dim):
            index = [0] * dim
            index[i] = 1
            result = result.subs(createMomentSymbol(index), 0)

    return fastSubs(result, subsDict)


@functools.lru_cache(maxsize=16)
def __getDiscreteCumulantGeneratingFunction(function, stencil, waveNumbers):
    assert len(stencil) == len(function)

    def scalarProduct(a, b):
        return sum(a_i * b_i for a_i, b_i in zip(a, b))

    laplaceTrafo = sum([factor * sp.exp(scalarProduct(waveNumbers, e)) for factor, e in zip(function, stencil)])
    return sp.ln(laplaceTrafo)


# ------------------------------------------- Public Functions ---------------------------------------------------------


@functools.lru_cache(maxsize=64)
def discreteCumulant(function, cumulant, stencil):
    """
    Computes cumulant of discrete function

    :param function: sequence of function components, has to have the same length as stencil
    :param cumulant: definition of cumulant, either as an index tuple, or as a polynomial
                     (similar to moment description)
    :param stencil: sequence of directions
    """
    assert len(stencil) == len(function)

    dim = len(stencil[0])
    waveNumbers = tuple([sp.Symbol("Xi_%d" % (i,)) for i in range(dim)])

    generatingFunction = __getDiscreteCumulantGeneratingFunction(function, stencil, waveNumbers)
    if type(cumulant) is tuple:
        return multiDifferentiation(generatingFunction, cumulant, waveNumbers)
    else:
        from lbmpy.moments import MOMENT_SYMBOLS
        result = 0
        for term, coefficient in cumulant.as_coefficients_dict().items():
            exponents = tuple([term.as_coeff_exponent(v_i)[1] for v_i in MOMENT_SYMBOLS[:dim]])
            generatingFunction = __getDiscreteCumulantGeneratingFunction(function, stencil, waveNumbers)
            cm = multiDifferentiation(generatingFunction, exponents, waveNumbers)
            result += coefficient * cm
        return result


@functools.lru_cache(maxsize=8)
def cumulantsFromPdfs(stencil, cumulantIndices=None, pdfSymbols=None):
    """
    Transformation of pdfs (or other discrete function on a stencil) to cumulant space

    :param stencil:
    :param cumulantIndices: sequence of cumulant indices, could be tuples or polynomial representation
                            if left to default and a full stencil was passed,
                            the full set i.e. `momentsUpToComponentOrder(2)` is used
    :param pdfSymbols: symbolic values for pdf values, if not passed they default to :math:`f_0, f_1, ...`
    :return: dict mapping cumulant index to expression
    """
    dim = len(stencil[0])
    if cumulantIndices is None:
        cumulantIndices = momentsUpToComponentOrder(2, dim=dim)
    assert len(stencil) == len(cumulantIndices), "Stencil has to have same length as cumulantIndices sequence"
    pdfSymbols = __getIndexedSymbols(pdfSymbols, "f", range(len(stencil)))
    return {idx: discreteCumulant(tuple(pdfSymbols), idx, stencil) for idx in cumulantIndices}


def cumulantAsFunctionOfRawMoments(index, momentsDict=None):
    """
    Returns an expression for the cumulant of given index as a function of raw moments

    :param index: a tuple of same length as spatial dimensions, specifying the cumulant
    :param momentsDict: a dictionary that maps moment indices to symbols/values. These values are used for
                        the moments in the returned expression. If this parameter is None, default symbols are used.
    """
    return __cumulantRawMomentTransform(index, momentsDict, sp.log, 'm', False)


def rawMomentAsFunctionOfCumulants(index, cumulantsDict=None):
    """
    Inverse transformation of :func:`cumulantAsFunctionOfRawMoments`. All parameters are similar to this function.
    """
    return __cumulantRawMomentTransform(index, cumulantsDict, sp.exp, 'c', False)


def cumulantAsFunctionOfCentralMoments(index, momentsDict=None):
    """Same as :func:`cumulantAsFunctionOfRawMoments` but with central instead of raw moments."""
    return __cumulantRawMomentTransform(index, momentsDict, sp.log, 'm', True)


def centralMomentAsFunctionOfCumulants(index, cumulantsDict=None):
    """Same as :func:`rawMomentAsFunctionOfCumulants` but with central instead of raw moments."""
    return __cumulantRawMomentTransform(index, cumulantsDict, sp.exp, 'c', True)
