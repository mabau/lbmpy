"""
This module provides functions to compute cumulants of discrete functions.
Additionally functions are provided to compute cumulants from moments and vice versa
"""

import functools
import sympy as sp

from lbmpy.diskcache import diskcache
from lbmpy.moments import momentsUpToComponentOrder
from lbmpy.continuous_distribution_measures import multiDifferentiation


# ------------------------------------------- Internal Functions -------------------------------------------------------


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

    def createMomentSymbol(idx):
        idx = tuple(idx)
        if dependentVarDict is None:
            return sp.Symbol(defaultPrefix + "_" + "_".join(["%d"] * len(idx)) % idx)
        else:
            return dependentVarDict[idx]
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
        return outerFunction(zerothMoment)

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
        for i in dim:
            index = [0] * dim
            index[i] = 1
            result = result.subs(createMomentSymbol(index), 0)

    return result


@functools.lru_cache(maxsize=16)
def __getDiscreteCumulantGeneratingFunction(function, stencil, waveNumbers):
    assert len(stencil) == len(function)

    def scalarProduct(a, b):
        return sum(a_i * b_i for a_i, b_i in zip(a, b))

    laplaceTrafo = sum([factor * sp.exp(scalarProduct(waveNumbers, e)) for factor, e in zip(function, stencil)])
    return sp.ln(laplaceTrafo)


# ------------------------------------------- Public Functions ---------------------------------------------------------


def cumulantAsFunctionOfRawMoments(index, momentsDict=None):
    return __cumulantRawMomentTransform(index, momentsDict, sp.log, 'm', False)


def rawMomentAsFunctionOfCumulants(index, cumulantsDict=None):
    return __cumulantRawMomentTransform(index, cumulantsDict, sp.exp, 'c', False)


def cumulantAsFunctionOfCentralMoments(index, momentsDict=None):
    return __cumulantRawMomentTransform(index, momentsDict, sp.log, 'm', True)


def centralMomentAsFunctionOfCumulants(index, cumulantsDict=None):
    return __cumulantRawMomentTransform(index, cumulantsDict, sp.exp, 'c', True)


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
def cumulantsFromPdfs(stencil, cumulantIndices=None, pdfSymbols=None, cumulantSymbols=None):
    """
    Creates equations to transform pdfs to cumulant space

    :param stencil:
    :param cumulantIndices: sequence of cumulant indices, could be tuples or polynomial representation
                            if left to default and a full stencil was passed,
                            the full set i.e. `momentsUpToComponentOrder(2)` is used
    :param pdfSymbols: symbolic values for pdf values, if not passed they default to :math:`f_0, f_1, ...`
    :param cumulantSymbols: symbolic values for cumulants (left hand sides of returned equations)
                            by default they are labeled :math:`c_{00}, c_{01}, c_{10}, ...`
    :return: sequence of equations for each cumulant one, on the right hand sides only pdfSymbols are used
    """
    dim = len(stencil[0])
    if cumulantIndices is None:
        cumulantIndices = list(momentsUpToComponentOrder(2, dim=dim))
    assert len(stencil) == len(cumulantIndices), "Stencil has to have same length as cumulantIndices sequence"
    cumulantSymbols = __getIndexedSymbols(cumulantSymbols, "c", cumulantIndices)
    pdfSymbols = __getIndexedSymbols(pdfSymbols, "f", range(len(stencil)))
    return [sp.Eq(cumulantSymbol, discreteCumulant(tuple(pdfSymbols), idx, stencil))
            for cumulantSymbol, idx in zip(cumulantSymbols, cumulantIndices)]


@diskcache
def cumulantsFromRawMoments(stencil, indices=None, momentSymbols=None, cumulantSymbols=None):
    """
    Creates equations to transform from raw moment representation to cumulants

    :param stencil:
    :param indices: indices of raw moments/ cumulant symbols, by default the full set is used
    :param momentSymbols: symbolic values for moments (symbols used for moments in equations)
    :param cumulantSymbols: symbolic values for cumulants (left hand sides of the equations)
    :return: equations to compute cumulants from raw moments
    """
    #TODO

@diskcache
def rawMomentsFromCumulants(stencil, cumulantSymbols=None, momentSymbols=None):
    #TODO
    pass
