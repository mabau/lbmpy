import functools
import sympy as sp

from lbmpy.diskcache import diskcache
from lbmpy.moments import continuousCumulantOrMoment, momentsUpToComponentOrder, discreteMoment
from lbmpy.transformations import replaceAdditive
from lbmpy.util import scalarProduct


def getDefaultIndexedSymbols(passedSymbols, prefix, indices):
    try:
        dim = len(indices[0])
    except TypeError:
        dim = 1

    if passedSymbols is not None:
        return passedSymbols
    else:
        formatString = "%s_" + "_".join(["%d"]*dim)
        return [sp.Symbol(formatString % ((prefix,) + i)) for i in indices]


@functools.lru_cache(maxsize=512)
def continuousCumulant(function, indexTuple, symbols=None):
    """
    Computes cumulant of given function ( for parameters see function continuousMoment )
    """
    return continuousCumulantOrMoment(function, indexTuple, symbols, True)


@functools.lru_cache(maxsize=16)
def getDiscreteCumulantGeneratingFunction(function, stencil, waveNumbers):
    assert len(stencil) == len(function)
    laplaceTrafo = sum([factor * sp.exp(scalarProduct(waveNumbers, e)) for factor, e in zip(function, stencil)])
    return sp.ln(laplaceTrafo)


@functools.lru_cache(maxsize=64)
def discreteCumulant(function, indexTuple, stencil):
    assert len(stencil) == len(function)

    dim = len(stencil[0])
    assert len(indexTuple) == dim
    waveNumbers = tuple([sp.Symbol("Xi_%d" % (i,)) for i in range(dim)])

    res = getDiscreteCumulantGeneratingFunction(function, stencil, waveNumbers)
    for m, waveNumber in zip(indexTuple, waveNumbers):
        for i in range(m):
            res = sp.diff(res, waveNumber)
    return res.subs({waveNumber: 0 for waveNumber in waveNumbers})


@functools.lru_cache(maxsize=8)
def cumulantsFromPdfs(stencil, pdfSymbols=None, cumulantSymbols=None):
    dim = len(stencil[0])
    indices = list(momentsUpToComponentOrder(2, dim=dim))
    cumulantSymbols = getDefaultIndexedSymbols(cumulantSymbols, "c", indices)
    pdfSymbols = getDefaultIndexedSymbols(pdfSymbols, "f", range(len(stencil)))
    return [sp.Eq(cumulantSymbol, discreteCumulant(tuple(pdfSymbols), idx, stencil))
            for cumulantSymbol, idx in zip(cumulantSymbols, indices)]


@diskcache
def cumulantsFromRawMoments(stencil, momentSymbols=None, cumulantSymbols=None):
    dim = len(stencil[0])
    indices = list(momentsUpToComponentOrder(2, dim=dim))
    momentSymbols = getDefaultIndexedSymbols(momentSymbols, "m", indices)
    cumulantSymbols = getDefaultIndexedSymbols(cumulantSymbols, "c", indices)

    # here cumulantsFromPdfs is used, then the moments are replaced

    pdfSymbols = tuple([sp.Symbol("f_%d" % (i,)) for i in range(len(stencil))])
    replacements = {momentSymbol: discreteMoment(pdfSymbols, idx, stencil)
                    for momentSymbol, idx in zip(momentSymbols, indices)}
    cumulantEquations = cumulantsFromPdfs(stencil, tuple(pdfSymbols), tuple(cumulantSymbols))
    result = []
    for cumulantSymbol, cumulantEq in zip(cumulantSymbols, cumulantEquations):
        c = cumulantEq.rhs
        for symbol, replacement in replacements.items():
            c = replaceAdditive(c, symbol, replacement, requiredMatchOriginal=1.0)
        result.append(sp.Eq(cumulantSymbol, c))
    return result


@diskcache
def rawMomentsFromCumulants(stencil, cumulantSymbols=None, momentSymbols=None):
    dim = len(stencil[0])
    indices = list(momentsUpToComponentOrder(2, dim=dim))
    momentSymbols = getDefaultIndexedSymbols(momentSymbols, "m", indices)
    cumulantSymbols = getDefaultIndexedSymbols(cumulantSymbols, "c", indices)
    forwardEqs = cumulantsFromRawMoments(stencil, tuple(momentSymbols), tuple(cumulantSymbols))
    solveResult = sp.solve(forwardEqs, momentSymbols)
    assert len(solveResult) == 1, "Could not invert the forward equations - manual implementation required?"
    return [sp.Eq(momentSymbol, r) for momentSymbol, r in zip(momentSymbols, solveResult[0])]
