from sympy import Rational as R
from sympy import simplify
import sympy as sp
from lbmpy.util import getSymbolicDensity, getSymbolicVelocityVector, getSymbolicSoundSpeed
from lbmpy.moments import momentMatrix, continuousMoment, MOMENT_SYMBOLS
from lbmpy.transformations import removeHigherOrderTerms

from joblib import Memory

memory = Memory(cachedir="/tmp/lbmpy", verbose=False)


def getWeights(stencil):
    Q = len(stencil)

    def weightForDirection(direction):
        absSum = sum([abs(d) for d in direction])
        return getWeights.weights[Q][absSum]
    return [weightForDirection(d) for d in stencil]
getWeights.weights = {
    9: {
        0: R(4, 9),
        1: R(1, 9),
        2: R(1, 36),
    },
    15: {
        0: R(2, 9),
        1: R(1, 9),
        3: R(1, 72),
    },
    19: {
        0: R(1, 3),
        1: R(1, 18),
        2: R(1, 36),
    },
    27: {
        0: R(8, 27),
        1: R(2, 27),
        2: R(1, 54),
        3: R(1, 216),
    }
}


@memory.cache
def standardDiscreteEquilibrium(stencil, rho=None, u=None, order=2, c_s_sq=None, compressible=True):
    """
    Returns the common quadratic LBM equilibrium as a list of sympy expressions
    :param stencil: one of the supported stencils ( D2Q9, D3Q19, ... )
    :param rho: sympy symbol for the density - defaults to symbols('rho')
    :param u: list with the length of stencil dimensionality with symbols for velocity,
    :param order: highest order of velocity terms (for hydrodynamics order 2 is sufficient)
    :param c_s_sq: square of speed of sound - if not specified it is chosen as 1/3
    """
    e = stencil
    w = getWeights(stencil)
    assert len(e) == len(w)

    D = len(e[0])

    if not rho:
        rho = getSymbolicDensity()
    if not u:
        u = getSymbolicVelocityVector(D, "u")

    if not c_s_sq:
        c_s_sq = getSymbolicSoundSpeed()**2

    rhoOutside = rho if compressible else R(1, 1)
    rhoInside = rho if not compressible else R(1, 1)

    res = []
    for w_q, e_q in zip(w, e):
        eTimesU = 0
        for c_q_alpha, u_alpha in zip(e_q, u):
            eTimesU += c_q_alpha * u_alpha

        fq = rhoInside + eTimesU / c_s_sq

        if order <= 1:
            res.append(fq * rhoOutside * w_q)
            continue

        uTimesU = 0
        for u_alpha in u:
            uTimesU += u_alpha * u_alpha
        fq += R(1, 2) / c_s_sq**2 * eTimesU ** 2 - R(1, 2) / c_s_sq * uTimesU

        if order <= 2:
            res.append(fq * rhoOutside * w_q)
            continue

        fq += R(1, 6) / c_s_sq**3 * eTimesU**3 - R(1, 2) / c_s_sq**2 * uTimesU * eTimesU

        res.append(simplify(fq * rhoOutside * w_q))

    return sp.Matrix(len(stencil), 1, res)


@memory.cache
def maxwellBoltzmannEquilibrium(dimension=3, rho=None, u=None, v=None, c_s_sq=None):
    """
    Returns sympy expression of Maxwell Boltzmann distribution
    :param dimension: number of space dimensions
    :param rho: sympy symbol for the density - defaults to symbols('rho')
                if custom symbol is used make sure to mark is as positive=True
    :param u: list with the length of spatial dimensions containing symbols for macroscopic velocity u
    :param u: list with the length of spatial dimensions containing symbols for peculiar velocity v
    :param c_s_sq: symbol for speed of sound squared, defaults to symbol c_s**2
    :return:
    """
    D = dimension

    if not rho:
        rho = getSymbolicDensity()
    if not u:
        u = getSymbolicVelocityVector(D, "u")
    if not v:
        v = getSymbolicVelocityVector(D, "v")
    if not c_s_sq:
        c_s = getSymbolicSoundSpeed()
        c_s_sq = c_s**2

    velTerm = sum([(v_i - u_i) ** 2 for v_i, u_i in zip(v, u)])
    return rho / (2 * sp.pi * c_s_sq) ** (R(D, 2)) * sp.exp(- velTerm / (2 * c_s_sq))


@memory.cache
def getEquilibriumMoments(listOfMoments, continuousEquilibrium, u, v, order=None):
    result = []
    for moment in listOfMoments:
        contMom = 0
        for term, coeff in moment.as_coefficients_dict().items():
            exponents = tuple([term.as_coeff_exponent(v_i)[1] for v_i in v])
            contMom += coeff * continuousMoment(continuousEquilibrium, exponents, symbols=v)
        if order:
            contMom = removeHigherOrderTerms(contMom, order, u)
        result.append(contMom)
    return result


@memory.cache
def getMaxwellBoltzmannEquilibriumMoments(moments, order=None, velSymbols=MOMENT_SYMBOLS):
    dim = len(velSymbols)
    rho = getSymbolicDensity()
    u = getSymbolicVelocityVector(dim, "u")
    mb = maxwellBoltzmannEquilibrium(dim, rho, u, velSymbols, c_s_sq=sp.Rational(1, 3))
    return getEquilibriumMoments(moments, mb, u, velSymbols, order)


@memory.cache
def generateEquilibriumByMatchingMoments(stencil, moments, order=None, velSymbols=MOMENT_SYMBOLS):
    Q = len(stencil)
    assert len(moments) == Q, "Moment count(%d) does not match stencil size(%d)" % (len(moments), Q)
    continuousMomentsVector = sp.Matrix(Q, 1, getMaxwellBoltzmannEquilibriumMoments(moments, order, velSymbols))
    M = momentMatrix(moments, stencil)
    assert M.rank() == Q, "Rank of moment matrix (%d) does not match stencil size (%d)" % (M.rank(), Q)
    return M.inv() * continuousMomentsVector

