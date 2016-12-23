"""
This module contains the continuous Maxwell-Boltzmann equilibrium and its discrete polynomial approximation, often
used to formulate lattice-Boltzmann methods for hydrodynamics.
Additionally functions are provided to compute moments and cumulants of these distributions.
"""

import sympy as sp
from sympy import Rational as R
from lbmpy.diskcache import diskcache


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


@diskcache
def discreteMaxwellianEquilibrium(stencil, rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")), order=2,
                                  c_s_sq=sp.Symbol("c_s") ** 2, compressible=True):
    """
    Returns the common discrete LBM equilibrium as a list of sympy expressions
    :param stencil: tuple of directions
    :param rho: sympy symbol for the density
    :param u: symbols for macroscopic velocity, only the first `dim` entries are used
    :param order: highest order of velocity terms (for hydrodynamics order 2 is sufficient)
    :param c_s_sq: square of speed of sound
    :param compressible: compressibility
    """
    weights = getWeights(stencil)
    assert len(stencil) == len(weights)

    dim = len(stencil[0])
    u = u[:dim]

    rhoOutside = rho if compressible else sp.Rational(1, 1)
    rhoInside = rho if not compressible else sp.Rational(1, 1)

    res = []
    for w_q, e_q in zip(weights, stencil):
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
        fq += sp.Rational(1, 2) / c_s_sq**2 * eTimesU ** 2 - sp.Rational(1, 2) / c_s_sq * uTimesU

        if order <= 2:
            res.append(fq * rhoOutside * w_q)
            continue

        fq += sp.Rational(1, 6) / c_s_sq**3 * eTimesU**3 - sp.Rational(1, 2) / c_s_sq**2 * uTimesU * eTimesU

        res.append(sp.expand(fq * rhoOutside * w_q))

    return tuple(res)


@diskcache
def generateEquilibriumByMatchingMoments(stencil, moments, rho=sp.Symbol("rho"),
                                         u=tuple(sp.symbols("u_0 u_1 u_2")), c_s_sq=sp.Symbol("c_s") ** 2, order=None):
    """
    Computes discrete equilibrium, by setting the discrete moments to values taken from the continuous Maxwellian.
    The number of moments has to match the number of directions in the stencil. For documentation of other parameters
    see :func:`getMomentsOfContinuousMaxwellianEquilibrium`
    """
    from lbmpy.moments import momentMatrix
    dim = len(stencil[0])
    Q = len(stencil)
    assert len(moments) == Q, "Moment count(%d) does not match stencil size(%d)" % (len(moments), Q)
    continuousMomentsVector = getMomentsOfContinuousMaxwellianEquilibrium(moments, rho, u, c_s_sq, dim, order)
    continuousMomentsVector = sp.Matrix(continuousMomentsVector)
    M = momentMatrix(moments, stencil)
    assert M.rank() == Q, "Rank of moment matrix (%d) does not match stencil size (%d)" % (M.rank(), Q)
    return M.inv() * continuousMomentsVector


@diskcache
def continuousMaxwellianEquilibrium(dim=3, rho=sp.Symbol("rho"),
                                    u=tuple(sp.symbols("u_0 u_1 u_2")),
                                    v=tuple(sp.symbols("v_0 v_1 v_2")),
                                    c_s_sq=sp.Symbol("c_s") ** 2):
    """
    Returns sympy expression of Maxwell Boltzmann distribution
    :param dim: number of space dimensions
    :param rho: sympy symbol for the density
    :param u: symbols for macroscopic velocity (expected value for velocity)
    :param v: symbols for particle velocity
    :param c_s_sq: symbol for speed of sound squared, defaults to symbol c_s**2
    """
    u = u[:dim]
    v = v[:dim]

    velTerm = sum([(v_i - u_i) ** 2 for v_i, u_i in zip(v, u)])
    return rho / (2 * sp.pi * c_s_sq) ** (sp.Rational(dim, 2)) * sp.exp(- velTerm / (2 * c_s_sq))


# -------------------------------- Equilibrium moments/cumulants  ------------------------------------------------------


@diskcache
def getMomentsOfContinuousMaxwellianEquilibrium(moments, rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")),
                                                c_s_sq=sp.Symbol("c_s") ** 2,
                                                dim=3, order=None):
    """
    Computes moments of the continuous Maxwell Boltzmann equilibrium distribution
    :param moments: moments to compute, either in polynomial or exponent-tuple form
    :param rho: symbol or value for the density
    :param u: symbols or values for the macroscopic velocity
    :param c_s_sq: symbol for speed of sound squared, defaults to symbol c_s**2
    :param dim: dimension (2 or 3)
    :param order: if this parameter is not None, terms that have a higher polynomial order in the macroscopic velocity
                  are removed

    >>> getMomentsOfContinuousMaxwellianEquilibrium( ( (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0) ) )
    [rho, rho*u_0, rho*u_1, rho*u_2, rho*(c_s**2 + u_0**2)]
    """
    from pystencils.sympyextensions import removeHigherOrderTerms
    from lbmpy.moments import MOMENT_SYMBOLS
    from lbmpy.continuous_distribution_measures import continuousMoment

    # trick to speed up sympy integration (otherwise it takes multiple minutes, or aborts):
    # use a positive, real symbol to represent c_s_sq -> then replace this symbol afterwards with the real c_s_sq
    c_s_sq_helper = sp.Symbol("csqHelper", positive=True, real=True)
    mb = continuousMaxwellianEquilibrium(dim, rho, u, MOMENT_SYMBOLS[:dim], c_s_sq_helper)
    result = [continuousMoment(mb, moment, MOMENT_SYMBOLS[:dim]).subs(c_s_sq_helper, c_s_sq) for moment in moments]
    if order is not None:
        result = [removeHigherOrderTerms(r, order, u) for r in result]

    return result


@diskcache
def getMomentsOfDiscreteMaxwellianEquilibrium(stencil,
                                              moments, rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")),
                                              c_s_sq=sp.Symbol("c_s") ** 2, order=None, compressible=True):
    """
    Compute moments of discrete maxwellian equilibrium
    :param stencil: stencil is required to compute moments of discrete function
    :param moments: moments in polynomial or exponent-tuple form
    :param rho: symbol or value for the density
    :param u: symbols or values for the macroscopic velocity
    :param c_s_sq: symbol for speed of sound squared, defaults to symbol c_s**2
    :param order: highest order of u terms
    :param compressible: compressible or incompressible form
    """
    from lbmpy.moments import discreteMoment
    if order is None:
        order = 4
    mb = discreteMaxwellianEquilibrium(stencil, rho, u, order, c_s_sq, compressible)
    return tuple([discreteMoment(mb, moment, stencil).expand() for moment in moments])


# -------------------------------- Equilibrium moments -----------------------------------------------------------------


def getCumulantsOfContinuousMaxwellianEquilibrium(cumulants, rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")),
                                                  c_s_sq=sp.Symbol("c_s") ** 2, dim=3):
    from lbmpy.moments import MOMENT_SYMBOLS
    from lbmpy.continuous_distribution_measures import continuousCumulant

    mb = continuousMaxwellianEquilibrium(dim, rho, u, MOMENT_SYMBOLS[:dim], c_s_sq)
    result = [continuousCumulant(mb, cumulant, MOMENT_SYMBOLS[:dim]) for cumulant in cumulants]

    return result

