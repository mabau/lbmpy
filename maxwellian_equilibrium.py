"""
This module contains the continuous Maxwell-Boltzmann equilibrium and its discrete polynomial approximation, often
used to formulate lattice-Boltzmann methods for hydrodynamics.
Additionally functions are provided to compute moments and cumulants of these distributions.
"""

import sympy as sp
import functools
from lbmpy.diskcache import diskcache


@functools.lru_cache()
def computeWeights1D(stencilWidth, c_s_sq):
    """
    Computes lattice weights for a symmetric 1D stencil with given stencil width.

    :param stencilWidth: stencil width, i.e. the maximum absolute value of direction component
                         e.g. stencilWidth=2 means direction -2, -1, 0, 1, 2
    :param c_s_sq: speed of sound squared
    :returns: dict mapping from integer offset to weight

    >>> computeWeights1D(1, sp.Rational(1,3))
    {-1: 1/6, 0: 2/3, 1: 1/6}
    """
    from lbmpy.moments import momentMatrix
    # Create a 1D stencil with the given width
    stencil = tuple([(i,) for i in range(-stencilWidth, stencilWidth+1)])
    moments = tuple([(i,) for i in range(2*stencilWidth+1)])
    contMoments = getMomentsOfContinuousMaxwellianEquilibrium(moments, rho=1, u=(0,), c_s_sq=c_s_sq, dim=1)
    M = momentMatrix(moments, stencil)
    weights = M.inv() * sp.Matrix(contMoments)
    return {i: w_i for i, w_i in zip(range(-stencilWidth, stencilWidth + 1), weights)}


@functools.lru_cache()
def getWeights(stencil, c_s_sq):
    """
    Computes the weights for a stencil

    The weight of a direction is determined by multiplying 1D weights i.e. there is a 1D weight associated
    with entries -1, 0 and 1 (for one neighborhood stencils). These 1D weights are determined by the
    continuous Maxwellian distribution.
    The weight of the higher dimensional direction is
    the product of the 1D weights, dependent on its entries. For example (1,-1,0) would be
    :math:`w_{1} w_{-1} w_{0}`.
    This product approach is described in detail in :cite:`karlin2015entropic`.

    :param stencil: tuple of directions
    :param c_s_sq:  speed of sound squared
    :returns: list of weights, one for each direction
    """
    maxNeighborhood = 0
    for d in stencil:
        for e in d:
            if abs(e) > maxNeighborhood:
                maxNeighborhood = abs(e)
    elementaryWeights = computeWeights1D(maxNeighborhood, c_s_sq)

    result = []
    for directionTuple in stencil:
        weight = 1
        for e in directionTuple:
            weight *= elementaryWeights[e]
        result.append(weight)
    return result


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
    weights = getWeights(stencil, c_s_sq)
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

