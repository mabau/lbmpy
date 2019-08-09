"""
This module contains the continuous Maxwell-Boltzmann equilibrium and its discrete polynomial approximation, often
used to formulate lattice-Boltzmann methods for hydrodynamics.
Additionally functions are provided to compute moments and cumulants of these distributions.
"""

import warnings

import sympy as sp
from sympy import Rational as R

from pystencils.cache import disk_cache


def get_weights(stencil, c_s_sq):
    q = len(stencil)

    if c_s_sq != sp.Rational(1, 3) and c_s_sq != sp.Symbol("c_s") ** 2:
        warnings.warn("Weights of discrete equilibrium are only valid if c_s^2 = 1/3")

    def weight_for_direction(direction):
        abs_sum = sum([abs(d) for d in direction])
        return get_weights.weights[q][abs_sum]
    return [weight_for_direction(d) for d in stencil]


get_weights.weights = {
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


@disk_cache
def discrete_maxwellian_equilibrium(stencil, rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")), order=2,
                                    c_s_sq=sp.Symbol("c_s") ** 2, compressible=True):
    """
    Returns the common discrete LBM equilibrium as a list of sympy expressions

    Args:
        stencil: tuple of directions
        rho: sympy symbol for the density
        u: symbols for macroscopic velocity, only the first 'dim' entries are used
        order: highest order of velocity terms (for hydrodynamics order 2 is sufficient)
        c_s_sq: square of speed of sound
        compressible: compressibility
    """
    weights = get_weights(stencil, c_s_sq)
    assert len(stencil) == len(weights)

    dim = len(stencil[0])
    u = u[:dim]

    rho_outside = rho if compressible else sp.Rational(1, 1)
    rho_inside = rho if not compressible else sp.Rational(1, 1)

    res = []
    for w_q, e_q in zip(weights, stencil):
        e_times_u = 0
        for c_q_alpha, u_alpha in zip(e_q, u):
            e_times_u += c_q_alpha * u_alpha

        fq = rho_inside + e_times_u / c_s_sq

        if order <= 1:
            res.append(fq * rho_outside * w_q)
            continue

        u_times_u = 0
        for u_alpha in u:
            u_times_u += u_alpha * u_alpha
        fq += sp.Rational(1, 2) / c_s_sq**2 * e_times_u ** 2 - sp.Rational(1, 2) / c_s_sq * u_times_u

        if order <= 2:
            res.append(fq * rho_outside * w_q)
            continue

        fq += sp.Rational(1, 6) / c_s_sq**3 * e_times_u**3 - sp.Rational(1, 2) / c_s_sq**2 * u_times_u * e_times_u

        res.append(sp.expand(fq * rho_outside * w_q))

    return tuple(res)


@disk_cache
def generate_equilibrium_by_matching_moments(stencil, moments, rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")),
                                             c_s_sq=sp.Symbol("c_s") ** 2, order=None):
    """
    Computes discrete equilibrium, by setting the discrete moments to values taken from the continuous Maxwellian.
    The number of moments has to match the number of directions in the stencil. For documentation of other parameters
    see :func:`get_moments_of_continuous_maxwellian_equilibrium`
    """
    from lbmpy.moments import moment_matrix
    dim = len(stencil[0])
    Q = len(stencil)
    assert len(moments) == Q, "Moment count(%d) does not match stencil size(%d)" % (len(moments), Q)
    continuous_moments_vector = get_moments_of_continuous_maxwellian_equilibrium(moments, dim, rho, u, c_s_sq, order)
    continuous_moments_vector = sp.Matrix(continuous_moments_vector)
    M = moment_matrix(moments, stencil)
    assert M.rank() == Q, "Rank of moment matrix (%d) does not match stencil size (%d)" % (M.rank(), Q)
    return M.inv() * continuous_moments_vector


@disk_cache
def continuous_maxwellian_equilibrium(dim=3, rho=sp.Symbol("rho"),
                                      u=tuple(sp.symbols("u_0 u_1 u_2")),
                                      v=tuple(sp.symbols("v_0 v_1 v_2")),
                                      c_s_sq=sp.Symbol("c_s") ** 2):
    """
    Returns sympy expression of Maxwell Boltzmann distribution

    Args:
        dim: number of space dimensions
        rho: sympy symbol for the density
        u: symbols for macroscopic velocity (expected value for velocity)
        v: symbols for particle velocity
        c_s_sq: symbol for speed of sound squared, defaults to symbol c_s**2
    """
    u = u[:dim]
    v = v[:dim]

    vel_term = sum([(v_i - u_i) ** 2 for v_i, u_i in zip(v, u)])
    return rho / (2 * sp.pi * c_s_sq) ** (sp.Rational(dim, 2)) * sp.exp(- vel_term / (2 * c_s_sq))


# -------------------------------- Equilibrium moments/cumulants  ------------------------------------------------------


@disk_cache
def get_moments_of_continuous_maxwellian_equilibrium(moments, dim, rho=sp.Symbol("rho"),
                                                     u=tuple(sp.symbols("u_0 u_1 u_2")),
                                                     c_s_sq=sp.Symbol("c_s") ** 2, order=None):
    """
    Computes moments of the continuous Maxwell Boltzmann equilibrium distribution

    Args:
        moments: moments to compute, either in polynomial or exponent-tuple form
        dim: dimension (2 or 3)
        rho: symbol or value for the density
        u: symbols or values for the macroscopic velocity
        c_s_sq: symbol for speed of sound squared, defaults to symbol c_s**2
        order: if this parameter is not None, terms that have a higher polynomial order in the macroscopic velocity
               are removed

    >>> get_moments_of_continuous_maxwellian_equilibrium( ( (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0) ), dim=3 )
    [rho, rho*u_0, rho*u_1, rho*u_2, rho*(c_s**2 + u_0**2)]
    """
    from pystencils.sympyextensions import remove_higher_order_terms
    from lbmpy.moments import MOMENT_SYMBOLS
    from lbmpy.continuous_distribution_measures import continuous_moment

    # trick to speed up sympy integration (otherwise it takes multiple minutes, or aborts):
    # use a positive, real symbol to represent c_s_sq -> then replace this symbol afterwards with the real c_s_sq
    c_s_sq_helper = sp.Symbol("csq_helper", positive=True, real=True)
    mb = continuous_maxwellian_equilibrium(dim, rho, u, MOMENT_SYMBOLS[:dim], c_s_sq_helper)
    result = [continuous_moment(mb, moment, MOMENT_SYMBOLS[:dim]).subs(c_s_sq_helper, c_s_sq) for moment in moments]
    if order is not None:
        result = [remove_higher_order_terms(r, order=order, symbols=u) for r in result]

    return result


@disk_cache
def get_moments_of_discrete_maxwellian_equilibrium(stencil, moments,
                                                   rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")),
                                                   c_s_sq=sp.Symbol("c_s") ** 2, order=None, compressible=True):
    """Compute moments of discrete maxwellian equilibrium.

    Args:
        stencil: stencil is required to compute moments of discrete function
        moments: moments in polynomial or exponent-tuple form
        rho: symbol or value for the density
        u: symbols or values for the macroscopic velocity
        c_s_sq: symbol for speed of sound squared, defaults to symbol c_s**2
        order: highest order of u terms
        compressible: compressible or incompressible form
    """
    from lbmpy.moments import discrete_moment
    if order is None:
        order = 4
    mb = discrete_maxwellian_equilibrium(stencil, rho, u, order, c_s_sq, compressible)
    return tuple([discrete_moment(mb, moment, stencil).expand() for moment in moments])


def compressible_to_incompressible_moment_value(term, rho, u):
    """Compressible to incompressible equilibrium moments

    Transforms so-called compressible equilibrium moments (as obtained from the continuous Maxwellian) by
    removing the density factor in all monomials where velocity components are multiplied to the density.

    Examples:
        >>> rho, *u = sp.symbols("rho u_:2")
        >>> compressible_to_incompressible_moment_value(rho  + rho * u[0] + rho * u[0]*u[1], rho, u)
        rho + u_0*u_1 + u_0

    Args:
        term: compressible equilibrium value
        rho: symbol for density
        u: symbol for velocity

    Returns:
        incompressible equilibrium value
    """
    term = sp.sympify(term)
    term = term.expand()
    if term.func != sp.Add:
        args = [term, ]
    else:
        args = term.args

    res = 0
    for t in args:
        contained_symbols = t.atoms(sp.Symbol)
        if rho in contained_symbols and len(contained_symbols.intersection(set(u))) > 0:
            res += t / rho
        else:
            res += t
    return res


# -------------------------------- Equilibrium moments -----------------------------------------------------------------


def get_cumulants_of_continuous_maxwellian_equilibrium(cumulants, dim, rho=sp.Symbol("rho"),
                                                       u=tuple(sp.symbols("u_0 u_1 u_2")), c_s_sq=sp.Symbol("c_s") ** 2,
                                                       order=None):
    from lbmpy.moments import MOMENT_SYMBOLS
    from lbmpy.continuous_distribution_measures import continuous_cumulant
    from pystencils.sympyextensions import remove_higher_order_terms

    # trick to speed up sympy integration (otherwise it takes multiple minutes, or aborts):
    # use a positive, real symbol to represent c_s_sq -> then replace this symbol afterwards with the real c_s_sq
    c_s_sq_helper = sp.Symbol("csq_helper", positive=True, real=True)
    mb = continuous_maxwellian_equilibrium(dim, rho, u, MOMENT_SYMBOLS[:dim], c_s_sq_helper)
    result = [continuous_cumulant(mb, cumulant, MOMENT_SYMBOLS[:dim]).subs(c_s_sq_helper, c_s_sq)
              for cumulant in cumulants]
    if order is not None:
        result = [remove_higher_order_terms(r, order=order, symbols=u) for r in result]

    return result


@disk_cache
def get_cumulants_of_discrete_maxwellian_equilibrium(stencil, cumulants,
                                                     rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")),
                                                     c_s_sq=sp.Symbol("c_s") ** 2, order=None, compressible=True):
    from lbmpy.cumulants import discrete_cumulant
    if order is None:
        order = 4
    mb = discrete_maxwellian_equilibrium(stencil, rho, u, order, c_s_sq, compressible)
    return tuple([discrete_cumulant(mb, cumulant, stencil).expand() for cumulant in cumulants])
