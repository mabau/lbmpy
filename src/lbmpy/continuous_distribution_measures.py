"""

"""

import sympy as sp

from lbmpy.moments import polynomial_to_exponent_representation
from pystencils.cache import disk_cache, memorycache
from pystencils.sympyextensions import complete_the_squares_in_exp, scalar_product


@memorycache()
def moment_generating_function(generating_function, symbols, symbols_in_result, velocity=None):
    r"""
    Computes the moment generating function of a probability distribution. It is defined as:

    .. math ::
        F[f(\mathbf{x})](t) = \int e^{<\mathbf{x}, t>} f(\mathbf{x})\; dx

    Args:
        generating_function: sympy expression
        symbols: a sequence of symbols forming the vector :math:`\mathbf{x}`
        symbols_in_result: a sequence forming the vector t
        velocity: if the generating function generates central moments, the velocity needs to be substracted. Thus the
                  velocity symbols need to be passed. All generating functions need to have the same parameters.

    Returns:
        transformation result F: an expression that depends now on symbols_in_result
        (symbols have been integrated out)

    Note:
         This function uses sympys symbolic integration mechanism, which may not work or take a large
         amount of time for some functions.
         Therefore this routine does some transformations/simplifications on the function first, which are
         taylored to expressions of the form exp(polynomial) i.e. Maxwellian distributions, so that these kinds
         of functions can be integrated quickly.

    """
    assert len(symbols) == len(symbols_in_result)
    for t_i, v_i in zip(symbols_in_result, symbols):
        generating_function *= sp.exp(t_i * v_i)

    # This is a custom transformation that speeds up the integrating process
    # of a MaxwellBoltzmann distribution
    # without this transformation the symbolic integration is sometimes not possible (e.g. in 2D without assumptions)
    # or is really slow
    # other functions should not be affected by this transformation
    # Without this transformation the following assumptions are required for the u and v variables of Maxwell Boltzmann
    #  2D: real=True ( without assumption it will not work)
    #  3D: no assumption ( with assumptions it will not work )
    generating_function = complete_the_squares_in_exp(generating_function.simplify(), symbols)
    generating_function = generating_function.collect(symbols)

    bounds = [(s_i, -sp.oo, sp.oo) for s_i in symbols]
    result = sp.integrate(generating_function, *bounds)

    return sp.simplify(result)


def central_moment_generating_function(func, symbols, symbols_in_result, velocity=sp.symbols("u_:3")):
    r"""
    Computes central moment generating func, which is defined as:

    .. math ::
        K( \mathbf{\Xi} ) = \exp ( - \mathbf{\Xi} \cdot \mathbf{u} ) M( \mathbf{\Xi} ).

    For parameter description see :func:`moment_generating_function`.
    """
    argument = - scalar_product(symbols_in_result, velocity)

    return sp.exp(argument) * moment_generating_function(func, symbols, symbols_in_result)


def cumulant_generating_function(func, symbols, symbols_in_result, velocity=None):
    r"""
    Computes cumulant generating func, which is the logarithm of the moment generating func:

    .. math ::
        C(\mathbf{\Xi}) = \log M(\mathbf{\Xi})

    For parameter description see :func:`moment_generating_function`.
    """
    return sp.ln(moment_generating_function(func, symbols, symbols_in_result))


@disk_cache
def multi_differentiation(generating_function, index, symbols):
    """
    Computes moment from moment-generating function or cumulant from cumulant-generating function,
    by differentiating the generating function, as specified by index and evaluating the derivative at symbols=0

    Args:
        generating_function: function with is differentiated
        index: the i'th index specifies how often to differentiate w.r.t. to symbols[i]
        symbols: symbol to differentiate
    """
    assert len(index) == len(symbols), "Length of index and length of symbols has to match"

    diff_args = []
    for order, t_i in zip(index, symbols):
        for i in range(order):
            diff_args.append(t_i)

    if len(diff_args) > 0:
        r = sp.diff(generating_function, *diff_args)
    else:
        r = generating_function

    for t_i in symbols:
        r = r.subs(t_i, 0)

    return r


@memorycache(maxsize=512)
def __continuous_moment_or_cumulant(func, moment, symbols, generating_function, velocity=sp.symbols("u_:3")):
    if type(moment) is tuple and not symbols:
        symbols = sp.symbols("xvar yvar zvar")

    dim = len(moment) if type(moment) is tuple else len(symbols)

    # not using sp.Dummy here - since it prohibits caching
    t = sp.symbols(f"tmpvar_:{dim}")
    symbols = symbols[:dim]
    generating_function = generating_function(func, symbols, t, velocity=velocity)

    if type(moment) is tuple:
        return multi_differentiation(generating_function, moment, t)
    else:
        assert symbols is not None, "When passing a polynomial as moment, also the moment symbols have to be passed"
        moment = sp.sympify(moment)

        result = 0
        for coefficient, exponents in polynomial_to_exponent_representation(moment, dim=dim):
            result += coefficient * multi_differentiation(generating_function, exponents, t)

        return result


def continuous_moment(func, moment, symbols=None):
    """Computes moment of given function.

    Args:
        func: function to compute moments of
        moment: tuple or polynomial describing the moment
        symbols: if moment is given as polynomial, pass the moment symbols, i.e. the dof of the polynomial
    """
    return __continuous_moment_or_cumulant(func, moment, symbols, moment_generating_function)


def continuous_central_moment(func, moment, symbols=None, velocity=sp.symbols("u_:3")):
    """Computes central moment of given function.

    Args:
        func: function to compute moments of
        moment: tuple or polynomial describing the moment
        symbols: if moment is given as polynomial, pass the moment symbols, i.e. the dof of the polynomial
    """
    return __continuous_moment_or_cumulant(func, moment, symbols, central_moment_generating_function,
                                           velocity=velocity)


def continuous_cumulant(func, moment, symbols=None):
    """Computes cumulant of continuous function.

    for parameter description see :func:`continuous_moment`
    """
    return __continuous_moment_or_cumulant(func, moment, symbols, cumulant_generating_function)
