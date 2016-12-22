"""

"""

import sympy as sp
import functools
from pystencils.sympyextensions import makeExponentialFuncArgumentSquares


@functools.lru_cache()
def momentGeneratingFunction(function, symbols, symbolsInResult):
    """
    Computes the moment generating function of a probability distribution. It is defined as:

    .. math ::
        F[f(\mathbf{x})](\mathbf{t}) = \int e^{<\mathbf{x}, \mathbf{t}>} f(x)\; dx

    :param function: sympy expression
    :param symbols: a sequence of symbols forming the vector x
    :param symbolsInResult: a sequence forming the vector t
    :return: transformation result F: an expression that depends now on symbolsInResult
             (symbols have been integrated out)

    .. note::
         This function uses sympys symbolic integration mechanism, which may not work or take a large
         amount of time for some functions.
         Therefore this routine does some transformations/simplifications on the function first, which are
         taylored to expressions of the form exp(polynomial) i.e. Maxwellian distributions, so that these kinds
         of functions can be integrated quickly.

    """
    assert len(symbols) == len(symbolsInResult)

    for t_i, v_i in zip(symbolsInResult, symbols):
        function *= sp.exp(t_i * v_i)

    # This is a custom transformation that speeds up the integrating process
    # of a MaxwellBoltzmann distribution
    # without this transformation the symbolic integration is sometimes not possible (e.g. in 2D without assumptions)
    # or is really slow
    # other functions should not be affected by this transformation
    # Without this transformation the following assumptions are required for the u and v variables of Maxwell Boltzmann
    #  2D: real=True ( without assumption it will not work)
    #  3D: no assumption ( with assumptions it will not work )
    function = makeExponentialFuncArgumentSquares(function, symbols)
    function = function.collect(symbols)

    bounds = [(s_i, -sp.oo, sp.oo) for s_i in symbols]
    result = sp.integrate(function, *bounds)

    return sp.simplify(result)


def cumulantGeneratingFunction(function, symbols, symbolsInResult):
    """
    Computes cumulant generating function, which is the logarithm of the moment generating function.
    For parameter description see :func:`momentGeneratingFunction`.
    """
    return sp.ln(momentGeneratingFunction(function, symbols, symbolsInResult))


def multiDifferentiation(generatingFunction, index, symbols):
    """
    Computes moment from moment-generating function or cumulant from cumulant-generating function,
    by differentiating the generating function, as specified by index and evaluating the derivative at symbols=0

    :param generatingFunction: function with is differentiated
    :param index: the i'th index specifies how often to differentiate w.r.t. to symbols[i]
    :param symbols: symbol to differentiate
    """
    assert len(index) == len(symbols), "Length of index and length of symbols has to match"

    diffArgs = []
    for order, t_i in zip(index, symbols):
        for i in range(order):
            diffArgs.append(t_i)

    if len(diffArgs) > 0:
        r = sp.diff(generatingFunction, *diffArgs)
    else:
        r = generatingFunction

    for t_i in symbols:
        r = r.subs(t_i, 0)

    return r


@functools.lru_cache(maxsize=512)
def __continuousMomentOrCumulant(function, moment, symbols, generatingFunction):

    dim = len(moment)
    t = tuple([sp.Symbol("tmpvar_%d" % i,) for i in range(dim)])  # not using sp.Dummy here - since it prohibits caching
    symbols = symbols[:dim]

    if type(moment) is tuple:
        return multiDifferentiation(generatingFunction(function, symbols, t), moment, t)
    else:
        result = 0
        for term, coeff in moment.as_coefficients_dict().items():
            exponents = tuple([term.as_coeff_exponent(v_i)[1] for v_i in symbols])
            cm = multiDifferentiation(generatingFunction(function, symbols, t), exponents, t)
            result += coeff * cm
        return result


def continuousMoment(function, moment, symbols):
    """
    Computes moment of given function

    :param function: function to compute moments of
    :param moment: tuple or polynomial describing the moment
    :param symbols: degrees of freedom of the function
    """
    return __continuousMomentOrCumulant(function, moment, symbols, momentGeneratingFunction)


def continuousCumulant(function, moment, symbols):
    """
    Computes cumulant of continuous function
    for parameter description see :func:`continuousMoment`
    """
    return __continuousMomentOrCumulant(function, moment, symbols, cumulantGeneratingFunction)

