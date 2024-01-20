"""
This module provides functions to compute cumulants of discrete functions.
Additionally functions are provided to compute cumulants from moments and vice versa
"""

import sympy as sp

from lbmpy.continuous_distribution_measures import multi_differentiation
from lbmpy.moments import moments_up_to_component_order
from pystencils.cache import memorycache
from pystencils.sympyextensions import fast_subs, scalar_product


def __get_indexed_symbols(passed_symbols, prefix, indices):
    """If passed symbols is not None, they are returned, if they are None
    indexed symbols of the form {prefix}_{index} are returned"""
    try:
        dim = len(indices[0])
    except TypeError:
        dim = 1

    if passed_symbols is not None:
        return passed_symbols
    else:
        format_string = "%s_" + "_".join(["%d"] * dim)
        tuple_indices = []
        for i in indices:
            tuple_indices.append(i if type(i) is tuple else (i,))
        return [sp.Symbol(format_string % ((prefix,) + i)) for i in tuple_indices]


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


def __cumulant_raw_moment_transform(index, dependent_var_dict, outer_function, default_prefix, centralized):
    """Function to express cumulants as function of moments and vice versa.

    Uses multivariate version of Faa di Bruno's formula.

    Args:
        index: tuple describing the index of the cumulant/moment to express as function of moments/cumulants
        dependent_var_dict: a dictionary from index tuple to moments/cumulants symbols, or None to use default symbols
        outer_function: logarithm to transform from moments->cumulants, exp for inverse direction
        default_prefix: if dependent_var_dict is None, this is used to construct symbols of the form prefix_i_j_k
        centralized: if True the first order moments/cumulants are set to zero
    """
    dim = len(index)
    subs_dict = {}

    def create_moment_symbol(idx):
        idx = tuple(idx)
        result_symbol = sp.Symbol(default_prefix + "_" + "_".join(["%d"] * len(idx)) % idx)
        if dependent_var_dict is not None and idx in dependent_var_dict:
            subs_dict[result_symbol] = dependent_var_dict[idx]
        return result_symbol

    zeroth_moment = create_moment_symbol((0,) * dim)

    def outer_function_derivative(n):
        x = zeroth_moment
        return sp.diff(outer_function(x), *tuple([x] * n))

    # index (2,1,0) means differentiate twice w.r.t to first variable, and once w.r.t to second variable
    # this is transformed here into representation [0,0,1] such that each entry is one diff operation
    partition_list = []
    for i, index_component in enumerate(index):
        for j in range(index_component):
            partition_list.append(i)

    if len(partition_list) == 0:  # special case for zero index
        return fast_subs(outer_function(zeroth_moment), subs_dict)

    # implementation of Faa di Bruno's formula:
    result = 0
    for partition in __partition(partition_list):
        factor = outer_function_derivative(len(partition))
        for elements in partition:
            moment_index = [0, ] * dim
            for i in elements:
                moment_index[i] += 1
            factor *= create_moment_symbol(moment_index)
        result += factor

    if centralized:
        for i in range(dim):
            index = [0] * dim
            index[i] = 1
            result = result.subs(create_moment_symbol(index), 0)

    return fast_subs(result, subs_dict)


@memorycache(maxsize=16)
def __get_discrete_cumulant_generating_function(func, stencil, wave_numbers):
    assert stencil.Q == len(func)

    laplace_transformation = sum([factor * sp.exp(scalar_product(wave_numbers, e)) for factor, e in zip(func, stencil)])
    return sp.ln(laplace_transformation)


# ------------------------------------------- Public Functions ---------------------------------------------------------


@memorycache(maxsize=64)
def discrete_cumulant(func, cumulant, stencil):
    """Computes cumulant of discrete function.

    Args:
        func: sequence of function components, has to have the same length as stencil
        cumulant: definition of cumulant, either as an index tuple, or as a polynomial
                  (similar to moment description)
        stencil: sequence of directions
    """
    assert stencil.Q == len(func)

    dim = len(stencil[0])
    wave_numbers = sp.symbols(f"Xi_:{dim}")

    generating_function = __get_discrete_cumulant_generating_function(func, stencil, wave_numbers)
    if type(cumulant) is tuple:
        return multi_differentiation(generating_function, cumulant, wave_numbers)
    else:
        from lbmpy.moments import MOMENT_SYMBOLS
        result = 0
        for term, coefficient in cumulant.as_coefficients_dict().items():
            exponents = tuple([term.as_coeff_exponent(v_i)[1] for v_i in MOMENT_SYMBOLS[:dim]])
            generating_function = __get_discrete_cumulant_generating_function(func, stencil, wave_numbers)
            cm = multi_differentiation(generating_function, exponents, wave_numbers)
            result += coefficient * cm
        return result


@memorycache(maxsize=8)
def cumulants_from_pdfs(stencil, cumulant_indices=None, pdf_symbols=None):
    """Transformation of pdfs (or other discrete function on a stencil) to cumulant space.

    Args:
        stencil: stencil object
        cumulant_indices: sequence of cumulant indices, could be tuples or polynomial representation
                          if left to default and a full stencil was passed,
                          the full set i.e. `moments_up_to_component_order` with 'order=2'  is used
        pdf_symbols: symbolic values for pdf values, if not passed they default to :math:`f_0, f_1, ...`

    Returns:
        dict mapping cumulant index to expression
    """
    dim = len(stencil[0])
    if cumulant_indices is None:
        cumulant_indices = moments_up_to_component_order(2, dim=dim)
    assert stencil.Q == len(cumulant_indices), "Stencil has to have same length as cumulant_indices sequence"
    if pdf_symbols is None:
        pdf_symbols = __get_indexed_symbols(pdf_symbols, "f", range(stencil.Q))
    return {idx: discrete_cumulant(tuple(pdf_symbols), idx, stencil) for idx in cumulant_indices}


def cumulant_as_function_of_raw_moments(index, moments_dict=None):
    """Returns an expression for the cumulant of given index as a function of raw moments.

    Args:
        index: a tuple of same length as spatial dimensions, specifying the cumulant
        moments_dict: a dictionary that maps moment indices to symbols/values. These values are used for
                     the moments in the returned expression. If this parameter is None, default symbols are used.
    """
    return __cumulant_raw_moment_transform(index, moments_dict, sp.log, 'm', False)


def raw_moment_as_function_of_cumulants(index, cumulants_dict=None):
    """Inverse transformation of :func:`cumulant_as_function_of_raw_moments`."""
    return __cumulant_raw_moment_transform(index, cumulants_dict, sp.exp, 'c', False)


def cumulant_as_function_of_central_moments(index, moments_dict=None):
    """Same as :func:`cumulant_as_function_of_raw_moments` but with central instead of raw moments."""
    return __cumulant_raw_moment_transform(index, moments_dict, sp.log, 'm', True)


def central_moment_as_function_of_cumulants(index, cumulants_dict=None):
    """Same as :func:`raw_moment_as_function_of_cumulants` but with central instead of raw moments."""
    return __cumulant_raw_moment_transform(index, cumulants_dict, sp.exp, 'c', True)
