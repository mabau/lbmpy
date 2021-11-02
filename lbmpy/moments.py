# -*- coding: utf-8 -*-
r"""
Module Overview
~~~~~~~~~~~~~~~

This module provides functions to

- generate moments according to certain patterns
- compute moments of discrete probability distribution functions
- create transformation rules into moment space
- orthogonalize moment bases


Moment Representation
~~~~~~~~~~~~~~~~~~~~~

Moments can be represented in two ways:

- by an index :math:`i,j`: defined as :math:`m_{ij} := \sum_{\mathbf{d} \in stencil} <\mathbf{d}, \mathbf{x}> f_i`
- or by a polynomial in the variables x,y and z. For example the polynomial :math:`x^2 y^1 z^3 + x + 1` is
  describing the linear combination of moments: :math:`m_{213} + m_{100} + m_{000}`

The polynomial description is more powerful, since one polynomial can express a linear combination of single moments.
All moment polynomials have to use ``MOMENT_SYMBOLS`` (which is a module variable) as degrees of freedom.

Example ::

    >>> from lbmpy.moments import MOMENT_SYMBOLS
    >>> x, y, z = MOMENT_SYMBOLS
    >>> second_order_moment = x*y + y*z


Functions
~~~~~~~~~

"""
import itertools
import math
from collections import Counter, defaultdict
from copy import copy
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar

import sympy as sp

from pystencils.cache import memorycache
from pystencils.sympyextensions import remove_higher_order_terms

MOMENT_SYMBOLS = sp.symbols('x y z')
T = TypeVar('T')


# ------------------------------ Discrete (Exponent Tuples) ------------------------------------------------------------


def moment_multiplicity(exponent_tuple):
    """
    Returns number of permutations of the given moment tuple

    Example:
    >>> moment_multiplicity((2,0,0))
    3
    >>> list(moment_permutations((2,0,0)))
    [(0, 0, 2), (0, 2, 0), (2, 0, 0)]
    """
    c = Counter(exponent_tuple)
    result = math.factorial(len(exponent_tuple))
    for d in c.values():
        result //= math.factorial(d)
    return result


def pick_representative_moments(moments):
    """Picks the representative i.e. of each permutation group only one is kept"""
    to_remove = []
    for m in moments:
        permutations = list(moment_permutations(m))
        to_remove += permutations[1:]
    return set(moments) - set(to_remove)


def moment_permutations(exponent_tuple):
    """Returns all (unique) permutations of the given tuple"""
    return __unique_permutations(exponent_tuple)


def moments_of_order(order, dim=3, include_permutations=True):
    """All tuples of length 'dim' which sum equals 'order'"""
    for item in __fixed_sum_tuples(dim, order, ordered=not include_permutations):
        assert(len(item) == dim)
        assert(sum(item) == order)
        yield item


def moments_up_to_order(order, dim=3, include_permutations=True):
    """All tuples of length 'dim' which sum is smaller than 'order' """
    single_moment_iterators = [moments_of_order(o, dim, include_permutations) for o in range(order + 1)]
    return tuple(itertools.chain(*single_moment_iterators))


def moments_up_to_component_order(order, dim=3):
    """All tuples of length 'dim' where each entry is smaller or equal to 'order' """
    return tuple(itertools.product(*[range(order + 1)] * dim))


def extend_moments_with_permutations(exponent_tuples):
    """Returns all permutations of the given exponent tuples"""
    all_moments = []
    for i in exponent_tuples:
        all_moments += list(moment_permutations(i))
    return __unique(all_moments)


def contained_moments(exponent_tuple, min_order=0, exclude_original=True):
    """Returns all moments contained in exponent_tuple, in the sense that their exponents are less than or
    equal to the corresponding exponents in exponent_tuple."""
    return [t for t
            in itertools.product(*(range(i + 1) for i in exponent_tuple))
            if sum(t) >= min_order and (not exclude_original or t != exponent_tuple)]


# ------------------------------ Representation Conversions ------------------------------------------------------------


def exponent_to_polynomial_representation(exponent_tuple):
    """
    Converts an exponent tuple to corresponding polynomial representation

    Example:
        >>> exponent_to_polynomial_representation( (2,1,3) )
        x**2*y*z**3
    """
    poly = 1
    for sym, tuple_entry in zip(MOMENT_SYMBOLS[:len(exponent_tuple)], exponent_tuple):
        poly *= sym ** tuple_entry
    return poly


def exponents_to_polynomial_representations(sequence_of_exponent_tuples):
    """Applies :func:`exponent_to_polynomial_representation` to given sequence"""
    return tuple([exponent_to_polynomial_representation(t) for t in sequence_of_exponent_tuples])


def polynomial_to_exponent_representation(polynomial, dim=3):
    """
    Converts a linear combination of moments in polynomial representation into exponent representation

    :returns list of tuples where the first element is the coefficient and the second element is the exponent tuple

    Example:
        >>> x , y, z = MOMENT_SYMBOLS
        >>> set(polynomial_to_exponent_representation(1 + (42 * x**2 * y**2 * z) )) == {(42, (2, 2, 1)), (1, (0, 0, 0))}
        True
    """
    assert dim <= 3
    x, y, z = MOMENT_SYMBOLS
    polynomial = polynomial.expand()
    coeff_exp_tuple_representation = []

    summands = [polynomial] if polynomial.func != sp.Add else polynomial.args
    for expr in summands:
        if len(expr.atoms(sp.Symbol) - set(MOMENT_SYMBOLS)) > 0:
            raise ValueError(f"Invalid moment polynomial: {str(expr)}")
        c, x_exp, y_exp, z_exp = sp.Wild('c'), sp.Wild('xexp'), sp.Wild('yexp'), sp.Wild('zc')
        match_res = expr.match(c * x**x_exp * y**y_exp * z**z_exp)
        assert match_res[x_exp].is_integer and match_res[y_exp].is_integer and match_res[z_exp].is_integer
        exp_tuple = (int(match_res[x_exp]), int(match_res[y_exp]), int(match_res[z_exp]),)
        if dim < 3:
            for i in range(dim, 3):
                assert exp_tuple[i] == 0, "Used symbols in polynomial are not representable in that dimension"
            exp_tuple = exp_tuple[:dim]
        coeff_exp_tuple_representation.append((match_res[c], exp_tuple))
    return coeff_exp_tuple_representation


def moment_sort_key(moment):
    """Sort key function for moments to sort them by (in decreasing priority)
     order, number of occuring symbols, length of string representation, string representation"""
    mom_str = str(moment)
    return get_order(moment), len(moment.atoms(sp.Symbol)), len(mom_str), mom_str


def exponent_tuple_sort_key(x):
    return moment_sort_key(exponent_to_polynomial_representation(x))


def sort_moments_into_groups_of_same_order(moments):
    """Returns a dictionary mapping the order (int) to a list of moments with that order."""
    result = defaultdict(list)
    for i, moment in enumerate(moments):
        order = get_order(moment)
        result[order].append(moment)
    return result

# -------------------- Common Function working with exponent tuples and polynomial moments -----------------------------


def statistical_quantity_symbol(name, exponents):
    return sp.Symbol(f'{name}_{"".join(str(i) for i in exponents)}')


def is_even(moment):
    """
    A moment is considered even when under sign reversal nothing changes i.e. :math:`m(-x,-y,-z) = m(x,y,z)`

    For the exponent tuple representation that means that the exponent sum is even  e.g.
        >>> x , y, z = MOMENT_SYMBOLS
        >>> is_even(x**2 * y**2)
        True
        >>> is_even(x**2 * y)
        False
        >>> is_even((1,0,0))
        False
        >>> is_even(1)
        True
    """
    if type(moment) is tuple:
        return sum(moment) % 2 == 0
    else:
        moment = sp.sympify(moment)
        opposite = moment
        for s in MOMENT_SYMBOLS:
            opposite = opposite.subs(s, -s)
        return sp.expand(moment - opposite) == 0


def get_moment_indices(moment_exponent_tuple):
    """Returns indices for a given exponent tuple:

    Example:
        >>> get_moment_indices((2,1,0))
        [0, 0, 1]
        >>> get_moment_indices((0,0,3))
        [2, 2, 2]
    """
    result = []
    for i, element in enumerate(moment_exponent_tuple):
        result += [i] * element
    return result


def get_exponent_tuple_from_indices(moment_indices, dim):
    result = [0] * dim
    for i in moment_indices:
        result[i] += 1
    return tuple(result)


def get_order(moment):
    """Computes polynomial order of given moment.

    Examples:
        >>> x , y, z = MOMENT_SYMBOLS
        >>> get_order(x**2 * y + x)
        3
        >>> get_order(z**4 * x**2)
        6
        >>> get_order((2,1,0))
        3
    """
    if isinstance(moment, tuple):
        return sum(moment)
    if len(moment.atoms(sp.Symbol)) == 0:
        return 0
    leading_coefficient = sp.polys.polytools.LM(moment)
    symbols_in_leading_coefficient = leading_coefficient.atoms(sp.Symbol)
    return sum([sp.degree(leading_coefficient, gen=m) for m in symbols_in_leading_coefficient])


def non_aliased_moment(moment_tuple: Sequence[int]) -> Tuple[int, ...]:
    """Takes a moment exponent tuple and returns the non-aliased version of it.

    For first neighborhood stencils, all moments with exponents 3, 5, 7... are equal to exponent 1,
    and exponents 4, 6, 8... are equal to exponent 2. This is because first neighborhood stencils only have values
    d ∈ {+1, 0, -1}. So for example d**5 is always the same as d**3 and d, and d**6 == d**4 == d**2

    Example:
         >>> non_aliased_moment((5, 4, 2))
         (1, 2, 2)
         >>> non_aliased_moment((9, 1, 2))
         (1, 1, 2)
    """
    moment = list(moment_tuple)
    result = []
    for element in moment:
        if element > 2:
            result.append(2 - (element % 2))
        else:
            result.append(element)
    return tuple(result)


def aliases_from_moment_list(moment_exponents, stencil):
    """Takes a list of moment exponent tuples and finds aliases within it
    according the given stencil. Two moments are aliases of each other on a stencil
    if they produce the same coefficients in the moment sum over all populations.

    Apart from the obvious aliases (e.g. ``(4,0,0)`` to ``(2,0,0)``, etc), there are aliasing
    effects for example on the D3Q15 stencil, where some third and fourth order moments
    are the same.
    """
    mm = moment_matrix(moment_exponents, stencil)
    rows_dict = dict()
    aliases = dict()
    for r, moment in enumerate(moment_exponents):
        row = tuple(mm[r, :])
        if row in rows_dict:
            aliases[moment] = rows_dict[row]
        else:
            rows_dict[row] = moment
    return aliases


def non_aliased_polynomial_raw_moments(polys, stencil, nested=False):
    """Takes a (potentially nested) list of raw moment polynomials and rewrites them by eliminating
    any aliased monomials. 

    All polynomials are expanded and occuring monomials are collected. Using `aliases_from_moment_list`,
    aliases are eliminated and substituted in the polynomials. 

    Attention: Only use this method for monomials in raw moment space. It will produce wrong results
    if used for central moments, since there is no direct aliasing in central moment space!
    """
    dim = len(stencil[0])
    if nested:
        polys_unnested = list(itertools.chain.from_iterable(polys))
    else:
        polys_unnested = polys
    monos = sorted(extract_monomials(polys_unnested, dim=dim), key=exponent_tuple_sort_key)
    aliases = aliases_from_moment_list(monos, stencil)

    if not aliases:  # Stop early if there are no aliases
        return polys

    def nonalias_polynomial(poly):
        exponents = polynomial_to_exponent_representation(poly, dim)
        exponents_unaliased = [(coeff, aliases.get(m, m)) for coeff, m in exponents]
        return sum(coeff * exponent_to_polynomial_representation(m) for coeff, m in exponents_unaliased)

    if nested:
        output_polys = []
        for group in polys:
            output_polys.append(list(map(nonalias_polynomial, group)))
        return output_polys
    else:
        return list(map(nonalias_polynomial, polys))


def is_bulk_moment(moment, dim):
    """The bulk moment is x**2+y**2+z**2"""
    if type(moment) is not tuple:
        moment = polynomial_to_exponent_representation(moment)
    quadratic = False
    found = [0 for _ in range(dim)]
    for prefactor, monomial in moment:
        if sum(monomial) == 2:
            quadratic = True
            for i, exponent in enumerate(monomial[:dim]):
                if exponent == 2:
                    found[i] += prefactor
        elif sum(monomial) > 2:
            return False
    return quadratic and found != [0] * dim and len(set(found)) == 1


def is_shear_moment(moment, dim):
    """Shear moments are the quadratic polynomials except for the bulk moment.
       Linear combinations with lower-order polynomials don't harm because these correspond to conserved moments."""
    if is_bulk_moment(moment, dim):
        return False
    if type(moment) is not tuple:
        moment = polynomial_to_exponent_representation(moment)
    quadratic = False
    for prefactor, monomial in moment:
        if sum(monomial) == 2:
            quadratic = True
        elif sum(monomial) > 2:
            return False
    return quadratic


@memorycache(maxsize=512)
def discrete_moment(func, moment, stencil, shift_velocity=None):
    r"""
    Computes discrete moment of given distribution function

    .. math ::
        \sum_{d \in stencil} p(d) f_i

    where :math:`p(d)` is the moment polynomial where :math:`x, y, z` have been replaced with the components of the
    stencil direction, and :math:`f_i` is the i'th entry in the passed function sequence

    Args:
        func: list of distribution functions for each direction
        moment: can either be a exponent tuple, or a sympy polynomial expression
        stencil: sequence of directions
        shift_velocity: velocity vector :math:`\mathbf{u}` to compute central moments, the lattice
                        velocity is replaced by (lattice_velocity - shift_velocity)
    """
    assert stencil.Q == len(func)
    if shift_velocity is None:
        shift_velocity = (0,) * len(stencil[0])
    res = 0
    for factor, e in zip(func, stencil):
        if type(moment) is tuple:
            for vel, shift, exponent in zip(e, shift_velocity, moment):
                factor *= (vel - shift) ** exponent
            res += factor
        else:
            weight = moment
            for variable, e_i, shift in zip(MOMENT_SYMBOLS, e, shift_velocity):
                weight = weight.subs(variable, e_i - shift)
            res += weight * factor

    return res


def moment_matrix(moments, stencil, shift_velocity=None):
    """
    Returns transformation matrix to moment space

    each row corresponds to a moment, each column to a direction of the stencil
    The entry i,j is the i'th moment polynomial evaluated at direction j
    """
    if shift_velocity is None:
        shift_velocity = (0,) * len(stencil[0])

    if type(moments[0]) is tuple:
        def generator(row, column):
            result = sp.Rational(1, 1)
            for exponent, stencil_entry, shift in zip(moments[row], stencil[column], shift_velocity):
                result *= (sp.sympify(stencil_entry) - shift) ** exponent
            return result
    else:
        def generator(row, column):
            evaluated = moments[row]
            for var, stencil_entry, shift in zip(MOMENT_SYMBOLS, stencil[column], shift_velocity):
                evaluated = evaluated.subs(var, stencil_entry - shift)
            return evaluated

    return sp.Matrix(len(moments), stencil.Q, generator)


def set_up_shift_matrix(moments, stencil, velocity_symbols=sp.symbols("u_:3")):
    """
    Sets up a shift matrix to shift raw moments to central moment space.

    Args:
        - moments: Sequence of polynomials or sequence of exponent tuples, sorted
                   ascendingly by moment order.
        - stencil: Nested tuple of lattice velocities
        - velocity_symbols: Sequence of symbols corresponding to the shift velocity
    """
    dim = len(stencil[0])
    if len(velocity_symbols) > dim:
        velocity_symbols = velocity_symbols[:dim]

    M = moment_matrix(moments, stencil, shift_velocity=None)
    MN = moment_matrix(moments, stencil, shift_velocity=velocity_symbols)

    N = sp.simplify(MN * M.inv())

    assert N.is_lower, "Calculating the shift matrix gave not a lower diagonal matrix. Thus it failed"
    assert sum(N[i, i] for i in range(stencil.Q)) == stencil.Q, "Calculating the shift matrix failed. " \
        "There are entries on the diagonal which are not equal to one"

    return N


def gram_schmidt(moments, stencil, weights=None):
    r"""
    Computes orthogonal set of moments using the method by Gram-Schmidt

    Args:
        moments: sequence of moments, either in tuple or polynomial form
        stencil: stencil as sequence of directions
        weights: optional weights, that define the scalar product which is used for normalization.
                 Scalar product :math:`< \mathbf{a},\mathbf{b} > = \sum a_i b_i w_i` with weights :math:`w_i`.
                 Passing no weights sets all weights to 1.
    Returns:
        set of orthogonal moments in polynomial form
    """
    if weights is None:
        weights = sp.eye(stencil.Q)
    if type(weights) is list:
        assert len(weights) == stencil.Q
        weights = sp.diag(*weights)

    if type(moments[0]) is tuple:
        moments = list(exponents_to_polynomial_representations(moments))
    else:
        moments = list(copy(moments))

    pdfs_to_moments = moment_matrix(moments, stencil).transpose()
    moment_matrix_columns = [pdfs_to_moments.col(i) for i in range(pdfs_to_moments.cols)]
    orthogonalized_vectors = []
    for i in range(len(moment_matrix_columns)):
        current_element = moment_matrix_columns[i]
        for j in range(i):
            prev_element = orthogonalized_vectors[j]
            denominator = prev_element.dot(weights * prev_element)
            if denominator == 0:
                raise ValueError(f"Not an independent set of vectors given: "
                                 f"vector {i} is dependent on previous vectors")
            overlap = current_element.dot(weights * prev_element) / denominator
            current_element -= overlap * prev_element
            moments[i] -= overlap * moments[j]
        orthogonalized_vectors.append(current_element)

    return moments


def get_default_moment_set_for_stencil(stencil):
    """
    Returns a sequence of moments that are commonly used to construct a LBM equilibrium for the given stencil
    """
    to_poly = exponents_to_polynomial_representations

    if stencil.D == 2 and stencil.Q == 9:
        return sorted(to_poly(moments_up_to_component_order(2, dim=2)), key=moment_sort_key)

    all27_moments = moments_up_to_component_order(2, dim=3)
    if stencil.D == 3 and stencil.Q == 27:
        return sorted(to_poly(all27_moments), key=moment_sort_key)
    if stencil.D == 3 and stencil.Q == 19:
        non_matched_moments = [(1, 2, 2), (1, 1, 2), (2, 2, 2), (1, 1, 1)]
        moments19 = set(all27_moments) - set(extend_moments_with_permutations(non_matched_moments))
        return sorted(to_poly(moments19), key=moment_sort_key)
    if stencil.D == 3 and stencil.Q == 15:
        x, y, z = MOMENT_SYMBOLS
        non_matched_moments = [(1, 2, 0), (2, 2, 0), (1, 1, 2), (1, 2, 2), (2, 2, 2)]
        additional_moments = (6 * (x ** 2 * y ** 2 + x ** 2 * z ** 2 + y ** 2 * z ** 2),
                              3 * (x * (y ** 2 + z ** 2)),
                              3 * (y * (x ** 2 + z ** 2)),
                              3 * (z * (x ** 2 + y ** 2)),
                              )
        to_remove = set(extend_moments_with_permutations(non_matched_moments))
        return sorted(to_poly(set(all27_moments) - to_remove) + additional_moments, key=moment_sort_key)
    if stencil.D == 3 and stencil.Q == 7:
        x, y, z = MOMENT_SYMBOLS
        non_matched_moments = [(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0),
                               (1, 2, 2), (2, 0, 0), (2, 2, 0), (2, 2, 2)]
        additional_moments = (x ** 2 - y ** 2,
                              x ** 2 - z ** 2,
                              x ** 2 + y ** 2 + z ** 2)
        to_remove = set(extend_moments_with_permutations(non_matched_moments))
        return sorted(to_poly(set(all27_moments) - to_remove) + additional_moments, key=moment_sort_key)

    raise NotImplementedError("No default moment system available for this stencil - define matched moments yourself")


def extract_monomials(sequence_of_polynomials, dim=3):
    """
    Returns a set of exponent tuples of all monomials contained in the given set of polynomials

    Args:
        sequence_of_polynomials: sequence of polynomials in the MOMENT_SYMBOLS
        dim: length of returned exponent tuples

    >>> x, y, z = MOMENT_SYMBOLS
    >>> extract_monomials([x**2 + y**2 + y, y + y**2]) == {(0, 1, 0),(0, 2, 0),(2, 0, 0)}
    True
    >>> extract_monomials([x**2 + y**2 + y, y + y**2], dim=2) == {(0, 1), (0, 2), (2, 0)}
    True
    """
    monomials = set()
    for polynomial in sequence_of_polynomials:
        for factor, exponent_tuple in polynomial_to_exponent_representation(polynomial):
            monomials.add(exponent_tuple[:dim])
    return monomials


def monomial_to_polynomial_transformation_matrix(monomials, polynomials):
    """
    Returns a transformation matrix from a monomial to a polynomial representation

    Args:
        monomials: sequence of exponent tuples
        polynomials: sequence of polynomials in the MOMENT_SYMBOLS

    >>> x, y, z = MOMENT_SYMBOLS
    >>> polys = [7 * x**2 + 3 * x + 2 * y **2, 9 * x**2 - 5 * x]
    >>> mons = list(extract_monomials(polys, dim=2))
    >>> mons.sort()
    >>> monomial_to_polynomial_transformation_matrix(mons, polys)
    Matrix([
    [2,  3, 7],
    [0, -5, 9]])
    """
    dim = len(monomials[0])

    result = sp.zeros(len(polynomials), len(monomials))
    for polynomial_idx, polynomial in enumerate(polynomials):
        for factor, exponent_tuple in polynomial_to_exponent_representation(polynomial):
            exponent_tuple = exponent_tuple[:dim]
            result[polynomial_idx, monomials.index(exponent_tuple)] = factor
    return result


def central_moment_reduced_monomial_to_polynomial_matrix(polynomials, stencil, velocity_symbols=None):
    r"""
    Returns a transformation matrix from a reduced set of monomial central moments to a set of polynomial
    central moments.

    Use for a set of :math:`q` central moment polynomials that reduces to a too-large set of :math:`r > q` 
    monomials (as given by `extract_monomials`). Reduces the monomials by eliminating aliases in raw moment
    space, and computes a reduced polynomialization matrix :math:`\mathbf{P}^{r}` that computes the given
    polynomials from the reduced set of monomials.
    """
    dim = len(stencil[0])
    if velocity_symbols is None:
        velocity_symbols = sp.symbols(f"u_:{dim}")

    reduced_polynomials = non_aliased_polynomial_raw_moments(polynomials, stencil)
    reduced_monomials = sorted(extract_monomials(reduced_polynomials), key=exponent_tuple_sort_key)

    assert len(reduced_monomials) == stencil.Q, "Could not extract a base set of correct size from the polynomials."

    N = set_up_shift_matrix(reduced_monomials, stencil, velocity_symbols=velocity_symbols)
    N_P = set_up_shift_matrix(polynomials, stencil, velocity_symbols=velocity_symbols)
    P_mono = monomial_to_polynomial_transformation_matrix(reduced_monomials, reduced_polynomials)

    P_reduced = (N_P * P_mono * N.inv()).expand()
    return P_reduced, reduced_monomials


# ---------------------------------- Visualization ---------------------------------------------------------------------


def moment_equality_table(stencil, discrete_equilibrium=None, continuous_equilibrium=None,
                          max_order=4, truncate_order=3):
    """
    Creates a table showing which moments of a discrete stencil/equilibrium coincide with the
    corresponding continuous moments

    Args:
        stencil: list of stencil velocities
        discrete_equilibrium: list of sympy expr to compute discrete equilibrium for each direction, if left
                             to default the standard discrete Maxwellian equilibrium is used
        continuous_equilibrium: continuous equilibrium, if left to default, the continuous Maxwellian is used
        max_order: compare moments up to this order (number of rows in table)
        truncate_order: moments are considered equal if they match up to this order

    Returns:
        Object to display in an Jupyter notebook
    """
    import ipy_table
    from lbmpy.continuous_distribution_measures import continuous_moment

    u = sp.symbols(f"u_:{stencil.D}")
    if discrete_equilibrium is None:
        from lbmpy.maxwellian_equilibrium import discrete_maxwellian_equilibrium
        discrete_equilibrium = discrete_maxwellian_equilibrium(stencil, c_s_sq=sp.Rational(1, 3), compressible=True,
                                                               u=u, order=truncate_order)
    if continuous_equilibrium is None:
        from lbmpy.maxwellian_equilibrium import continuous_maxwellian_equilibrium
        continuous_equilibrium = continuous_maxwellian_equilibrium(dim=stencil.D, u=u, c_s_sq=sp.Rational(1, 3))

    table = []
    matched_moments = 0
    non_matched_moments = 0

    moments_list = [list(moments_of_order(o, stencil.D, include_permutations=False)) for o in range(max_order + 1)]

    colors = dict()
    nr_of_columns = max([len(v) for v in moments_list]) + 1

    header_row = [' '] * nr_of_columns
    header_row[0] = 'order'
    table.append(header_row)

    for order, moments in enumerate(moments_list):
        row = [' '] * nr_of_columns
        row[0] = f'{order}'
        for moment, col_idx in zip(moments, range(1, len(row))):
            multiplicity = moment_multiplicity(moment)
            dm = discrete_moment(discrete_equilibrium, moment, stencil)
            cm = continuous_moment(continuous_equilibrium, moment, symbols=sp.symbols("v_0 v_1 v_2")[:stencil.D])
            difference = sp.simplify(dm - cm)
            if truncate_order:
                difference = sp.simplify(remove_higher_order_terms(difference, symbols=u, order=truncate_order))
            if difference != 0:
                colors[(order + 1, col_idx)] = 'Orange'
                non_matched_moments += multiplicity
            else:
                colors[(order + 1, col_idx)] = 'lightGreen'
                matched_moments += multiplicity

            row[col_idx] = f'{moment}  x {moment_multiplicity(moment)}'

        table.append(row)

    table_display = ipy_table.make_table(table)
    ipy_table.set_row_style(0, color='#ddd')
    for cell_idx, color in colors.items():
        ipy_table.set_cell_style(cell_idx[0], cell_idx[1], color=color)

    print(f"Matched moments {matched_moments} - non matched moments {non_matched_moments} "
          f"- total {matched_moments + non_matched_moments}")

    return table_display


def moment_equality_table_by_stencil(name_to_stencil_dict, moments, truncate_order=3):
    """
    Creates a table for display in IPython notebooks that shows which moments agree between continuous and
    discrete equilibrium, group by stencils

    Args:
        name_to_stencil_dict: dict from stencil name to stencil
        moments: sequence of moments to compare - assumes that permutations have similar properties
                 so just one representative is shown labeled with its multiplicity
        truncate_order: compare up to this order
    """
    import ipy_table
    from lbmpy.maxwellian_equilibrium import discrete_maxwellian_equilibrium
    from lbmpy.maxwellian_equilibrium import continuous_maxwellian_equilibrium
    from lbmpy.continuous_distribution_measures import continuous_moment

    stencil_names = []
    stencils = []
    for key, value in name_to_stencil_dict.items():
        stencil_names.append(key)
        stencils.append(value)

    moments = list(pick_representative_moments(moments))

    colors = {}
    for stencil_idx, stencil in enumerate(stencils):
        dim = stencil.D
        u = sp.symbols(f"u_:{dim}")
        discrete_equilibrium = discrete_maxwellian_equilibrium(stencil, c_s_sq=sp.Rational(1, 3), compressible=True,
                                                               u=u, order=truncate_order)
        continuous_equilibrium = continuous_maxwellian_equilibrium(dim=dim, u=u, c_s_sq=sp.Rational(1, 3))

        for moment_idx, moment in enumerate(moments):
            moment = moment[:dim]
            dm = discrete_moment(discrete_equilibrium, moment, stencil)
            cm = continuous_moment(continuous_equilibrium, moment, symbols=sp.symbols("v_0 v_1 v_2")[:dim])
            difference = sp.simplify(dm - cm)
            if truncate_order:
                difference = sp.simplify(remove_higher_order_terms(difference, symbols=u, order=truncate_order))
            colors[(moment_idx + 1, stencil_idx + 2)] = 'Orange' if difference != 0 else 'lightGreen'

    table = []
    header_row = [' ', '#'] + stencil_names
    table.append(header_row)
    for moment in moments:
        row = [str(moment), str(moment_multiplicity(moment))] + [' '] * len(stencils)
        table.append(row)

    table_display = ipy_table.make_table(table)
    ipy_table.set_row_style(0, color='#ddd')
    for cell_idx, color in colors.items():
        ipy_table.set_cell_style(cell_idx[0], cell_idx[1], color=color)

    return table_display


# --------------------------------------- Internal Functions -----------------------------------------------------------

def __unique(seq: Sequence[T]) -> List[T]:
    """Removes duplicates from a sequence in an order preserving way.

    Example:
        >>> __unique((1, 2, 3, 1, 4, 6, 3))
        [1, 2, 3, 4, 6]
    """
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


def __unique_permutations(elements: Sequence[T]) -> Iterable[T]:
    """Generates all unique permutations of the passed sequence.

    Example:
        >>> list(__unique_permutations([1, 1, 2]))
        [(1, 1, 2), (1, 2, 1), (2, 1, 1)]

    """
    if len(elements) == 1:
        yield elements[0],
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in __unique_permutations(remaining_elements):
                yield (first_element,) + sub_permutation


def __fixed_sum_tuples(tuple_length: int, tuple_sum: int,
                       allowed_values: Optional[Sequence[int]] = None,
                       ordered: bool = False) -> Iterable[Tuple[int, ...]]:
    """Generates all possible tuples of positive integers with a fixed sum of all entries.

    Args:
        tuple_length: length of the returned tuples
        tuple_sum: summing over the entries of a yielded tuple should give this number
        allowed_values: optional sequence of positive integers that are considered as tuple entries
                        zero has to be in the set of allowed values
                        if None, all possible positive integers are allowed
        ordered: if True, only tuples are returned where the entries are descending, i.e. where the entries are ordered

    Examples:
        Generate all 2-tuples where the sum of entries is 3
        >>> list(__fixed_sum_tuples(tuple_length=2, tuple_sum=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        Same with ordered tuples only
        >>> list(__fixed_sum_tuples(tuple_length=2, tuple_sum=3, ordered=True))
        [(2, 1), (3, 0)]

        Restricting the allowed values, note that zero has to be in the allowed values!
        >>> list(__fixed_sum_tuples(tuple_length=3, tuple_sum=4, allowed_values=(0, 1, 3)))
        [(0, 1, 3), (0, 3, 1), (1, 0, 3), (1, 3, 0), (3, 0, 1), (3, 1, 0)]
    """
    if not allowed_values:
        allowed_values = set(range(0, tuple_sum + 1))

    assert 0 in allowed_values and all(i >= 0 for i in allowed_values)

    def recursive_helper(current_list, position, total_sum):
        new_position = position + 1
        if new_position < len(current_list):
            for i in allowed_values:
                current_list[position] = i
                new_sum = total_sum - i
                if new_sum < 0:
                    continue
                for item in recursive_helper(current_list, new_position, new_sum):
                    yield item
        else:
            if total_sum in allowed_values:
                current_list[-1] = total_sum
                if not ordered:
                    yield tuple(current_list)
                if ordered and current_list == sorted(current_list, reverse=True):
                    yield tuple(current_list)

    return recursive_helper([0] * tuple_length, 0, tuple_sum)
