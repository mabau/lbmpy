import sympy as sp
import pystencils as ps

from pystencils.field import Field


def welford_assignments(field, mean_field, sum_of_squares_field=None, sum_of_cubes_field=None):
    r"""
    Creates the assignments for the kernel creation of Welford's algorithm
    (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm).
    This algorithm is an online algorithm for the mean and variance calculation of sample data, here implemented for
    the execution on scalar or vector fields, e.g., velocity.
    The calculation of the variance / the third-order central moments is optional and only performed if a field for
    the sum of squares / sum of cubes is given.

    The mean value is directly updated in the mean vector field.
    The variance / covariance must be retrieved in a post-processing step. Let :math `M_{2,n}` denote the value of the
    sum of squares after the first :math `n` samples. According to Welford the biased sample variance
    :math `\sigma_n^2` and the unbiased sample variance :math `s_n^2` are given by

    .. math ::
        \sigma_n^2 = \frac{M_{2,n}}{n}
        s_n^2 = \frac{M_{2,n}}{n-1},

    respectively.

    Liekwise, to the 3rd-order central moment(s) of the (vector) field, the sum of cubes field must be divided
    by :math `n` in a post-processing step.
    """

    if isinstance(field, Field):
        dim = field.values_per_cell()
        welford_field = field.center
    elif isinstance(field, Field.Access):
        dim = field.field.values_per_cell()
        welford_field = field
    else:
        raise ValueError("Vector field has to be a pystencils Field or Field.Access")

    if isinstance(mean_field, Field):
        welford_mean_field = mean_field.center
    elif isinstance(mean_field, Field.Access):
        welford_mean_field = mean_field
    else:
        raise ValueError("Mean vector field has to be a pystencils Field or Field.Access")

    if sum_of_squares_field is None:
        # sum of products will not be calculated, i.e., the covariance matrix is not retrievable
        welford_sum_of_squares_field = None
    else:
        if isinstance(sum_of_squares_field, Field):
            welford_sum_of_squares_field = sum_of_squares_field.center
        elif isinstance(sum_of_squares_field, Field.Access):
            welford_sum_of_squares_field = sum_of_squares_field
        else:
            raise ValueError("Sum of squares field has to be a pystencils Field or Field.Access")

        assert welford_sum_of_squares_field.field.values_per_cell() == dim ** 2, \
            (f"Sum of squares field does not have the right layout. "
             f"Index dimension: {welford_sum_of_squares_field.field.values_per_cell()}, expected: {dim ** 2}")

    if sum_of_cubes_field is None:
        # sum of cubes will not be calculated, i.e., the 3rd-order central moments are not retrievable
        welford_sum_of_cubes_field = None
    else:
        if isinstance(sum_of_cubes_field, Field):
            welford_sum_of_cubes_field = sum_of_cubes_field.center
        elif isinstance(sum_of_cubes_field, Field.Access):
            welford_sum_of_cubes_field = sum_of_cubes_field
        else:
            raise ValueError("Sum of cubes field has to be a pystencils Field or Field.Access")

        assert welford_sum_of_cubes_field.field.values_per_cell() == dim ** 3, \
            (f"Sum of cubes field does not have the right layout. "
             f"Index dimension: {welford_sum_of_cubes_field.field.values_per_cell()}, expected: {dim ** 3}")

    # for the calculation of the thrid-order moments, the variance must also be calculated
    if welford_sum_of_cubes_field is not None:
        assert welford_sum_of_squares_field is not None

    # actual assignments

    counter = sp.Symbol('counter')
    delta = sp.symbols(f"delta_:{dim}")

    main_assignments = list()

    for i in range(dim):
        main_assignments.append(ps.Assignment(delta[i], welford_field.at_index(i) - welford_mean_field.at_index(i)))
        main_assignments.append(ps.Assignment(welford_mean_field.at_index(i),
                                              welford_mean_field.at_index(i) + delta[i] / counter))

    if sum_of_squares_field is not None:
        delta2 = sp.symbols(f"delta2_:{dim}")
        for i in range(dim):
            main_assignments.append(
                ps.Assignment(delta2[i], welford_field.at_index(i) - welford_mean_field.at_index(i)))
        for i in range(dim):
            for j in range(dim):
                idx = i * dim + j
                main_assignments.append(ps.Assignment(
                    welford_sum_of_squares_field.at_index(idx),
                    welford_sum_of_squares_field.at_index(idx) + delta[i] * delta2[j]))

        if sum_of_cubes_field is not None:
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        idx = (i * dim + j) * dim + k
                        main_assignments.append(ps.Assignment(
                            welford_sum_of_cubes_field.at_index(idx),
                            welford_sum_of_cubes_field.at_index(idx)
                            - delta[k] / counter * welford_sum_of_squares_field(i * dim + j)
                            - delta[j] / counter * welford_sum_of_squares_field(k * dim + i)
                            - delta[i] / counter * welford_sum_of_squares_field(j * dim + k)
                            + delta2[i] * (2 * delta[j] - delta2[j]) * delta[k]
                        ))

    return main_assignments
