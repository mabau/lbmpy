import sympy as sp
import pystencils as ps

from pystencils.field import Field


def welford_assignments(field, mean_field, sum_of_products_field=None):
    r"""
    Creates the assignments for the kernel creation of Welford's algorithm
    (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm).
    This algorithm is an online algorithm for the mean and variance calculation of sample data, here implemented for
    the execution on vector fields, e.g., velocity.
    The calculation of the variance is optional and only performed if a field for the sum of products is given.

    The mean value is directly updated in the mean vector field.
    The variance must be retrieved in a post-processing step. Let :math `M_{2,n}` denote the value of the sum of
    products after the first :math `n` samples. According to Welford the biased sample variance
    :math `\sigma_n^2` and the unbiased sample variance :math `s_n^2` are given by

    .. math ::
        \sigma_n^2 = \frac{M_{2,n}}{n}
        s_n^2 = \frac{M_{2,n}}{n-1},

    respectively.
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

    if sum_of_products_field is None:
        # sum of products will not be calculated, i.e., the variance is not retrievable
        welford_sum_of_products_field = None
    else:
        if isinstance(sum_of_products_field, Field):
            welford_sum_of_products_field = sum_of_products_field.center
            assert sum_of_products_field.values_per_cell() == dim**2, \
                (f"Sum of products field does not have the right layout. "
                 f"Index dimension: {sum_of_products_field.values_per_cell()}, expected: {dim**2}")
        elif isinstance(sum_of_products_field, Field.Access):
            welford_sum_of_products_field = sum_of_products_field
            assert sum_of_products_field.field.values_per_cell() == dim**2, \
                (f"Sum of products field does not have the right layout. "
                 f"Index dimension: {sum_of_products_field.field.values_per_cell()}, expected: {dim**2}")
        else:
            raise ValueError("Variance vector field has to be a pystencils Field or Field.Access")

    counter = sp.Symbol('counter')
    delta = sp.symbols(f"delta_:{dim}")
    delta2 = sp.symbols(f"delta2_:{dim}")

    main_assignments = list()

    for i in range(dim):
        main_assignments.append(ps.Assignment(delta[i], welford_field.at_index(i) - welford_mean_field.at_index(i)))
        main_assignments.append(ps.Assignment(welford_mean_field.at_index(i), welford_mean_field.at_index(i) + delta[i] / counter))
        main_assignments.append(ps.Assignment(delta2[i], welford_field.at_index(i) - welford_mean_field.at_index(i)))

    if sum_of_products_field is not None:
        for i in range(dim):
            for j in range(dim):
                idx = i * dim + j
                main_assignments.append(ps.Assignment(welford_sum_of_products_field.at_index(idx),
                                                      welford_sum_of_products_field.at_index(idx) + delta[i] * delta2[j]))

    return main_assignments
