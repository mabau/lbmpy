import pytest
import numpy as np
import pystencils as ps
from lbmpy.flow_statistics import welford_assignments


@pytest.mark.parametrize('order', [1, 2, 3])
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_welford(order, dim):

    target = ps.Target.CPU

    # create data handling and fields
    dh = ps.create_data_handling((1, 1, 1), periodicity=False, default_target=target)

    vector_field = dh.add_array('vector', values_per_cell=dim)
    dh.fill(vector_field.name, 0.0, ghost_layers=False)

    mean_field = dh.add_array('mean', values_per_cell=dim)
    dh.fill(mean_field.name, 0.0, ghost_layers=False)

    if order >= 2:
        sos = dh.add_array('sos', values_per_cell=dim**2)
        dh.fill(sos.name, 0.0, ghost_layers=False)

        if order == 3:
            soc = dh.add_array('soc', values_per_cell=dim**3)
            dh.fill(soc.name, 0.0, ghost_layers=False)
        else:
            soc = None
    else:
        sos = None
        soc = None

    welford = welford_assignments(field=vector_field, mean_field=mean_field,
                                  sum_of_squares_field=sos, sum_of_cubes_field=soc)

    welford_kernel = ps.create_kernel(welford).compile()

    # set random seed
    np.random.seed(42)
    n = int(1e4)
    x = np.random.normal(size=n * dim).reshape(n, dim)

    analytical_mean = np.zeros(dim)
    analytical_covariance = np.zeros(dim**2)
    analytical_third_order_moments = np.zeros(dim**3)

    # calculate analytical mean
    for i in range(dim):
        analytical_mean[i] = np.mean(x[:, i])

    # calculate analytical covariances
    for i in range(dim):
        for j in range(dim):
            analytical_covariance[i * dim + j] \
                = (np.sum((x[:, i] - analytical_mean[i]) * (x[:, j] - analytical_mean[j]))) / n

    # calculate analytical third-order central moments
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                analytical_third_order_moments[(i * dim + j) * dim + k] \
                    = (np.sum((x[:, i] - analytical_mean[i])
                              * (x[:, j] - analytical_mean[j])
                              * (x[:, k] - analytical_mean[k]))) / n

    # Time loop
    counter = 1
    for i in range(n):
        for d in range(dim):
            new_value = x[i, d]
            if dim > 1:
                dh.fill(array_name=vector_field.name, val=new_value, value_idx=d, ghost_layers=False)
            else:
                dh.fill(array_name=vector_field.name, val=new_value, ghost_layers=False)
        dh.run_kernel(welford_kernel, counter=counter)
        counter += 1

    welford_mean = dh.gather_array(mean_field.name).flatten()
    np.testing.assert_allclose(welford_mean, analytical_mean)

    if order >= 2:
        welford_covariance = dh.gather_array(sos.name).flatten() / n
        np.testing.assert_allclose(welford_covariance, analytical_covariance)
        if order == 3:
            welford_third_order_moments = dh.gather_array(soc.name).flatten() / n
            np.testing.assert_allclose(welford_third_order_moments, analytical_third_order_moments)


if __name__ == '__main__':
    test_welford(1, 1)
