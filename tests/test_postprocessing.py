import numpy as np
import pytest

from lbmpy.postprocessing import scalar_field_interpolator, vector_field_interpolator


def test_interpolation():
    pytest.importorskip('scipy.ndimage')

    scalar_arr = np.arange(0, 3*3).reshape(3, 3)
    scalar_ip = scalar_field_interpolator(scalar_arr)
    np.testing.assert_equal(scalar_ip([[1, 1.5], [0.5, 1]]), [2.5, 0.5])

    vector_arr = np.arange(0, 3 * 3 * 2).reshape(3, 3, 2)
    vector_ip = vector_field_interpolator(vector_arr)
    np.testing.assert_equal(vector_ip([[1, 1.5], [0.5, 1]]), [[5., 6.], [1., 2.]])
