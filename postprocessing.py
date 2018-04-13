import numpy as np


def vector_field_interpolator(vector_field):
    from scipy.interpolate import RegularGridInterpolator
    coordinates = [np.arange(s) + 0.5 for s in vector_field.shape[:-1]]
    return RegularGridInterpolator(coordinates, values=vector_field)


def scalar_field_interpolator(scalar_field):
    from scipy.interpolate import RegularGridInterpolator
    coordinates = [np.arange(s) + 0.5 for s in scalar_field.shape]
    return RegularGridInterpolator(coordinates, values=scalar_field)


def vorticity_2d(velocity_field):
    assert len(velocity_field.shape) == 3
    assert velocity_field.shape[2] == 2
    grad_y_of_x = np.gradient(velocity_field[:, :, 0], axis=1)
    grad_x_of_y = np.gradient(velocity_field[:, :, 1], axis=0)
    return grad_x_of_y - grad_y_of_x
