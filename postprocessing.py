import numpy as np


def vectorFieldInterpolator(vectorField):
    from scipy.interpolate import RegularGridInterpolator
    coords = [np.arange(s) + 0.5 for s in  vectorField.shape[:-1]]
    return RegularGridInterpolator(*coords, vectorField)


def scalarFieldInterpolator(scalarField):
    from scipy.interpolate import RegularGridInterpolator
    coords = [np.arange(s) + 0.5 for s in scalarField.shape]
    return RegularGridInterpolator(coords, values=scalarField)


def vorticity2D(velocityField):
    assert len(velocityField.shape) == 3
    assert velocityField.shape[2] == 2
    grad_y_of_x = np.gradient(velocityField[:, :, 0], axis=1)
    grad_x_of_y = np.gradient(velocityField[:, :, 1], axis=0)
    return grad_x_of_y - grad_y_of_x

