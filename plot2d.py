from matplotlib.pyplot import *
import numpy as np
from numpy.linalg import norm


def removeGhostLayers(field):
    return field[1:-1, 1:-1]


def vectorField(field, step=2, **kwargs):
    field = removeGhostLayers(field)
    veln = field.swapaxes(0, 1)
    quiver(veln[::step, ::step, 0], veln[::step, ::step, 1], **kwargs)


def vectorFieldMagnitude(field, **kwargs):
    field = norm(field, axis=2, ord=2)
    scalarField(field, **kwargs)


def scalarField(field, **kwargs):
    field = removeGhostLayers(field)
    field = np.swapaxes(field, 0, 1)
    imshow(field, origin='lower', **kwargs)

