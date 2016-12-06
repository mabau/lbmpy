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
    return scalarField(field, **kwargs)


def scalarField(field, **kwargs):
    field = removeGhostLayers(field)
    field = np.swapaxes(field, 0, 1)
    return imshow(field, origin='lower', **kwargs)


def vectorFieldMagnitudeAnimation(runFunction, plotSetupFunction=lambda: None,
                                  plotUpdateFunction=lambda: None, interval=30, frames=180, **kwargs):
    import matplotlib.animation as animation

    fig = gcf()
    im = None
    field = runFunction()
    im = vectorFieldMagnitude(field, **kwargs)
    plotSetupFunction()

    def updatefig(*args):
        field = runFunction()
        field = norm(field, axis=2, ord=2)
        field = np.swapaxes(field, 0, 1)
        im.set_array(field)
        plotUpdateFunction()
        return im,

    return animation.FuncAnimation(fig, updatefig, interval=interval, frames=frames)
