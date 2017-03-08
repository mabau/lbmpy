from matplotlib.pyplot import *


def vectorField(field, step=2, **kwargs):
    veln = field.swapaxes(0, 1)
    quiver(veln[::step, ::step, 0], veln[::step, ::step, 1], **kwargs)


def vectorFieldMagnitude(field, **kwargs):
    from numpy.linalg import norm
    field = norm(field, axis=2, ord=2)
    return scalarField(field, **kwargs)


def scalarField(field, **kwargs):
    import numpy as np
    field = np.swapaxes(field, 0, 1)
    return imshow(field, origin='lower', **kwargs)


def vectorFieldMagnitudeAnimation(runFunction, plotSetupFunction=lambda: None,
                                  plotUpdateFunction=lambda: None, interval=30, frames=180, **kwargs):
    import matplotlib.animation as animation
    import numpy as np
    from numpy.linalg import norm

    fig = gcf()
    im = None
    field = runFunction()
    im = vectorFieldMagnitude(field, **kwargs)
    plotSetupFunction()

    def updatefig(*args):
        f = runFunction()
        f = norm(f, axis=2, ord=2)
        f = np.swapaxes(f, 0, 1)
        im.set_array(f)
        plotUpdateFunction()
        return im,

    return animation.FuncAnimation(fig, updatefig, interval=interval, frames=frames)
