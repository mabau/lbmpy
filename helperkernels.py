import numpy as np


def initializePdfField(latticeModel, pdfArray):
    if latticeModel.compressible is None:
        pdfArray.fill(0.0)
    else:
        if latticeModel.dim == 2:
            for i, weight in enumerate(latticeModel.weights):
                pdfArray[:, :, i] = float(weight)
        elif latticeModel.dim == 3:
            for i, weight in enumerate(latticeModel.weights):
                pdfArray[:, :, :, i] = float(weight)
        else:
            raise NotImplementedError()


def computeVelocity(latticeModel, pdfArray):
    vel = np.zeros(pdfArray.shape[:-1] + (2,))
    for i, dir in enumerate(latticeModel.stencil):
        if dir[0] != 0:
            vel[:, :, 0] += dir[0] * pdfArray[:, :, i]
        if dir[1] != 0:
            vel[:, :, 1] += dir[1] * pdfArray[:, :, i]
    return vel
