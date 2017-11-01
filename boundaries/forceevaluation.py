import numpy as np
from lbmpy.stencils import inverseDirection


def calculateForceOnNoSlipBoundary(boundaryObject, boundaryHandling, pdfArray):
    indArr = boundaryHandling.getBoundaryIndexArray(boundaryObject)
    method = boundaryHandling.lbMethod

    stencil = np.array(method.stencil)

    if method.dim == 2:
        x, y = indArr['x'], indArr['y']
        values = 2 * pdfArray[x, y,  indArr['dir']]
    else:
        assert method.dim == 3
        x, y, z = indArr['x'], indArr['y'], indArr['z']
        values = 2 * pdfArray[x, y, z, indArr['dir']]

    forces = stencil[indArr['dir']] * values[:, np.newaxis]
    return forces.sum(axis=0)


def calculateForceOnBoundary(boundaryObject, boundaryHandling, pdfArray):
    indArr = boundaryHandling.getBoundaryIndexArray(boundaryObject)
    method = boundaryHandling.lbMethod

    stencil = np.array(method.stencil)
    invDirection = np.array([method.stencil.index(inverseDirection(d))
                            for d in method.stencil])

    if method.dim == 2:
        x, y = indArr['x'], indArr['y']
        offsets = stencil[indArr['dir']]

        fluidValues = pdfArray[x, y,  indArr['dir']]
        boundaryValues = pdfArray[x + offsets[:, 0],
                                  y + offsets[:, 1],
                                  invDirection[indArr['dir']]]
    else:
        assert method.dim == 3
        x, y, z = indArr['x'], indArr['y'], indArr['z']
        offsets = stencil[indArr['dir']]

        fluidValues = pdfArray[x, y, z, indArr['dir']]
        boundaryValues = pdfArray[x + offsets[:, 0],
                                  y + offsets[:, 1],
                                  z + offsets[:, 2],
                                  invDirection[indArr['dir']]]

    values = fluidValues + boundaryValues
    forces = stencil[indArr['dir']] * values[:, np.newaxis]
    return forces.sum(axis=0)
