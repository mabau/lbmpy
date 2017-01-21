import numpy as np
import itertools

#try:
if True:
    import pyximport;
    pyximport.install()
    from lbmpy.boundaries.createindexlistcython import createBoundaryIndexList2D, createBoundaryIndexList3D
    cythonFuncsAvailable = True
#except Exception:
#    cythonFuncsAvailable = False
#    createBoundaryIndexList2D = None
#    createBoundaryIndexList3D = None


def _createBoundaryIndexListPython(flagFieldArr, nrOfGhostLayers, boundaryMask, fluidMask, stencil):
    result = []
    gl = nrOfGhostLayers
    for cell in itertools.product(*[range(gl, i-gl) for i in flagFieldArr.shape]):
        if not flagFieldArr[cell] & fluidMask:
            continue
        for dirIdx, direction in enumerate(stencil):
            neighborCell = tuple([cell_i + dir_i for cell_i, dir_i in zip(cell, direction)])
            if flagFieldArr[neighborCell] & boundaryMask:
                result.append(list(cell) + [dirIdx])

    return np.array(result, dtype=np.int32)


def createBoundaryIndexList(flagField, stencil, boundaryMask, fluidMask, nrOfGhostLayers=1):
    if cythonFuncsAvailable:
        stencil = np.array(stencil, dtype=np.int32)
        if len(flagField.shape) == 2:
            return np.array(createBoundaryIndexList2D(flagField, nrOfGhostLayers, boundaryMask, fluidMask, stencil))
        elif len(flagField.shape) == 3:
            return np.array(createBoundaryIndexList3D(flagField, nrOfGhostLayers, boundaryMask, fluidMask, stencil))
        else:
            raise ValueError("Flag field has to be a 2 or 3 dimensional numpy array")
    else:
        return _createBoundaryIndexListPython(flagField, nrOfGhostLayers, boundaryMask, fluidMask, stencil)
