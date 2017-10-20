import numpy as np
from lbmpy.boundaries import NoSlip


def addPipe(boundaryHandling, diameter, boundary=NoSlip()):
    """
    Sets boundary for the wall of a pipe with flow in x direction.

    :param boundaryHandling: boundary handling object, works for serial and parallel versions 
    :param diameter: pipe diameter, can be either a constant value or a callback function.
                     the callback function has the signature (xCoordArray, domainShapeInCells) and has to return
                     a array of same shape as the received xCoordArray, with the diameter for each x position
    :param boundary: boundary object that is set at the wall, defaults to NoSlip (bounce back)
    """
    domainShape = boundaryHandling.domainShape
    dim = len(domainShape)
    assert dim in (2, 3)

    def callback(*coordinates):
        flowCoord = coordinates[0]

        if callable(diameter):
            flowCoordLine = flowCoord[:, 0, 0] if dim == 3 else flowCoord[:, 0]
            diameterValue = diameter(flowCoordLine, domainShape)
            diameterValue = diameterValue[:, np.newaxis, np.newaxis] if dim == 3 else diameterValue[:, np.newaxis]
        else:
            diameterValue = diameter

        radiusSq = (diameterValue / 2) ** 2

        mid = [domainShape[i] // 2 for i in range(1, dim)]
        distance = sum((c_i - mid_i) ** 2 for c_i, mid_i in zip(coordinates[1:], mid))
        return distance > radiusSq

    boundaryHandling.setBoundary(boundary, maskCallback=callback)
