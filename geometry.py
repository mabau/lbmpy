import numpy as np
from lbmpy.boundaries import NoSlip, UBB
from pystencils.slicing import normalizeSlice, shiftSlice, sliceIntersection


def addParabolicVelocityInflow(boundaryHandling, u_max, indexExpr, velCoord=0, diameter=None):
    def velocityInfoCallback(boundaryData):
        for i, name in enumerate(['vel_0', 'vel_1', 'vel_2']):
            if i != velCoord:
                boundaryData[name] = 0
        if diameter is None:
            radius = min(sh for i, sh in enumerate(boundaryHandling.domainShape) if i != velCoord) // 2
        else:
            radius = diameter // 2
        print("radius", radius)
        y, z = boundaryData.linkPositions(1), boundaryData.linkPositions(2)
        centeredY = y - radius
        centeredZ = z - radius
        distToCenter = np.sqrt(centeredY ** 2 + centeredZ ** 2)
        boundaryData['vel_%d' % (velCoord,)] = u_max * (1 - distToCenter / radius)

    inflow = UBB(velocityInfoCallback, dim=boundaryHandling.dim)
    boundaryHandling.setBoundary(inflow, indexExpr=indexExpr, includeGhostLayers=False)


def addPipe(boundaryHandling, diameter, boundary=NoSlip()):
    """
    Sets boundary for the wall of a pipe with flow in x direction.

    :param boundaryHandling: boundary handling object, works for serial and parallel versions 
    :param diameter: pipe diameter, can be either a constant value or a callback function.
                     the callback function has the signature (xCoordArray, domainShapeInCells) andp has to return
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


def addBlackAndWhiteImage(boundaryHandling, imageFile, targetSlice=None, plane=(0, 1), boundary=NoSlip(),
                          keepAspectRatio=False):
    try:
        from scipy.misc import imread
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError("scipy image read could not be imported! Install 'scipy' and 'pillow'")

    domainSize = boundaryHandling.domainShape
    if targetSlice is None:
        targetSlice = [slice(None, None, None)] * len(domainSize)

    dim = len(boundaryHandling.domainShape)

    imageSlice = normalizeSlice(targetSlice, domainSize)
    targetSize = [imageSlice[i].stop - imageSlice[i].start for i in plane]

    imgArr = imread(imageFile, flatten=True).astype(int)
    imgArr = np.rot90(imgArr, 3)

    zoomFactor = [targetSize[i] / imgArr.shape[i] for i in range(2)]
    if keepAspectRatio:
        zoomFactor = min(zoomFactor)
    zoomedImage = zoom(imgArr, zoomFactor, order=0)

    # binarize
    zoomedImage[zoomedImage <= 254] = 0
    zoomedImage[zoomedImage > 254] = 1
    zoomedImage = np.logical_not(zoomedImage.astype(np.bool))

    # resize necessary if aspect ratio should be constant
    if zoomedImage.shape != targetSize:
        resizedImage = np.zeros(targetSize, dtype=np.bool)
        mid = [(ts - s)//2 for ts, s in zip(targetSize, zoomedImage.shape)]
        resizedImage[mid[0]:zoomedImage.shape[0]+mid[0], mid[1]:zoomedImage.shape[1]+mid[1]] = zoomedImage
        zoomedImage = resizedImage

    def callback(*coordinates):
        result = np.zeros_like(coordinates[0], dtype=np.bool)
        maskStart = [int(coordinates[i][(0,) * dim] - 0.5) for i in range(dim)]
        maskEnd = [int(coordinates[i][(-1,) * dim] + 1 - 0.5) for i in range(dim)]

        maskSlice = [slice(start, stop) for start, stop in zip(maskStart, maskEnd)]
        intersectionSlice = sliceIntersection(maskSlice, imageSlice)
        if intersectionSlice is None:
            return result
        else:
            maskTargetSlice = shiftSlice(intersectionSlice, [-e for e in maskStart])
            imageTargetSlice = shiftSlice(intersectionSlice, [-s.start for s in imageSlice])
            maskTargetSlice = [maskTargetSlice[i] if i in plane else slice(None, None) for i in range(dim)]
            imageTargetSlice = [imageTargetSlice[i] if i in plane else np.newaxis for i in range(dim)]
            result[maskTargetSlice] = zoomedImage[imageTargetSlice]
            return result

    boundaryHandling.setBoundary(boundary, indexExpr=imageSlice, maskCallback=callback, includeGhostLayers=False)

