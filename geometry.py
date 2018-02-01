import numpy as np
from lbmpy.boundaries import NoSlip, UBB
from pystencils.slicing import normalizeSlice, shiftSlice, sliceIntersection, sliceFromDirection


def getParabolicInitialVelocity(domainSize, u_max, velCoord=0, diameter=None):
    if diameter is None:
        radius = int(round(min(sh for i, sh in enumerate(domainSize) if i != velCoord) / 2))
    else:
        radius = int(round(diameter / 2))

    params = [np.arange(s) + 0.5 for s in domainSize]
    grid = np.meshgrid(*params, indexing='ij')

    dist = 0
    for i in range(len(domainSize)):
        if i == velCoord:
            continue
        center = int(round(domainSize[i] / 2))
        dist += (grid[i] - center) ** 2
    dist = np.sqrt(dist)

    u = np.zeros(domainSize + [len(domainSize)])
    u[..., velCoord] = u_max * (1 - (dist / radius) ** 2)
    return u


def addBox(boundaryHandling, boundary=NoSlip()):
    borders = ['N', 'S', 'E', 'W']
    if boundaryHandling.dim == 3:
        borders += ['T', 'B']
    for d in borders:
        boundaryHandling.setBoundary(boundary, sliceFromDirection(d, boundaryHandling.dim))


def addParabolicVelocityInflow(boundaryHandling, u_max, indexExpr, velCoord=0, diameter=None):
    dim = boundaryHandling.dim

    def velocityInfoCallback(boundaryData):
        for i, name in enumerate(['vel_0', 'vel_1', 'vel_2'][:dim]):
            if i != velCoord:
                boundaryData[name] = 0.0
        if diameter is None:
            radius = int(round(min(sh for i, sh in enumerate(boundaryHandling.shape) if i != velCoord) / 2))
        else:
            radius = int(round(diameter / 2))

        if dim == 3:
            normalCoord1 = (velCoord + 1) % 3
            normalCoord2 = (velCoord + 2) % 3
            y, z = boundaryData.linkPositions(normalCoord1), boundaryData.linkPositions(normalCoord2)
            centeredNormal1 = y - int(round(boundaryHandling.shape[normalCoord1] / 2))
            centeredNormal2 = z - int(round(boundaryHandling.shape[normalCoord2] / 2))
            distToCenter = np.sqrt(centeredNormal1 ** 2 + centeredNormal2 ** 2)
        elif dim == 2:
            normalCoord = (velCoord + 1) % 2
            centeredNormal = boundaryData.linkPositions(normalCoord) - radius
            distToCenter = np.sqrt(centeredNormal ** 2)
        else:
            raise ValueError("Invalid dimension")

        velProfile = u_max * (1 - (distToCenter / radius)**2)
        boundaryData['vel_%d' % (velCoord,)] = velProfile

    inflow = UBB(velocityInfoCallback, dim=boundaryHandling.dim)
    boundaryHandling.setBoundary(inflow, sliceObj=indexExpr, ghostLayers=True)


def setupChannelWalls(boundaryHandling, diameterCallback, duct=False, wallBoundary=NoSlip()):
    dim = boundaryHandling.dim
    directions = ('N', 'S', 'T', 'B') if dim == 3 else ('N', 'S')
    for direction in directions:
        boundaryHandling.setBoundary(wallBoundary, sliceFromDirection(direction, dim))

    if duct and diameterCallback is not None:
        raise ValueError("For duct flows, passing a diameter callback does not make sense.")

    if not duct:
        diameter = min(boundaryHandling.shape[1:])
        addPipe(boundaryHandling, diameterCallback if diameterCallback else diameter, wallBoundary)


def addPipe(boundaryHandling, diameter, boundary=NoSlip()):
    """
    Sets boundary for the wall of a pipe with flow in x direction.

    :param boundaryHandling: boundary handling object, works for serial and parallel versions 
    :param diameter: pipe diameter, can be either a constant value or a callback function.
                     the callback function has the signature (xCoordArray, domainShapeInCells) andp has to return
                     a array of same shape as the received xCoordArray, with the diameter for each x position
    :param boundary: boundary object that is set at the wall, defaults to NoSlip (bounce back)
    """
    domainShape = boundaryHandling.shape
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


def readImage(path, flatten=False):
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Image loading failed. Required package 'pillow' is missing")

    im = Image.open(path)
    if flatten:
        im = im.convert('F')
    return np.array(im)


def addBlackAndWhiteImage(boundaryHandling, imageFile, targetSlice=None, plane=(0, 1), boundary=NoSlip(),
                          keepAspectRatio=False):
    """
    
    :param boundaryHandling: 
    :param imageFile: 
    :param targetSlice: 
    :param plane: 
    :param boundary: 
    :param keepAspectRatio: 
    :return: 
    """
    from scipy.ndimage import zoom

    domainSize = boundaryHandling.shape
    if targetSlice is None:
        targetSlice = [slice(None, None, None)] * len(domainSize)

    dim = boundaryHandling.dim

    imageSlice = normalizeSlice(targetSlice, domainSize)
    targetSize = [imageSlice[i].stop - imageSlice[i].start for i in plane]

    imgArr = readImage(imageFile, flatten=True).astype(int)
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

    boundaryHandling.setBoundary(boundary, sliceObj=imageSlice, maskCallback=callback,
                                 ghostLayers=False, innerGhostLayers=True)
