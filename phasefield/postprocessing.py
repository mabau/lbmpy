import numpy as np
from matplotlib.path import Path
import itertools
import scipy
import warnings


def getIsolines(dataset, level=0.5, refinementFactor=1):
    from matplotlib._contour import QuadContourGenerator
    indexArrays = np.meshgrid(*[np.arange(s) for s in dataset.shape])
    gen = QuadContourGenerator(*indexArrays, dataset, None, True, 0)
    result = gen.create_contour(level)
    if refinementFactor > 1:
        result = [Path(p).interpolated(refinementFactor).vertices for p in result]
    return result


def findJumpIndices(array, threshold=0, minLength=3):
    jumps = []
    offset = 0
    while True:
        if array[0] < threshold:
            jump = np.argmax(array > threshold)
        else:
            jump = np.argmax(array < threshold)
        if jump == 0:
            return jumps
        if len(array) <= minLength + jump:
            return jumps
        jumps.append(offset + jump)
        offset += jump + minLength

        array = array[jump + minLength:]


def findBranchingPoint(pathVertices1, pathVertices2, maxDistance=0.5):
    tree = scipy.spatial.KDTree(pathVertices1)
    distances, indices = tree.query(pathVertices2, k=1, distance_upper_bound=maxDistance)
    distances[distances == np.inf] = -1
    jumpIndices = findJumpIndices(distances, 0, 3)
    return pathVertices2[jumpIndices]


def findAllBranchingPoints(phaseField1, phaseField2, maxDistance=0.1):
    result = []
    isoLines = [getIsolines(p, level=0.5, refinementFactor=4) for p in (phaseField1, phaseField2)]
    for path1, path2 in itertools.product(*isoLines):
        bbs = findBranchingPoint(path1, path2, maxDistance)
        result += list(bbs)
    return np.array(result)


def findIntersections(pathVertices1, pathVertices2):
    from numpy import where, dstack, diff, meshgrid

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # min, max and all for arrays
        amin = lambda x1, x2: where(x1 < x2, x1, x2)
        amax = lambda x1, x2: where(x1 > x2, x1, x2)
        aall = lambda abools: dstack(abools).all(axis=2)
        slope = lambda line: (lambda d: d[:, 1] / d[:, 0])(diff(line, axis=0))

        x11, x21 = meshgrid(pathVertices1[:-1, 0], pathVertices2[:-1, 0])
        x12, x22 = meshgrid(pathVertices1[1:, 0], pathVertices2[1:, 0])
        y11, y21 = meshgrid(pathVertices1[:-1, 1], pathVertices2[:-1, 1])
        y12, y22 = meshgrid(pathVertices1[1:, 1], pathVertices2[1:, 1])

        m1, m2 = meshgrid(slope(pathVertices1), slope(pathVertices2))
        m1inv, m2inv = 1 / m1, 1 / m2

        yi = (m1 * (x21 - x11 - m2inv * y21) + y11) / (1 - m1 * m2inv)
        xi = (yi - y21) * m2inv + x21

        xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
                  amin(x21, x22) < xi, xi <= amax(x21, x22))
        yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
                  amin(y21, y22) < yi, yi <= amax(y21, y22))

        return xi[aall(xconds)], yi[aall(yconds)]


def findAllIntersectionPoints(phaseField1, phaseField2):
    isoLines = [getIsolines(p, level=1.0/3, refinementFactor=4)
                for p in (phaseField1, phaseField2)]
    result = []
    for path1, path2 in itertools.product(*isoLines):
        xArr, yArr = findIntersections(path1, path2)
        if xArr is not None and yArr is not None:
            for x, y in zip(xArr, yArr):
                result.append(np.array([x, y]))
    return np.array(result)


def groupPoints(triplePoints, outerPoints):
    """For each triple points the two closest point in 'outerPoints' are searched. 
    Returns list of tuples [ (triplePoints0, matchedPoint0, matchedPoint2), ... ]
    """
    assert len(outerPoints) == 2 * len(triplePoints)
    outerPoints = list(outerPoints)
    result = []
    for triplePoint in triplePoints:
        outerPoints.sort(key=lambda p: np.sum((triplePoint - p) ** 2))
        result.append([triplePoint, outerPoints.pop(0), outerPoints.pop(0)])
    return result


def getAngle(pMid, p1, p2):
    """Returns angle in degree spanned by a midpoint and two outer points"""
    v = [p - pMid for p in [p1, p2]]
    v = [p / np.linalg.norm(p) for p in v]
    scalarProd = np.sum(v[0] * v[1])
    result = np.rad2deg(np.arccos(scalarProd))
    return result


def getTriplePointInfos(phi0, phi1, phi2, branchingDistance=0.5):
    """

    :param branchingDistance: where the 1/2 contour lines  move apart farther than this value, the
                              branching points are detected
    :return: list of 3-tuples that contain (triplePoint, branchingPoint1, branchingPoint2)
             the angle can be determined at the triple point 
    """
    # first triple points are searched where the contours lines of level 1/3 of two phases intersect
    # the angle at the triple points is measured with contour lines of level 1/2 at "branching points"
    # i.e. at points where the lines move away from each other

    bb1 = findAllBranchingPoints(phi0, phi1, branchingDistance)
    bb2 = findAllBranchingPoints(phi0, phi2, branchingDistance)
    ip = findAllIntersectionPoints(phi0, phi1)
    return groupPoints(ip, np.vstack([bb1, bb2]))


