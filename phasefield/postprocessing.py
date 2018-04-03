import numpy as np
from matplotlib.path import Path
import itertools
import scipy
import scipy.spatial
import warnings


def get_isolines(dataset, level=0.5, refinement_factor=4):
    from matplotlib._contour import QuadContourGenerator
    index_arrays = np.meshgrid(*[np.arange(s) for s in dataset.shape])
    gen = QuadContourGenerator(*index_arrays, dataset, None, True, 0)
    result = gen.create_contour(level)
    if refinement_factor > 1:
        result = [Path(p).interpolated(refinement_factor).vertices for p in result]
    return result


def find_jump_indices(array, threshold=0, min_length=3):
    jumps = []
    offset = 0
    while True:
        if array[0] < threshold:
            jump = np.argmax(array > threshold)
        else:
            jump = np.argmax(array < threshold)
        if jump == 0:
            return jumps
        if len(array) <= min_length + jump:
            return jumps
        jumps.append(offset + jump)
        offset += jump + min_length

        array = array[jump + min_length:]


def find_branching_point(path_vertices1, path_vertices2, max_distance=0.5, branching_point_filter=3):
    tree = scipy.spatial.KDTree(path_vertices1)
    distances, indices = tree.query(path_vertices2, k=1, distance_upper_bound=max_distance)
    distances[distances == np.inf] = -1
    jump_indices = find_jump_indices(distances, 0, branching_point_filter)
    return path_vertices2[jump_indices]


def find_all_branching_points(phase_field1, phase_field2, max_distance=0.1, branching_point_filter=3):
    result = []
    iso_lines = [get_isolines(p, level=0.5, refinement_factor=16) for p in (phase_field1, phase_field2)]
    for path1, path2 in itertools.product(*iso_lines):
        bbs = find_branching_point(path1, path2, max_distance, branching_point_filter)
        result += list(bbs)
    return np.array(result)


def find_intersections(path_vertices1, path_vertices2):
    from numpy import where, dstack, diff, meshgrid

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # min, max and all for arrays
        amin = lambda x1, x2: where(x1 < x2, x1, x2)
        amax = lambda x1, x2: where(x1 > x2, x1, x2)
        aall = lambda abools: dstack(abools).all(axis=2)
        slope = lambda line: (lambda d: d[:, 1] / d[:, 0])(diff(line, axis=0))

        x11, x21 = meshgrid(path_vertices1[:-1, 0], path_vertices2[:-1, 0])
        x12, x22 = meshgrid(path_vertices1[1:, 0], path_vertices2[1:, 0])
        y11, y21 = meshgrid(path_vertices1[:-1, 1], path_vertices2[:-1, 1])
        y12, y22 = meshgrid(path_vertices1[1:, 1], path_vertices2[1:, 1])

        m1, m2 = meshgrid(slope(path_vertices1), slope(path_vertices2))
        m1inv, m2inv = 1 / m1, 1 / m2

        yi = (m1 * (x21 - x11 - m2inv * y21) + y11) / (1 - m1 * m2inv)
        xi = (yi - y21) * m2inv + x21

        xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
                  amin(x21, x22) < xi, xi <= amax(x21, x22))
        yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
                  amin(y21, y22) < yi, yi <= amax(y21, y22))

        return xi[aall(xconds)], yi[aall(yconds)]


def find_all_intersection_points(phase_field1, phase_field2):
    iso_lines = [get_isolines(p, level=1.0 / 3, refinement_factor=4)
                 for p in (phase_field1, phase_field2)]
    result = []
    for path1, path2 in itertools.product(*iso_lines):
        x_arr, y_arr = find_intersections(path1, path2)
        if x_arr is not None and y_arr is not None:
            for x, y in zip(x_arr, y_arr):
                result.append(np.array([x, y]))
    return np.array(result)


def group_points(triple_points, outer_points):
    """For each triple points the two closest point in 'outer_points' are searched.
    Returns list of tuples [ (triplePoints0, matchedPoint0, matchedPoint2), ... ]
    """
    assert len(outer_points) == 2 * len(triple_points)
    outer_points = list(outer_points)
    result = []
    for triplePoint in triple_points:
        outer_points.sort(key=lambda p: np.sum((triplePoint - p) ** 2))
        result.append([triplePoint, outer_points.pop(0), outer_points.pop(0)])
    return result


def get_angle(p_mid, p1, p2):
    """Returns angle in degree spanned by a midpoint and two outer points"""
    v = [p - p_mid for p in [p1, p2]]
    v = [p / np.linalg.norm(p) for p in v]
    scalar_prod = np.sum(v[0] * v[1])
    result = np.rad2deg(np.arccos(scalar_prod))
    return result


def get_triple_point_info(phi0, phi1, phi2, branching_distance=0.5, branching_point_filter=3):
    """

    :param branching_distance: where the 1/2 contour lines  move apart farther than this value, the
                              branching points are detected
    :return: list of 3-tuples that contain (triplePoint, branchingPoint1, branchingPoint2)
             the angle can be determined at the triple point 
    """
    # first triple points are searched where the contours lines of level 1/3 of two phases intersect
    # the angle at the triple points is measured with contour lines of level 1/2 at "branching points"
    # i.e. at points where the lines move away from each other

    bb1 = find_all_branching_points(phi0, phi1, branching_distance, branching_point_filter)
    bb2 = find_all_branching_points(phi0, phi2, branching_distance, branching_point_filter)
    ip = find_all_intersection_points(phi0, phi1)
    return group_points(ip, np.vstack([bb1, bb2]))


def get_angles_of_phase(phi0, phi1, phi2, branching_distance=0.5, branching_point_filter=3):
    return [get_angle(*r) for r in get_triple_point_info(phi0, phi1, phi2, branching_distance, branching_point_filter)]
