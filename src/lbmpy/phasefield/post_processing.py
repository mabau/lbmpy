import warnings
from collections import namedtuple

import numpy as np

TriplePoint = namedtuple("TriplePoint", ['center', 'branch_points', 'angles'])


def get_angle(center, p1, p2):
    """Returns angle in degree spanned by a midpoint and two outer points"""
    v = [p - center for p in [p1, p2]]
    v = [p / np.linalg.norm(p) for p in v]
    scalar_prod = np.sum(v[0] * v[1])
    result = np.rad2deg(np.arccos(scalar_prod))
    return result


def find_intersections(path_vertices1, path_vertices2):
    from numpy import where, dstack, diff, meshgrid

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # min, max and all for arrays

        def arr_min(x1, x2):
            return where(x1 < x2, x1, x2)

        def arr_max(x1, x2):
            return where(x1 > x2, x1, x2)

        def arr_all(abools):
            return dstack(abools).all(axis=2)

        def slope(line):
            return (lambda d: d[:, 1] / d[:, 0])(diff(line, axis=0))

        x11, x21 = meshgrid(path_vertices1[:-1, 0], path_vertices2[:-1, 0])
        x12, x22 = meshgrid(path_vertices1[1:, 0], path_vertices2[1:, 0])
        y11, y21 = meshgrid(path_vertices1[:-1, 1], path_vertices2[:-1, 1])
        y12, y22 = meshgrid(path_vertices1[1:, 1], path_vertices2[1:, 1])

        m1, m2 = meshgrid(slope(path_vertices1), slope(path_vertices2))
        m2inv = 1 / m2

        yi = (m1 * (x21 - x11 - m2inv * y21) + y11) / (1 - m1 * m2inv)
        xi = (yi - y21) * m2inv + x21

        x_cond = (arr_min(x11, x12) < xi, xi <= arr_max(x11, x12),
                  arr_min(x21, x22) < xi, xi <= arr_max(x21, x22))
        y_cond = (arr_min(y11, y12) < yi, yi <= arr_max(y11, y12),
                  arr_min(y21, y22) < yi, yi <= arr_max(y21, y22))

        return xi[arr_all(x_cond)], yi[arr_all(y_cond)]


def get_line_intersections(line1, line2):
    """Returns all intersection points of two lines."""
    xs = []
    ys = []
    for c1 in line1:
        for c2 in line2:
            res = find_intersections(c1, c2)
            xs += list(res[0])
            ys += list(res[1])
    return np.array([xs, ys]).T


def find_closest_point(point, points):
    distances_sq = np.sum((points - point) ** 2, axis=1)
    return points[np.argmin(distances_sq)], np.sqrt(np.min(distances_sq))


def get_triple_point_centers(phase_arr, phase_indices, threshold=0.8):
    """Returns an array of triple points. """
    from skimage import measure

    assert len(phase_indices) == 3
    contour_lines = [measure.find_contours(phase_arr[..., phase_idx], 1 / 3)
                     for phase_idx in phase_indices]

    intersections = [get_line_intersections(contour_lines[a], contour_lines[b])
                     for a, b in [(0, 1), (1, 2), (0, 2)]]
    if min(len(i) for i in intersections) == 0:
        return np.array([[], []])

    triple_points = []
    for point in intersections[0]:
        closest_point1, dist1 = find_closest_point(point, intersections[1])
        closest_point2, dist2 = find_closest_point(point, intersections[2])
        if dist1 > threshold or dist2 > threshold:
            continue
        triple_points.append((point + closest_point1 + closest_point2) / 3)
    return np.array(triple_points)


def get_triple_points(phase_arr, phase_indices, contour_line_eps=0.01, threshold=0.8):
    """Returns information about triple points.

    Args:
        phase_arr: phase field with indices [x, y, phaseIdx]
        phase_indices: sequence of 3 indices, into the last phase_arr coordinate
                       triple points between these three phases are returned
        contour_line_eps: angles are computed between lines that go from the triple point to so called branching points
                     branching points are at locations where the (0.5-contour_line_eps) contour lines intersect
        threshold: triple points are searched by intersecting 1/3 contour lines
                   they should theoretically intersect in a single point
                   in practices these are 3 different points close to each other
                   if the points are further apart than threshold no triple points is generated
    Returns:
        Sequence of TriplePoint tuples
    """
    from skimage import measure

    triple_point_centers = get_triple_point_centers(phase_arr, phase_indices, threshold)
    if len(triple_point_centers) == 0:
        print("No triple point centers have been found")
        return []

    contour_lines = [measure.find_contours(phase_arr[..., phase_idx], 0.5 - contour_line_eps)
                     for phase_idx in phase_indices]

    p0p1 = get_line_intersections(contour_lines[0], contour_lines[1])
    p1p2 = get_line_intersections(contour_lines[1], contour_lines[2])
    p2p0 = get_line_intersections(contour_lines[2], contour_lines[0])

    result = []
    for center in triple_point_centers:
        branch_points = [find_closest_point(center, intersection)[0] for intersection in (p2p0, p0p1, p1p2)]
        angles = [get_angle(center, branch_points[i], branch_points[j])
                  for i, j in ((0, 1), (1, 2), (2, 0))]
        result.append(TriplePoint(center, branch_points, angles))
    return result


def analytic_neumann_angles(kappas):
    """Computes analytic Neumann angles using surface tension parameters.

    >>> analytic_neumann_angles([0.1, 0.1, 0.1])
    [120.00000000000001, 120.00000000000001, 120.00000000000001]
    >>> r = analytic_neumann_angles([0.1, 0.2, 0.3])
    >>> assert np.allclose(sum(r), 360)

    """
    def g(i, j):
        return kappas[i] + kappas[j]  # factor of α/6 can be omitted - only relative size is important

    neumann0 = np.arccos(- (g(0, 1) ** 2 + g(0, 2) ** 2 - g(1, 2) ** 2) / 2 / g(0, 1) / g(0, 2)) * 180 / np.pi
    neumann1 = np.arccos(- (g(0, 1) ** 2 + g(1, 2) ** 2 - g(0, 2) ** 2) / 2 / g(0, 1) / g(1, 2)) * 180 / np.pi
    neumann2 = np.arccos(- (g(1, 2) ** 2 + g(0, 2) ** 2 - g(0, 1) ** 2) / 2 / g(1, 2) / g(0, 2)) * 180 / np.pi
    return [neumann0, neumann1, neumann2]


def plot_triple_points(triple_points, line_length=8, axis=None):
    from matplotlib.text import Annotation
    import matplotlib.pyplot as plt

    if axis is None:
        axis = plt.gca()

    # Draw points
    for tp in triple_points:
        # Points
        for p in tp.branch_points + [tp.center]:
            axis.plot([p[0]], [p[1]], 'o', ms=5, color='k')

        line_ends = []
        for p in tp.branch_points:
            center_to_p = p - tp.center
            center_to_p = center_to_p / np.linalg.norm(center_to_p) * line_length
            line_ends.append(tp.center + center_to_p)

        # Lines
        for target in line_ends:
            axis.plot([tp.center[0], target[0]],
                      [tp.center[1], target[1]], '--', color='gray', lw=2)

        # Text
        for i in range(3):
            text_pos = (line_ends[i] + line_ends[(i + 1) % 3]) / 2

            annotation = Annotation('%.1f°' % (tp.angles[i],), (text_pos[0], text_pos[1]),
                                    axes=axis, fontsize=15, ha='center')
            axis.add_artist(annotation)


def plot_contour_lines(phase_arr, axis=None, **kwargs):
    import matplotlib.pyplot as plt
    from skimage import measure

    if axis is None:
        axis = plt.gca()

    for threshold in (0.5,):
        contours = measure.find_contours(phase_arr[:, :, 0], threshold)
        for n, contour in enumerate(contours):
            axis.plot(contour[:, 0], contour[:, 1], linewidth=2, **kwargs)

        contours = measure.find_contours(phase_arr[:, :, 1], threshold)
        for n, contour in enumerate(contours):
            axis.plot(contour[:, 0], contour[:, 1], linewidth=2, **kwargs)

        contours = measure.find_contours(phase_arr[:, :, 2].copy(), threshold)
        for n, contour in enumerate(contours):
            axis.plot(contour[:, 0], contour[:, 1], linewidth=2, **kwargs)

        axis.set_aspect('equal', 'box')
