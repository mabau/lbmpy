from collections import namedtuple

import numpy as np


def circle_intersections(midpoint0, midpoint1, radius0, radius1):
    """Returns intersection points of two circles."""
    midpoint_distance = np.linalg.norm(midpoint0 - midpoint1)
    x0, y0 = midpoint0
    x1, y1 = midpoint1

    k1 = (y0 - y1) / (x0 - x1)  # slope for the line linking two centers
    b1 = y1 - k1 * x1           # line equation of the line linking two centers

    k2 = -1 / k1  # slope for the line linking two cross points
    b2 = (radius0 ** 2 - radius1 ** 2 - x0 ** 2 + x1 ** 2 - y0 ** 2 + y1 ** 2) / (2 * (y1 - y0))

    if midpoint_distance > (radius0 + radius1):  # no intersection
        return []

    if np.abs(radius1 - radius0) == midpoint_distance or midpoint_distance == radius0 + radius1:
        xx = -(b1 - b2) / (k1 - k2)
        yy = -(-b2 * k1 + b1 * k2) / (k1 - k2)
        return [(xx, yy)]
    elif np.abs(radius1 - radius0) < midpoint_distance < radius0 + radius1:
        xx1 = (-b2 * k2 + x1 + k2 * y1 - np.sqrt(-b2 ** 2 + radius1 ** 2 + k2 ** 2 * radius1 ** 2 - 2 * b2
               * k2 * x1 - k2 ** 2 * x1 ** 2 + 2 * b2 * y1 + 2 * k2 * x1 * y1 - y1 ** 2)) / (1 + k2 ** 2)
        yy1 = k2 * xx1 + b2

        xx2 = (-b2 * k2 + x1 + k2 * y1 + np.sqrt(-b2 ** 2 + radius1 ** 2 + k2 ** 2 * radius1 ** 2 - 2 * b2
               * k2 * x1 - k2 ** 2 * x1 ** 2 + 2 * b2 * y1 + 2 * k2 * x1 * y1 - y1 ** 2)) / (1 + k2 ** 2)
        yy2 = k2 * xx2 + b2

        return [(xx1, yy1), (xx2, yy2)]
    else:
        return []


def interface_region(concentration_arr, phase0, phase1, area=3):
    import scipy.ndimage as sc_image

    area_phase0 = sc_image.morphology.binary_dilation(concentration_arr[..., phase0] > 0.5, iterations=area)
    area_phase1 = sc_image.morphology.binary_dilation(concentration_arr[..., phase1] > 0.5, iterations=area)
    return np.logical_and(area_phase0, area_phase1)


def contour_line(concentration_arr, phase0, phase1, cutoff=0.05):
    from skimage import measure
    concentration_arr = concentration_arr.copy()

    mask = interface_region(concentration_arr, phase0, phase1)
    concentration_arr = concentration_arr[..., phase0]
    concentration_arr[np.logical_not(mask)] = np.nan
    contours = measure.find_contours(concentration_arr, 0.5)

    contours = [c for c in contours if len(c) > 8]

    if len(contours) != 1:
        raise ValueError("Multiple or non contour lines (%d) found" % (len(contours),))

    contour = contours[0]
    absolute_cutoff = int(len(contour) * cutoff) + 1
    return contour[absolute_cutoff:-absolute_cutoff]


def fit_circle(points):
    # see https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    from scipy import optimize

    def point_distances(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def f(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        ri = point_distances(*c)
        return ri - ri.mean()

    center_estimate = np.mean(points, axis=0)
    center, _ = optimize.leastsq(f, center_estimate)
    radius = point_distances(*center).mean()

    circle = namedtuple('Circle', ['center', 'radius'])
    return circle(center, radius)


def neumann_angles_from_surface_tensions(surface_tensions):
    g = surface_tensions
    angles = [
        np.arccos(-(g(0, 1) * g(0, 1) + g(0, 2) * g(0, 2) - g(1, 2) * g(1, 2)) / 2 / g(0, 1) / g(0, 2)),
        np.arccos(-(g(0, 1) * g(0, 1) + g(1, 2) * g(1, 2) - g(0, 2) * g(0, 2)) / 2 / g(0, 1) / g(1, 2)),
        np.arccos(-(g(1, 2) * g(1, 2) + g(0, 2) * g(0, 2) - g(0, 1) * g(0, 1)) / 2 / g(1, 2) / g(0, 2)),
    ]
    return [np.rad2deg(a) for a in angles]


def surface_tension_from_kappas(kappas, surface_width):
    def surface_tensions(i, j):
        if i == j:
            return 0
        return (kappas[i] + kappas[j]) / 6 * surface_width
    return surface_tensions


def liquid_lens_neumann_angles(concentration, drop_phase_idx=2, enclosing_phase1=0, enclosing_phase2=1):
    """Assumes a liquid lens setup, where a drop is enclosed between two other phases in the middle of the domain.

    Args:
        concentration: array with concentrations
        drop_phase_idx: index of the drop phase
        enclosing_phase1: index of the first enclosing phase
        enclosing_phase2: index of the second enclosing phase

    Returns:
        tuple with three angles
        (angle enclosing phase1, angle enclosing phase2, angle droplet)
    """
    circles = [fit_circle(contour_line(concentration, enclosing_phase1, drop_phase_idx)),
               fit_circle(contour_line(concentration, enclosing_phase2, drop_phase_idx))]

    intersections = circle_intersections(circles[0].center, circles[1].center, circles[0].radius, circles[1].radius)

    y1, y2 = (i[1] for i in intersections)
    assert np.abs(y1 - y2) < 0.1, "Liquid lens is not aligned in y direction (rotated?)"

    left_intersection = sorted(intersections, key=lambda e: e[0])[0]
    lower_circle, upper_circle = sorted(circles, key=lambda c: c.center[1])

    def rotate90_ccw(vector):
        return np.array([-vector[1], vector[0]])

    def rotate90_cw(vector):
        return np.array([vector[1], -vector[0]])

    def normalize(vector):
        return vector / np.linalg.norm(vector)

    vector_20 = normalize(rotate90_ccw(lower_circle.center - left_intersection))
    vector_01 = np.array([-1, 0])
    vector_12 = normalize(rotate90_cw(upper_circle.center - left_intersection))

    angles = [np.rad2deg(np.arccos(np.dot(v1, v2))) for v1, v2 in [(vector_20, vector_01),
                                                                   (vector_01, vector_12),
                                                                   (vector_12, vector_20)]
              ]
    return angles
