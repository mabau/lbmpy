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
        xx1 = (-b2 * k2 + x1 + k2 * y1 - np.sqrt(-b2 ** 2 + radius1 ** 2 + k2 ** 2 * radius1 ** 2 - 2 * b2 *
               k2 * x1 - k2 ** 2 * x1 ** 2 + 2 * b2 * y1 + 2 * k2 * x1 * y1 - y1 ** 2)) / (1 + k2 ** 2)
        yy1 = k2 * xx1 + b2

        xx2 = (-b2 * k2 + x1 + k2 * y1 + np.sqrt(-b2 ** 2 + radius1 ** 2 + k2 ** 2 * radius1 ** 2 - 2 * b2 *
               k2 * x1 - k2 ** 2 * x1 ** 2 + 2 * b2 * y1 + 2 * k2 * x1 * y1 - y1 ** 2)) / (1 + k2 ** 2)
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
        return np.sqrt((points[:, 0]-xc)**2 + (points[:, 1]-yc)**2)

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


def liquid_lens_neumann_angles(concentration, drop_phase_idx=2, enclosing_phase1=0, enclosing_phase2=1):
    circles = [fit_circle(contour_line(concentration, enclosing_phase1, drop_phase_idx)),
               fit_circle(contour_line(concentration, enclosing_phase2, drop_phase_idx))]

    intersections = circle_intersections(circles[0].center, circles[1].center, circles[0].radius, circles[1].radius)

    y1, y2 = (i[1] for i in intersections)
    assert np.abs(y1 - y2) < 0.1, "Liquid lens is not aligned in y direction (rotated?)"
    y_horizontal = (y1 + y2) / 2

    xs = np.sqrt(circles[0].radius ** 2 - (y_horizontal - circles[0].center[1]) ** 2)
    angle = np.rad2deg(np.arctan(-xs / np.sqrt(circles[0].radius ** 2 - xs ** 2))) % 180
    angle = 180 - angle
    neumann3_upper = angle
    neumann1 = 180 - neumann3_upper

    xs = np.sqrt(circles[1].radius ** 2 - (y_horizontal - circles[1].center[1]) ** 2)
    angle = np.rad2deg(np.arctan(-xs / np.sqrt(circles[1].radius ** 2 - xs ** 2))) % 180
    angle = 180 - angle
    neumann3_lower = angle
    neumann2 = 180 - neumann3_lower

    neumann3 = neumann3_upper + neumann3_lower
    return neumann1, neumann2, neumann3
