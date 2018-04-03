from lbmpy.plot2d import *


def _draw_angles(ax, grouped_points):
    from matplotlib.lines import Line2D
    from matplotlib.text import Annotation
    from lbmpy.phasefield.postprocessing import get_angle

    x_data = [grouped_points[1][0], grouped_points[0][0], grouped_points[2][0]]
    y_data = [grouped_points[1][1], grouped_points[0][1], grouped_points[2][1]]

    v = [p - grouped_points[0] for p in grouped_points[1:]]
    v = [p / np.linalg.norm(p, ord=2) for p in v]
    direction = v[0] + v[1]

    if direction[1] > 1:
        ha = 'left'
    elif direction[1] < -1:
        ha = 'right'
    else:
        ha = 'center'

    if direction[0] > 1:
        va = 'bottom'
    elif direction[0] < -1:
        va = 'top'
    else:
        va = 'center'

    text_pos = grouped_points[0] + 10 * direction
    lines = Line2D(y_data, x_data, axes=ax, linewidth=3, color='k')
    ax.add_line(lines)

    angle = get_angle(*grouped_points)
    annotation = Annotation('%.1f°' % (angle,), (text_pos[1], text_pos[0]), axes=ax, fontsize=15, ha=ha, va=va)
    ax.add_artist(annotation)


def phase_indices(phi, **kwargs):
    single_phase_arrays = [phi[..., i] for i in range(phi.shape[-1])]
    last_phase = 1 - sum(single_phase_arrays)
    single_phase_arrays.append(last_phase)
    idx_array = np.array(single_phase_arrays).argmax(axis=0)
    return scalar_field(idx_array, **kwargs)


def angles(phi0, phi1, phi2, branching_distance=0.5, branching_point_filter=3, only_first=True):
    from lbmpy.phasefield.postprocessing import get_triple_point_info

    levels = [0.5, ]
    scalar_field_contour(phi0, levels=levels)
    scalar_field_contour(phi1, levels=levels)
    scalar_field_contour(phi2, levels=levels)

    if only_first:
        angle_info = get_triple_point_info(phi0, phi1, phi2, branching_distance, branching_point_filter)
    else:
        angle_info = get_triple_point_info(phi0, phi1, phi2, branching_distance, branching_point_filter) + \
                     get_triple_point_info(phi1, phi0, phi2, branching_distance, branching_point_filter) + \
                     get_triple_point_info(phi2, phi1, phi0, branching_distance, branching_point_filter)

    for points in angle_info:
        _draw_angles(gca(), points)
