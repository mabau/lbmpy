import numpy as np
from lbmpy.boundaries import NoSlip, UBB
from pystencils.slicing import normalize_slice, shift_slice, slice_intersection, slice_from_direction


def get_parabolic_initial_velocity(domain_size, u_max, vel_coord=0, diameter=None):
    if diameter is None:
        radius = int(round(min(sh for i, sh in enumerate(domain_size) if i != vel_coord) / 2))
    else:
        radius = int(round(diameter / 2))

    params = [np.arange(s) + 0.5 for s in domain_size]
    grid = np.meshgrid(*params, indexing='ij')

    dist = 0
    for i in range(len(domain_size)):
        if i == vel_coord:
            continue
        center = int(round(domain_size[i] / 2))
        dist += (grid[i] - center) ** 2
    dist = np.sqrt(dist)

    u = np.zeros(domain_size + [len(domain_size)])
    u[..., vel_coord] = u_max * (1 - (dist / radius) ** 2)
    return u


def add_box(boundary_handling, boundary=NoSlip(), replace=True):
    borders = ['N', 'S', 'E', 'W']
    if boundary_handling.dim == 3:
        borders += ['T', 'B']
    for d in borders:
        flag = boundary_handling.set_boundary(boundary, slice_from_direction(d, boundary_handling.dim), replace=replace)
    return flag


def add_parabolic_velocity_inflow(boundary_handling, u_max, index_expr, vel_coord=0, diameter=None):
    dim = boundary_handling.dim

    def velocity_info_callback(boundary_data):
        for i, name in enumerate(['vel_0', 'vel_1', 'vel_2'][:dim]):
            if i != vel_coord:
                boundary_data[name] = 0.0
        if diameter is None:
            radius = int(round(min(sh for i, sh in enumerate(boundary_handling.shape) if i != vel_coord) / 2))
        else:
            radius = int(round(diameter / 2))

        if dim == 3:
            normal_coord1 = (vel_coord + 1) % 3
            normal_coord2 = (vel_coord + 2) % 3
            y, z = boundary_data.link_positions(normal_coord1), boundary_data.link_positions(normal_coord2)
            centered_normal1 = y - int(round(boundary_handling.shape[normal_coord1] / 2))
            centered_normal2 = z - int(round(boundary_handling.shape[normal_coord2] / 2))
            dist_to_center = np.sqrt(centered_normal1 ** 2 + centered_normal2 ** 2)
        elif dim == 2:
            normal_coord = (vel_coord + 1) % 2
            centered_normal = boundary_data.link_positions(normal_coord) - radius
            dist_to_center = np.sqrt(centered_normal ** 2)
        else:
            raise ValueError("Invalid dimension")

        vel_profile = u_max * (1 - (dist_to_center / radius)**2)
        boundary_data['vel_%d' % (vel_coord,)] = vel_profile

    inflow = UBB(velocity_info_callback, dim=boundary_handling.dim)
    boundary_handling.set_boundary(inflow, slice_obj=index_expr, ghost_layers=True)


def setup_channel_walls(boundary_handling, diameter_callback, duct=False, wall_boundary=NoSlip()):
    dim = boundary_handling.dim
    directions = ('N', 'S', 'T', 'B') if dim == 3 else ('N', 'S')
    for direction in directions:
        boundary_handling.set_boundary(wall_boundary, slice_from_direction(direction, dim))

    if duct and diameter_callback is not None:
        raise ValueError("For duct flows, passing a diameter callback does not make sense.")

    if not duct:
        diameter = min(boundary_handling.shape[1:])
        add_pipe(boundary_handling, diameter_callback if diameter_callback else diameter, wall_boundary)


def add_pipe(boundary_handling, diameter, boundary=NoSlip()):
    """
    Sets boundary for the wall of a pipe with flow in x direction.

    :param boundary_handling: boundary handling object, works for serial and parallel versions
    :param diameter: pipe diameter, can be either a constant value or a callback function.
                     the callback function has the signature (x_coord_array, domain_shape_in_cells) and has to return
                     a array of same shape as the received x_coord_array, with the diameter for each x position
    :param boundary: boundary object that is set at the wall, defaults to NoSlip (bounce back)
    """
    domain_shape = boundary_handling.shape
    dim = len(domain_shape)
    assert dim in (2, 3)

    def callback(*coordinates):
        flow_coord = coordinates[0]

        if callable(diameter):
            flow_coord_line = flow_coord[:, 0, 0] if dim == 3 else flow_coord[:, 0]
            diameter_value = diameter(flow_coord_line, domain_shape)
            diameter_value = diameter_value[:, np.newaxis, np.newaxis] if dim == 3 else diameter_value[:, np.newaxis]
        else:
            diameter_value = diameter

        radius_sq = (diameter_value / 2) ** 2

        mid = [domain_shape[i] // 2 for i in range(1, dim)]
        distance = sum((c_i - mid_i) ** 2 for c_i, mid_i in zip(coordinates[1:], mid))
        return distance > radius_sq

    boundary_handling.set_boundary(boundary, mask_callback=callback)


def read_image(path, flatten=False):
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Image loading failed. Required package 'pillow' is missing")

    im = Image.open(path)
    if flatten:
        im = im.convert('F')
    return np.array(im)


def add_black_and_white_image(boundary_handling, image_file, target_slice=None, plane=(0, 1), boundary=NoSlip(),
                              keep_aspect_ratio=False):
    from scipy.ndimage import zoom

    domain_size = boundary_handling.shape
    if target_slice is None:
        target_slice = [slice(None, None, None)] * len(domain_size)

    dim = boundary_handling.dim

    image_slice = normalize_slice(target_slice, domain_size)
    target_size = [image_slice[i].stop - image_slice[i].start for i in plane]

    img_arr = read_image(image_file, flatten=True).astype(int)
    img_arr = np.rot90(img_arr, 3)

    zoom_factor = [target_size[i] / img_arr.shape[i] for i in range(2)]
    if keep_aspect_ratio:
        zoom_factor = min(zoom_factor)
    zoomed_image = zoom(img_arr, zoom_factor, order=0)

    # binarize
    zoomed_image[zoomed_image <= 254] = 0
    zoomed_image[zoomed_image > 254] = 1
    zoomed_image = np.logical_not(zoomed_image.astype(np.bool))

    # resize necessary if aspect ratio should be constant
    if zoomed_image.shape != target_size:
        resized_image = np.zeros(target_size, dtype=np.bool)
        mid = [(ts - s)//2 for ts, s in zip(target_size, zoomed_image.shape)]
        resized_image[mid[0]:zoomed_image.shape[0]+mid[0], mid[1]:zoomed_image.shape[1]+mid[1]] = zoomed_image
        zoomed_image = resized_image

    def callback(*coordinates):
        result = np.zeros_like(coordinates[0], dtype=np.bool)
        mask_start = [int(coordinates[i][(0,) * dim] - 0.5) for i in range(dim)]
        mask_end = [int(coordinates[i][(-1,) * dim] + 1 - 0.5) for i in range(dim)]

        mask_slice = [slice(start, stop) for start, stop in zip(mask_start, mask_end)]
        intersection_slice = slice_intersection(mask_slice, image_slice)
        if intersection_slice is None:
            return result
        else:
            mask_target_slice = shift_slice(intersection_slice, [-e for e in mask_start])
            image_target_slice = shift_slice(intersection_slice, [-s.start for s in image_slice])
            mask_target_slice = [mask_target_slice[i] if i in plane else slice(None, None) for i in range(dim)]
            image_target_slice = [image_target_slice[i] if i in plane else np.newaxis for i in range(dim)]
            result[mask_target_slice] = zoomed_image[image_target_slice]
            return result

    boundary_handling.set_boundary(boundary, slice_obj=image_slice, mask_callback=callback,
                                   ghost_layers=False, inner_ghost_layers=True)
