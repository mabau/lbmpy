import numpy as np

from lbmpy.boundaries import UBB, NoSlip
from pystencils.slicing import (
    normalize_slice, shift_slice, slice_from_direction, slice_intersection)


def add_box_boundary(boundary_handling, boundary=NoSlip(), replace=True):
    """ Adds wall boundary conditions at all domain boundaries.

    Args:
        boundary_handling: boundary handling object
        boundary: the boundary to set at domain boundaries
        replace: see BoundaryHandling.set_boundary , True overwrites flag field, False only adds the boundary flag

    Returns:
        flag used for the wall boundary
    """
    borders = ['N', 'S', 'E', 'W']
    if boundary_handling.dim == 3:
        borders += ['T', 'B']
    flag = None
    for d in borders:
        flag = boundary_handling.set_boundary(boundary, slice_from_direction(d, boundary_handling.dim), replace=replace)
    assert flag is not None
    return flag


def add_sphere(boundary_handling, midpoint, radius, boundary=NoSlip(), replace=True):
    """Sets boundary in spherical region."""
    def set_sphere(x, y):
        return (x - midpoint[0]) ** 2 + (y - midpoint[1]) ** 2 < radius ** 2
    return boundary_handling.set_boundary(boundary, mask_callback=set_sphere, replace=replace)


def add_pipe_inflow_boundary(boundary_handling, u_max, slice_obj, flow_direction=0, diameter=None):
    """Adds velocity inflow UBB boundary for pipe flow.

    Args:
        boundary_handling: boundary handling object
        u_max: maximum velocity at center of the pipe
        slice_obj: slice describing where the boundary should be set
        flow_direction: dimension index, e.g. 0 for a channel that flows along x,
        diameter: pipe/diameter, if None is passed the maximum diameter is assumed

    Returns:
        flag used for the inflow boundary

    Examples:
        >>> from lbmpy.lbstep import LatticeBoltzmannStep
        >>> from pystencils import make_slice
        >>> pipe = LatticeBoltzmannStep(domain_size=(20, 10, 10), relaxation_rate=1.8, periodicity=(True, False, False))
        >>> flag = add_pipe_inflow_boundary(pipe.boundary_handling, u_max=0.05,
        ...                                 slice_obj=make_slice[0, :, :], flow_direction=0)
    """
    dim = boundary_handling.dim

    def velocity_info_callback(boundary_data):
        for i, name in enumerate(['vel_0', 'vel_1', 'vel_2'][:dim]):
            if i != flow_direction:
                boundary_data[name] = 0.0
        if diameter is None:
            radius = int(round(min(sh for i, sh in enumerate(boundary_handling.shape) if i != flow_direction) / 2))
        else:
            radius = int(round(diameter / 2))

        if dim == 3:
            normal_coord1 = (flow_direction + 1) % 3
            normal_coord2 = (flow_direction + 2) % 3
            y, z = boundary_data.link_positions(normal_coord1), boundary_data.link_positions(normal_coord2)
            centered_normal1 = y - int(round(boundary_handling.shape[normal_coord1] / 2))
            centered_normal2 = z - int(round(boundary_handling.shape[normal_coord2] / 2))
            dist_to_center = np.sqrt(centered_normal1 ** 2 + centered_normal2 ** 2)
        elif dim == 2:
            normal_coord = (flow_direction + 1) % 2
            centered_normal = boundary_data.link_positions(normal_coord) - radius
            dist_to_center = np.sqrt(centered_normal ** 2)
        else:
            raise ValueError("Invalid dimension")

        vel_profile = u_max * (1 - (dist_to_center / radius)**2)
        boundary_data['vel_%d' % (flow_direction,)] = vel_profile

    inflow = UBB(velocity_info_callback, dim=boundary_handling.dim)
    return boundary_handling.set_boundary(inflow, slice_obj=slice_obj, ghost_layers=True)


def add_pipe_walls(boundary_handling, diameter, boundary=NoSlip()):
    """Sets boundary for the wall of a pipe with flow in x direction.

    Args:
        boundary_handling: boundary handling object, works for serial and parallel versions
        diameter: pipe diameter, can be either a constant value or a callback function.
                  the callback function has the signature (x_coord_array, domain_shape_in_cells) and has to return
                  a array of same shape as the received x_coord_array, with the diameter for each x position
        boundary: boundary object that is set at the wall, defaults to NoSlip (bounce back)

    Returns:
        flag used for the wall boundary
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

    return boundary_handling.set_boundary(boundary, mask_callback=callback)


def get_pipe_velocity_field(domain_size, u_max, flow_direction=0, diameter=None):
    """Analytic velocity field for 2D or 3D pipe flow.

    Args:
        domain_size: 2-tuple or 3-tuple: shape of the domain
        u_max: velocity in the middle of the pipe
        flow_direction: dimension index, e.g. 0 for a channel that flows along x,
        diameter: pipe/diameter, if None is passed the maximum diameter is assumed

    Returns:
        numpy array with velocity field, velocity values outside the pipe are invalid

    Examples:
        Set up channel flow with analytic solution
        >>> from lbmpy.scenarios import create_channel
        >>> domain_size = (10, 10, 5)
        >>> u_max = 0.05
        >>> initial_velocity = get_pipe_velocity_field(domain_size, u_max)
        >>> scenario = create_channel(domain_size=domain_size, u_max=u_max, relaxation_rate=1.8,
        ...                           initial_velocity=initial_velocity)
    """
    if diameter is None:
        radius = int(round(min(sh for i, sh in enumerate(domain_size) if i != flow_direction) / 2))
    else:
        radius = int(round(diameter / 2))

    params = [np.arange(s) + 0.5 for s in domain_size]
    grid = np.meshgrid(*params, indexing='ij')

    dist = 0
    for i in range(len(domain_size)):
        if i == flow_direction:
            continue
        center = int(round(domain_size[i] / 2))
        dist += (grid[i] - center) ** 2
    dist = np.sqrt(dist)

    u = np.zeros(tuple(domain_size) + (len(domain_size),))
    u[..., flow_direction] = u_max * (1 - (dist / radius) ** 2)
    return u


def get_shear_flow_velocity_field(domain_size, u_max, random_y_factor=0.01):
    """Returns a velocity field with two shear layers.

    -----------------------------
    | ->  ->  ->  ->  ->  ->  ->| Upper layer moves to the right
    | ->  ->  ->  ->  ->  ->  ->|
    | <-  <-  <-  <-  <-  <-  <-| Middle layer to the left
    | <-  <-  <-  <-  <-  <-  <-|
    | ->  ->  ->  ->  ->  ->  ->| Lower layer to the right again
    | ->  ->  ->  ->  ->  ->  ->|
    -----------------------------

    Args:
        domain_size: size of the domain, tuple with (x,y) or (x,y,z)
        u_max: magnitude of x component of the velocity
        random_y_factor: to break the symmetry the y velocity component is initialized randomly
                         its maximum value is u_max * random_y_factor
    """
    height = domain_size[2] if len(domain_size) == 3 else domain_size[1]
    velocity = np.empty(tuple(domain_size) + (len(domain_size),))
    velocity[..., 0] = u_max
    velocity[..., height // 3: height // 3 * 2, 0] = -u_max
    for j in range(1, len(domain_size)):
        velocity[..., j] = random_y_factor * u_max * (np.random.rand(*domain_size) - 0.5)
    return velocity


def add_black_and_white_image(boundary_handling, image_file, target_slice=None, plane=(0, 1), boundary=NoSlip(),
                              keep_aspect_ratio=False):
    """Sets boundary from a black and white image, setting boundary where image is black.

    For 3D domains the image is extruded along a coordinate. Requires scipy for image resizing.

    Args:
        boundary_handling: boundary handling object
        image_file: path to image file
        target_slice: rectangular sub-region where the image should be set, or None for everywhere
        plane: 2-tuple with coordinate indices, (0, 1) means that the image is mapped into the x-y plane and
               extruded in z direction
        boundary: boundary object to set, where the image has black pixels
        keep_aspect_ratio: by default the image is reshaped to match the target_slice. If this parameter is True the
                           aspect ratio is kept, effectively making the target_slice smaller

    Returns:
        numpy array with velocity field, velocity values outside the pipe are invalid
    """
    from scipy.ndimage import zoom

    domain_size = boundary_handling.shape
    if target_slice is None:
        target_slice = [slice(None, None, None)] * len(domain_size)

    dim = boundary_handling.dim

    image_slice = normalize_slice(target_slice, domain_size)
    target_size = [image_slice[i].stop - image_slice[i].start for i in plane]

    img_arr = _read_image(image_file, flatten=True).astype(int)
    img_arr = np.rot90(img_arr, 3)

    zoom_factor = [target_size[i] / img_arr.shape[i] for i in range(2)]
    if keep_aspect_ratio:
        zoom_factor = min(zoom_factor)
    zoomed_image = zoom(img_arr, zoom_factor, order=0)

    # binarize
    zoomed_image[zoomed_image <= 254] = 0
    zoomed_image[zoomed_image > 254] = 1
    zoomed_image = np.logical_not(zoomed_image.astype(bool))

    # resize necessary if aspect ratio should be constant
    if zoomed_image.shape != target_size:
        resized_image = np.zeros(target_size, dtype=bool)
        mid = [(ts - s) // 2 for ts, s in zip(target_size, zoomed_image.shape)]
        resized_image[mid[0]:zoomed_image.shape[0] + mid[0], mid[1]:zoomed_image.shape[1] + mid[1]] = zoomed_image
        zoomed_image = resized_image

    def callback(*coordinates):
        result = np.zeros_like(coordinates[0], dtype=bool)
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
            result[tuple(mask_target_slice)] = zoomed_image[tuple(image_target_slice)]
            return result

    return boundary_handling.set_boundary(boundary, slice_obj=image_slice, mask_callback=callback,
                                          ghost_layers=False, inner_ghost_layers=True)


def _read_image(path, flatten=False):
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Image loading failed. Required package 'pillow' is missing")

    im = Image.open(path)
    if flatten:
        im = im.convert('F')
    return np.array(im)
