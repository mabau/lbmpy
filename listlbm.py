import numpy as np


def get_fluid_cell_array(flag_arr, fluid_flag, ghost_layers):
    inner_slice = [slice(ghost_layers, -ghost_layers)] * len(flag_arr.shape)
    flag_arr_no_gl = flag_arr[inner_slice]
    return np.argwhere(np.bitwise_and(flag_arr_no_gl, fluid_flag)) + ghost_layers


def get_cell_index(fluid_cell_arr, cell):
    assert len(fluid_cell_arr.shape) == len(cell)
    first_coord = [slice(None, None)] + [0] * (len(fluid_cell_arr.shape) - 1)
    begin = np.searchsorted(fluid_cell_arr[first_coord], cell[0], 'left')
    end = np.searchsorted(fluid_cell_arr[first_coord], cell[0], 'right')
    if begin == end:
        raise ValueError("Element not found")

    if len(fluid_cell_arr.shape) == 1:
        return begin
    else:
        sub_array = fluid_cell_arr[begin:end, 1]
        return begin + get_cell_index(sub_array, cell[1:])


def get_index_array(fluid_cell_arr, flag_arr, ghost_layers, stencil, fluid_flag, noslip_flag):

    def pdf_index(cell_index, direction_index):
        return cell_index + direction_index * len(fluid_cell_arr)

    def inverse_idx(idx):
        return stencil.index(tuple(-d_i for d_i in stencil[idx]))

    result = []
    ctr = 0
    for direction_idx, direction in enumerate(stencil):
        for own_cell_idx, cell in enumerate(fluid_cell_arr):
            inv_neighbor_cell = np.array([cell_i - dir_i for cell_i, dir_i in zip(cell, direction)])

            if flag_arr[tuple(inv_neighbor_cell)] & fluid_flag:
                neighbor_cell_idx = get_cell_index(fluid_cell_arr, inv_neighbor_cell)
                result.append(pdf_index(neighbor_cell_idx, direction_idx))
            elif flag_arr[tuple(inv_neighbor_cell)] & noslip_flag:  # no-slip before periodicity!
                result.append(pdf_index(own_cell_idx, inverse_idx(direction_idx)))
            else:
                # periodicity handling
                # print(inv_neighbor_cell, end="")
                at_border = False
                for i, x_i in enumerate(inv_neighbor_cell):
                    if x_i == (ghost_layers - 1):
                        inv_neighbor_cell[i] += flag_arr.shape[i] - (2 * ghost_layers)
                        at_border = True
                    elif x_i == flag_arr.shape[i] - ghost_layers:
                        inv_neighbor_cell[i] -= flag_arr.shape[i] - (2 * ghost_layers)
                        at_border = True
                if at_border:
                    assert flag_arr[tuple(inv_neighbor_cell)] & fluid_flag
                    neighbor_cell_idx = get_cell_index(fluid_cell_arr, inv_neighbor_cell)
                    result.append(pdf_index(neighbor_cell_idx, direction_idx))
                else:
                    raise ValueError("Could not find neighbor for {} direction {}".format(cell, direction))

            ctr += 1  # TODO
    return np.array(result, dtype=np.uint32)


def plot_index_array(fluid_cell_arr, stencil, ghost_layers=1, index_arr=None, **kwargs):
    """Visualizes index array.

    Args:
        fluid_cell_arr: array of fluid cells
        stencil: stencil
        ghost_layers: number of ghost layers
        index_arr: index array, or None to show pdf index
        **kwargs: passed to LbGrid.add_arrow

    Returns:
        LbGrid
    """
    from lbmpy.plot2d import LbGrid
    x_max, y_max = np.max(fluid_cell_arr, axis=0)
    grid = LbGrid(x_max + 1 - ghost_layers, y_max + 1 - ghost_layers)
    index = 0
    for direction in stencil:
        for fluid_cell in fluid_cell_arr:
            annotation = index_arr[index] if index_arr is not None else index
            grid.add_arrow((fluid_cell[0] - ghost_layers, fluid_cell[1] - ghost_layers),
                           direction, direction, str(annotation), **kwargs)
            index += 1
    return grid
