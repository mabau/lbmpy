import numpy as np
import pystencils.boundaries.createindexlist as cil
from lbmpy.stencils import get_stencil


def test_equivalence_cython_python_version():
    if not cil.cython_funcs_available:
        return

    stencil_2d = get_stencil("D2Q9")
    stencil_3d = get_stencil("D3Q19")

    for dtype in [int, np.int16, np.uint32]:
        fluid_mask = dtype(1)
        mask = dtype(2)
        flag_field_2d = np.ones([15, 16], dtype=dtype) * fluid_mask
        flag_field_3d = np.ones([15, 16, 17], dtype=dtype) * fluid_mask

        flag_field_2d[0, :] = mask
        flag_field_2d[-1, :] = mask
        flag_field_2d[7, 7] = mask

        flag_field_3d[0, :, :] = mask
        flag_field_3d[-1, :, :] = mask
        flag_field_3d[7, 7, 7] = mask

        result_python_2d = cil._create_boundary_neighbor_index_list_python(flag_field_2d, 1, mask, fluid_mask,
                                                                           stencil_2d, False)

        result_python_3d = cil._create_boundary_neighbor_index_list_python(flag_field_3d, 1, mask, fluid_mask,
                                                                           stencil_3d, False)

        result_cython_2d = cil.create_boundary_index_list(flag_field_2d, stencil_2d, mask,
                                                          fluid_mask, 1, True, False)
        result_cython_3d = cil.create_boundary_index_list(flag_field_3d, stencil_3d, mask,
                                                          fluid_mask, 1, True, False)

        np.testing.assert_equal(result_python_2d, result_cython_2d)
        np.testing.assert_equal(result_python_3d, result_cython_3d)


def test_equivalence_cell_idx_list_cython_python_version():
    if not cil.cython_funcs_available:
        return

    stencil_2d = get_stencil("D2Q9")
    stencil_3d = get_stencil("D3Q19")

    for dtype in [int, np.int16, np.uint32]:
        fluid_mask = dtype(1)
        mask = dtype(2)
        flag_field_2d = np.ones([15, 16], dtype=dtype) * fluid_mask
        flag_field_3d = np.ones([15, 16, 17], dtype=dtype) * fluid_mask

        flag_field_2d[0, :] = mask
        flag_field_2d[-1, :] = mask
        flag_field_2d[7, 7] = mask

        flag_field_3d[0, :, :] = mask
        flag_field_3d[-1, :, :] = mask
        flag_field_3d[7, 7, 7] = mask

        result_python_2d = cil._create_boundary_cell_index_list_python(flag_field_2d, mask, fluid_mask,
                                                                       stencil_2d, False)

        result_python_3d = cil._create_boundary_cell_index_list_python(flag_field_3d, mask, fluid_mask,
                                                                       stencil_3d, False)

        result_cython_2d = cil.create_boundary_index_list(flag_field_2d, stencil_2d, mask, fluid_mask, None,
                                                          False, False)
        result_cython_3d = cil.create_boundary_index_list(flag_field_3d, stencil_3d, mask, fluid_mask, None,
                                                          False, False)

        np.testing.assert_equal(result_python_2d, result_cython_2d)
        np.testing.assert_equal(result_python_3d, result_cython_3d)
