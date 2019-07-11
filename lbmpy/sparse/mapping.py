from typing import Tuple

import numpy as np
import sympy as sp

from lbmpy.boundaries.boundaryhandling import LbmWeightInfo
from pystencils import Assignment, Field, TypedSymbol
from pystencils.boundaries.boundaryhandling import BoundaryOffsetInfo
from pystencils.boundaries.createindexlist import (
    boundary_index_array_coordinate_names, create_boundary_index_list, direction_member_name)


class SparseLbMapper:
    """Manages the mapping of cell coordinates to indices and back.

    Args:
          flag_arr: integer array where each bit corresponds to a boundary or 'fluid'
    """
    def __init__(self, stencil, flag_arr, fluid_flag, no_slip_flag, other_boundary_mask):
        self._flag_arr = flag_arr
        self._coordinate_arr = None
        self._sorter = None  # array of indices that sort _coordinate_arr
        self._dirty = True
        self.fluid_flag = fluid_flag
        self.no_slip_flag = no_slip_flag
        self.other_boundary_mask = other_boundary_mask
        self._num_fluid_cells = None
        self.stencil = stencil

    @property
    def coordinates(self):
        if self._dirty:
            self._assemble()

        return self._coordinate_arr

    @property
    def fluid_coordinates(self):
        if self._dirty:
            self._assemble()
        return self._coordinate_arr[:self.num_fluid_cells]

    @property
    def num_fluid_cells(self):
        return self._num_fluid_cells

    @property
    def flag_array(self):
        return self._flag_arr

    def cell_idx(self, coordinate: Tuple[int, ...]) -> np.uint32:
        """Maps from coordinates (x,y,z) or (x,y) tuple to the list index. Raises ValueError if coordinate not found."""
        if self._dirty:
            self._assemble()
        coordinate = np.array(coordinate, dtype=self._coordinate_arr.dtype)
        left = np.searchsorted(self._coordinate_arr, coordinate, sorter=self._sorter, side='left')
        right = np.searchsorted(self._coordinate_arr, coordinate, sorter=self._sorter, side='right')
        if left + 1 != right:
            raise IndexError("Coordinate not found")
        else:
            return self._sorter[left]

    def cell_idx_bulk(self, coordinates):
        coordinates = coordinates.astype(self._coordinate_arr.dtype)
        left = np.searchsorted(self._coordinate_arr, coordinates, sorter=self._sorter, side='left')
        right = np.searchsorted(self._coordinate_arr, coordinates, sorter=self._sorter, side='right')
        left[left + 1 != right] = -1
        return self._sorter[left]

    def coordinate(self, cell_idx: int) -> Tuple[np.uint32, ...]:
        """Maps from a cell index to its coordinate.

        Args:
            cell_idx: index of the cell in list - any integer between 0 and len(mapping)
        Returns:
            (x, y, [z]) coordinate tuple
        """
        if self._dirty:
            self._assemble()
        return self._coordinate_arr[cell_idx]

    def __len__(self):
        if self._dirty:
            self._assemble()
        return len(self._coordinate_arr)

    def _assemble(self):
        dim = len(self._flag_arr.shape)
        struct_type = np.dtype([(name, np.uint32) for name in boundary_index_array_coordinate_names[:dim]])

        # Add fluid cells
        coordinates_fluid = np.argwhere(np.bitwise_and(self._flag_arr, self.fluid_flag)).astype(np.uint32)
        coordinates_boundary = np.argwhere(np.bitwise_and(self._flag_arr, self.other_boundary_mask)).astype(np.uint32)
        self._num_fluid_cells = coordinates_fluid.shape[0]

        total_cells = len(coordinates_fluid) + len(coordinates_boundary)

        self._coordinate_arr = np.empty((total_cells,), dtype=struct_type)
        for d, d_name in zip(range(dim), ['x', 'y', 'z']):
            self._coordinate_arr[d_name][:self._num_fluid_cells] = coordinates_fluid[:, d]
            self._coordinate_arr[d_name][self._num_fluid_cells:] = coordinates_boundary[:, d]

        self._sorter = np.argsort(self._coordinate_arr).astype(np.uint32)
        self._dirty = False

    def create_index_array(self, ghost_layers=1):
        # TODO support different layouts here
        stencil = self.stencil

        def pdf_index(cell_index, direction_index):
            return cell_index + direction_index * len(self)

        def inverse_idx(idx):
            return stencil.index(tuple(-d_i for d_i in stencil[idx]))

        flag_arr = self.flag_array
        no_slip_flag = self.no_slip_flag
        fluid_boundary_mask = self.other_boundary_mask | self.fluid_flag

        result = []
        for direction_idx, direction in enumerate(stencil):
            if all(d_i == 0 for d_i in direction):
                assert direction_idx == 0
                continue
            for own_cell_idx, cell in enumerate(self.fluid_coordinates):
                inv_neighbor_cell = np.array([cell_i - dir_i for cell_i, dir_i in zip(cell, direction)])
                if flag_arr[tuple(inv_neighbor_cell)] & fluid_boundary_mask:
                    neighbor_cell_idx = self.cell_idx(tuple(inv_neighbor_cell))
                    result.append(pdf_index(neighbor_cell_idx, direction_idx))
                elif flag_arr[tuple(inv_neighbor_cell)] & no_slip_flag:  # no-slip before periodicity!
                    result.append(pdf_index(own_cell_idx, inverse_idx(direction_idx)))
                else:
                    at_border = False
                    for i, x_i in enumerate(inv_neighbor_cell):
                        if x_i == (ghost_layers - 1):
                            inv_neighbor_cell[i] += flag_arr.shape[i] - (2 * ghost_layers)
                            at_border = True
                        elif x_i == flag_arr.shape[i] - ghost_layers:
                            inv_neighbor_cell[i] -= flag_arr.shape[i] - (2 * ghost_layers)
                            at_border = True
                    if at_border:
                        assert flag_arr[tuple(inv_neighbor_cell)] & fluid_boundary_mask
                        neighbor_cell_idx = self.cell_idx(tuple(inv_neighbor_cell))
                        result.append(pdf_index(neighbor_cell_idx, direction_idx))
                    else:
                        raise ValueError("Could not find neighbor for {} direction {}".format(cell, direction))

        index_array = np.array(result, dtype=np.uint32)
        index_arr = index_array.reshape([len(stencil) - 1, self.num_fluid_cells])
        index_arr = index_arr.swapaxes(0, 1)
        return index_arr


class SparseLbBoundaryMapper:
    NEIGHBOR_IDX_NAME = 'nidx{}'
    DIR_SYMBOL = TypedSymbol("dir", np.uint32)

    def __init__(self, boundary, method, pdf_field_sparse):
        full_pdf_field = Field.create_generic('pdfFull', spatial_dimensions=method.dim, index_dimensions=1)

        additional_data_field = Field.create_generic('additionalData', spatial_dimensions=1,
                                                     dtype=boundary.additional_data)
        boundary_eqs = boundary(full_pdf_field, self.DIR_SYMBOL, method, additional_data_field)
        neighbor_offsets = {fa.offsets for eq in boundary_eqs for fa in eq.atoms(Field.Access)}
        neighbor_offsets = list(neighbor_offsets)

        neighbor_offsets_dtype = [(self.NEIGHBOR_IDX_NAME.format(i), np.uint32)
                                  for i in range(len(neighbor_offsets))]

        index_field_dtype = np.dtype([('dir', np.uint32),
                                      *neighbor_offsets_dtype,
                                      *boundary.additional_data])
        index_field = Field.create_generic('indexField', spatial_dimensions=1, dtype=index_field_dtype)
        boundary_eqs = boundary(full_pdf_field, self.DIR_SYMBOL, method, index_field)

        offset_subs = {off: sp.Symbol(self.NEIGHBOR_IDX_NAME.format(i)) for i, off in enumerate(neighbor_offsets)}

        new_boundary_eqs = []
        for eq in boundary_eqs:
            substitutions = {
                fa: pdf_field_sparse.absolute_access([index_field(offset_subs[fa.offsets].name)], fa.index)
                for fa in eq.atoms(Field.Access)
                if fa.field == full_pdf_field
            }
            new_boundary_eqs.append(eq.subs(substitutions))

        self.boundary_eqs = new_boundary_eqs
        self.boundary_eqs_orig = boundary_eqs
        self.method = method
        self.index_field_dtype = index_field_dtype
        self.neighbor_offsets = neighbor_offsets
        self.index_field = index_field

    def _build_substitutions(self):
        dim = self.method.dim
        stencil = self.method.stencil

        result = [{BoundaryOffsetInfo.offset_from_dir(self.DIR_SYMBOL, dim)[current_dim]: offset[current_dim]
                   for current_dim in range(dim)}
                  for i, offset in enumerate(self.method.stencil)
                  ]
        for dir_idx, subs_dict in enumerate(result):
            inv_idx = stencil.index(tuple(-e for e in stencil[dir_idx]))
            subs_dict[BoundaryOffsetInfo.inv_dir(self.DIR_SYMBOL)] = inv_idx
        return result

    def create_index_arr(self, mapping: SparseLbMapper, boundary_mask, nr_of_ghost_layers=1):
        stencil = self.method.stencil
        flag_dtype = mapping.flag_array.dtype.type
        idx_arr = create_boundary_index_list(mapping.flag_array, stencil,
                                             flag_dtype(boundary_mask), flag_dtype(mapping.fluid_flag),
                                             nr_of_ghost_layers)

        result = np.empty(idx_arr.shape, dtype=self.index_field_dtype)

        dim = self.method.dim

        coord_names = boundary_index_array_coordinate_names[:dim]
        center_coordinates = idx_arr[coord_names]

        substitutions = self._build_substitutions()
        for j, neighbor_offset in enumerate(self.neighbor_offsets):
            neighbor_coordinates = center_coordinates.copy()
            offsets = np.array([tuple(int(sp.sympify(e).subs(substitution)) for e in neighbor_offset)
                                for substitution in substitutions])
            for i, coord_name in enumerate(coord_names):
                neighbor_coordinates[coord_name] += offsets[:, i][idx_arr['dir']]
            result[self.NEIGHBOR_IDX_NAME.format(j)] = mapping.cell_idx_bulk(neighbor_coordinates)

        result[direction_member_name] = idx_arr[direction_member_name]
        # Find neighbor indices
        return result

    def assignments(self):
        return [BoundaryOffsetInfo(self.method.stencil),
                LbmWeightInfo(self.method),
                Assignment(self.DIR_SYMBOL, self.index_field(self.DIR_SYMBOL.name)),
                *self.boundary_eqs]
