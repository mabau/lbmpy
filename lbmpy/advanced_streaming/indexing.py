import numpy as np
import sympy as sp
import pystencils as ps

from pystencils.typing import TypedSymbol, create_type
from pystencils.backends.cbackend import CustomCodeNode

from lbmpy.advanced_streaming.utility import get_accessor, inverse_dir_index, is_inplace, Timestep

from itertools import product


def _array_pattern(dtype, name, content):
    return f"const {str(dtype)} {name} [] = {{ {','.join(str(c) for c in content)} }}; \n"


class BetweenTimestepsIndexing:

    #   ==============================================
    #       Symbols for usage in kernel definitions
    #   ==============================================

    @property
    def proxy_fields(self):
        return ps.fields(f"f_out({self._q}), f_in({self._q}): [{self._dim}D]")

    @property
    def dir_symbol(self):
        return TypedSymbol('dir', create_type(self._index_dtype))

    @property
    def inverse_dir_symbol(self):
        """Symbol denoting the inversion of a PDF field index. 
        Use only at top-level of index to f_out or f_in, otherwise it can't be correctly replaced."""
        return sp.IndexedBase('invdir')

    #   =============================
    #       Constructor and State
    #   =============================

    def __init__(self, pdf_field, stencil, prev_timestep=Timestep.BOTH, streaming_pattern='pull',
                 index_dtype=np.int32, offsets_dtype=np.int32):
        if prev_timestep == Timestep.BOTH and is_inplace(streaming_pattern):
            raise ValueError('Cannot create index arrays for both kinds of timesteps for inplace streaming pattern '
                             + streaming_pattern)

        prev_accessor = get_accessor(streaming_pattern, prev_timestep)
        next_accessor = get_accessor(streaming_pattern, prev_timestep.next())

        outward_accesses = prev_accessor.write(pdf_field, stencil)
        inward_accesses = next_accessor.read(pdf_field, stencil)

        self._accesses = {'out': outward_accesses, 'in': inward_accesses}

        self._pdf_field = pdf_field
        self._stencil = stencil
        self._dim = stencil.D
        self._q = stencil.Q
        self._coordinate_names = ['x', 'y', 'z'][:self._dim]

        self._index_dtype = create_type(index_dtype)
        self._offsets_dtype = create_type(offsets_dtype)

        self._required_index_arrays = set()
        self._required_offset_arrays = set()
        self._trivial_index_translations, self._trivial_offset_translations = self._collect_trivial_translations()

    def _index_array_symbol(self, f_dir, inverse):
        assert f_dir in ['in', 'out']
        inv = '_inv' if inverse else ''
        name = f"f_{f_dir}{inv}_dir_idx"
        return TypedSymbol(name, self._index_dtype)

    def _offset_array_symbols(self, f_dir, inverse):
        assert f_dir in ['in', 'out']
        inv = '_inv' if inverse else ''
        name_base = f"f_{f_dir}{inv}_offsets_"
        symbols = [TypedSymbol(name_base + d, self._index_dtype) for d in self._coordinate_names]
        return symbols

    def _array_symbols(self, f_dir, inverse, index):
        if (f_dir, inverse) in self._trivial_index_translations:
            translated_index = index
        else:
            index_array_symbol = self._index_array_symbol(f_dir, inverse)
            translated_index = sp.IndexedBase(index_array_symbol, shape=(1,))[index]
            self._required_index_arrays.add((f_dir, inverse))

        if (f_dir, inverse) in self._trivial_offset_translations:
            offsets = (0, ) * self._dim
        else:
            offset_array_symbols = self._offset_array_symbols(f_dir, inverse)
            offsets = tuple(sp.IndexedBase(s, shape=(1,))[index] for s in offset_array_symbols)
            self._required_offset_arrays.add((f_dir, inverse))

        return {'index': translated_index, 'offsets': offsets}

    #   =================================
    #       Proxy fields substitution
    #   =================================

    def substitute_proxies(self, assignments):
        if isinstance(assignments, ps.Assignment):
            assignments = [assignments]

        if not isinstance(assignments, ps.AssignmentCollection):
            assignments = ps.AssignmentCollection(assignments)

        accesses = self._accesses
        f_out, f_in = self.proxy_fields
        inv_dir = self.inverse_dir_symbol

        accessor_subs = dict()

        for fa in assignments.atoms(ps.Field.Access):
            if fa.field == f_out:
                f_dir = 'out'
            elif fa.field == f_in:
                f_dir = 'in'
            else:
                continue

            inv = False
            idx = fa.index[0]
            if isinstance(idx, sp.Indexed) and idx.base == inv_dir:
                idx = idx.indices[0]
                if isinstance(sp.sympify(idx), sp.Integer):
                    idx = inverse_dir_index(self._stencil, idx)
                inv = True

            if isinstance(sp.sympify(idx), sp.Integer):
                accessor_subs[fa] = accesses[f_dir][idx].get_shifted(*fa.offsets)
            else:
                arr = self._array_symbols(f_dir, inv, idx)
                accessor_subs[fa] = self._pdf_field[arr['offsets']](arr['index']).get_shifted(*fa.offsets)

        return assignments.new_with_substitutions(accessor_subs)

    #   =================
    #       Internals
    #   =================

    def _get_translated_indices_and_offsets(self, f_dir, inv):
        accesses = self._accesses[f_dir]

        if inv:
            inverse_indices = [inverse_dir_index(self._stencil, i)
                               for i in range(len(self._stencil))]
            accesses = [accesses[idx] for idx in inverse_indices]

        indices = [a.index[0] for a in accesses]
        offsets = []
        for d in range(self._dim):
            offsets.append([a.offsets[d] for a in accesses])
        return indices, offsets

    def _collect_trivial_translations(self):
        trivial_index_translations = set()
        trivial_offset_translations = set()
        trivial_indices = list(range(self._q))
        trivial_offsets = [[0] * self._q] * self._dim
        for f_dir, inv in product(['in', 'out'], [False, True]):
            indices, offsets = self._get_translated_indices_and_offsets(f_dir, inv)
            if indices == trivial_indices:
                trivial_index_translations.add((f_dir, inv))
            if offsets == trivial_offsets:
                trivial_offset_translations.add((f_dir, inv))
        return trivial_index_translations, trivial_offset_translations

    def create_code_node(self):
        return BetweenTimestepsIndexing.TranslationArraysNode(self)

    class TranslationArraysNode(CustomCodeNode):

        def __init__(self, indexing):
            code = ''
            symbols_defined = set()

            for f_dir, inv in indexing._required_index_arrays:
                indices, offsets = indexing._get_translated_indices_and_offsets(f_dir, inv)
                index_array_symbol = indexing._index_array_symbol(f_dir, inv)
                symbols_defined.add(index_array_symbol)
                code += _array_pattern(indexing._index_dtype, index_array_symbol.name, indices)

            for f_dir, inv in indexing._required_offset_arrays:
                indices, offsets = indexing._get_translated_indices_and_offsets(f_dir, inv)
                offset_array_symbols = indexing._offset_array_symbols(f_dir, inv)
                symbols_defined |= set(offset_array_symbols)
                for d, arrsymb in enumerate(offset_array_symbols):
                    code += _array_pattern(indexing._offsets_dtype, arrsymb.name, offsets[d])

            super(BetweenTimestepsIndexing.TranslationArraysNode, self).__init__(
                code, symbols_read=set(), symbols_defined=symbols_defined)

        def __str__(self):
            return "Variable PDF Access Translation Arrays"

        def __repr__(self):
            return "Variable PDF Access Translation Arrays"

#   end class AdvancedStreamingIndexing


class NeighbourOffsetArrays(CustomCodeNode):

    @staticmethod
    def neighbour_offset(dir_idx, stencil):
        if isinstance(sp.sympify(dir_idx), sp.Integer):
            return stencil[dir_idx]
        else:
            return tuple([sp.IndexedBase(symbol, shape=(1,))[dir_idx]
                         for symbol in NeighbourOffsetArrays._offset_symbols(len(stencil[0]))])

    @staticmethod
    def _offset_symbols(dim):
        return [TypedSymbol(f"neighbour_offset_{d}", create_type(np.int32)) for d in ['x', 'y', 'z'][:dim]]

    def __init__(self, stencil, offsets_dtype=np.int32):
        offsets_dtype = create_type(offsets_dtype)
        dim = len(stencil[0])

        array_symbols = NeighbourOffsetArrays._offset_symbols(dim)
        code = "\n"
        for i, arrsymb in enumerate(array_symbols):
            code += _array_pattern(offsets_dtype, arrsymb.name, (d[i] for d in stencil))

        offset_symbols = NeighbourOffsetArrays._offset_symbols(dim)
        super(NeighbourOffsetArrays, self).__init__(code, symbols_read=set(),
                                                    symbols_defined=set(offset_symbols))


class MirroredStencilDirections(CustomCodeNode):

    @staticmethod
    def mirror_stencil(direction, mirror_axis):
        assert mirror_axis <= len(direction), f"only {len(direction)} axis available for mirage"
        direction = list(direction)
        direction[mirror_axis] = -direction[mirror_axis]

        return tuple(direction)

    @staticmethod
    def _mirrored_symbol(mirror_axis):
        axis = ['x', 'y', 'z']
        return TypedSymbol(f"{axis[mirror_axis]}_axis_mirrored_stencil_dir", create_type(np.int32))

    def __init__(self, stencil, mirror_axis, dtype=np.int32):
        offsets_dtype = create_type(dtype)

        mirrored_stencil_symbol = MirroredStencilDirections._mirrored_symbol(mirror_axis)
        mirrored_directions = [stencil.index(MirroredStencilDirections.mirror_stencil(direction, mirror_axis))
                               for direction in stencil]
        code = "\n"
        code += _array_pattern(offsets_dtype, mirrored_stencil_symbol.name, mirrored_directions)

        super(MirroredStencilDirections, self).__init__(code, symbols_read=set(),
                                                        symbols_defined={mirrored_stencil_symbol})
