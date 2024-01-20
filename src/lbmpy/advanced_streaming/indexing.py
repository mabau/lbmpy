import numpy as np
import sympy as sp
import pystencils as ps

from pystencils.typing import TypedSymbol, create_type
from lbmpy.advanced_streaming.utility import get_accessor, inverse_dir_index, is_inplace, Timestep
from lbmpy.custom_code_nodes import TranslationArraysNode

from itertools import product


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
        array_content = list()
        symbols_defined = set()
        for f_dir, inv in self._required_index_arrays:
            indices, offsets = self._get_translated_indices_and_offsets(f_dir, inv)
            index_array_symbol = self._index_array_symbol(f_dir, inv)
            symbols_defined.add(index_array_symbol)
            array_content.append((self._index_dtype, index_array_symbol.name, indices))

        for f_dir, inv in self._required_offset_arrays:
            indices, offsets = self._get_translated_indices_and_offsets(f_dir, inv)
            offset_array_symbols = self._offset_array_symbols(f_dir, inv)
            symbols_defined |= set(offset_array_symbols)
            for d, arrsymb in enumerate(offset_array_symbols):
                array_content.append((self._offsets_dtype, arrsymb.name, offsets[d]))

        return TranslationArraysNode(array_content, symbols_defined)

#   end class AdvancedStreamingIndexing
