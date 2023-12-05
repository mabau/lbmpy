import numpy as np
import sympy as sp

from pystencils.typing import TypedSymbol, create_type
from pystencils.backends.cbackend import CustomCodeNode


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
        return [TypedSymbol(f"neighbour_offset_{d}", create_type('int32')) for d in ['x', 'y', 'z'][:dim]]

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
        return TypedSymbol(f"{axis[mirror_axis]}_axis_mirrored_stencil_dir", create_type('int32'))

    def __init__(self, stencil, mirror_axis, dtype=np.int32):
        offsets_dtype = create_type(dtype)

        mirrored_stencil_symbol = MirroredStencilDirections._mirrored_symbol(mirror_axis)
        mirrored_directions = [stencil.index(MirroredStencilDirections.mirror_stencil(direction, mirror_axis))
                               for direction in stencil]
        code = "\n"
        code += _array_pattern(offsets_dtype, mirrored_stencil_symbol.name, mirrored_directions)

        super(MirroredStencilDirections, self).__init__(code, symbols_read=set(),
                                                        symbols_defined={mirrored_stencil_symbol})


class LbmWeightInfo(CustomCodeNode):
    def __init__(self, lb_method, data_type='double'):
        self.weights_symbol = TypedSymbol("weights", data_type)
        data_type_string = "double" if self.weights_symbol.dtype.numpy_dtype == np.float64 else "float"

        weights = [str(w.evalf(17)) for w in lb_method.weights]
        if data_type_string == "float":
            weights = "f, ".join(weights)
            weights += "f"  # suffix for the last element
        else:
            weights = ", ".join(weights)
        w_sym = self.weights_symbol
        code = f"const {data_type_string} {w_sym.name} [] = {{{ weights }}};\n"
        super(LbmWeightInfo, self).__init__(code, symbols_read=set(), symbols_defined={w_sym})

    def weight_of_direction(self, dir_idx, lb_method=None):
        if isinstance(sp.sympify(dir_idx), sp.Integer):
            return lb_method.weights[dir_idx].evalf(17)
        else:
            return sp.IndexedBase(self.weights_symbol, shape=(1,))[dir_idx]


class TranslationArraysNode(CustomCodeNode):

    def __init__(self, array_content, symbols_defined):
        code = ''
        for content in array_content:
            code += _array_pattern(*content)
        super(TranslationArraysNode, self).__init__(code, symbols_read=set(), symbols_defined=symbols_defined)

    def __str__(self):
        return "Variable PDF Access Translation Arrays"

    def __repr__(self):
        return "Variable PDF Access Translation Arrays"


def _array_pattern(dtype, name, content):
    return f"const {str(dtype)} {name} [] = {{ {','.join(str(c) for c in content)} }}; \n"
