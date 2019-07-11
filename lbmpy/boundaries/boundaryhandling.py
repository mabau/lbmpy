import numpy as np
import sympy as sp

from pystencils import Assignment, TypedSymbol, create_indexed_kernel
from pystencils.backends.cbackend import CustomCodeNode
from pystencils.boundaries import BoundaryHandling
from pystencils.boundaries.boundaryhandling import BoundaryOffsetInfo
from pystencils.stencil import inverse_direction


class LatticeBoltzmannBoundaryHandling(BoundaryHandling):

    def __init__(self, lb_method, data_handling, pdf_field_name, name="boundary_handling", flag_interface=None,
                 target='cpu', openmp=True):
        self.lb_method = lb_method
        super(LatticeBoltzmannBoundaryHandling, self).__init__(data_handling, pdf_field_name, lb_method.stencil,
                                                               name, flag_interface, target, openmp)

    def force_on_boundary(self, boundary_obj):
        from lbmpy.boundaries import NoSlip
        if isinstance(boundary_obj, NoSlip):
            return self._force_on_no_slip(boundary_obj)
        else:
            self.__call__()
            return self._force_on_boundary(boundary_obj)

    # ------------------------------ Implementation Details ------------------------------------------------------------

    def _force_on_no_slip(self, boundary_obj):
        dh = self._data_handling
        ff_ghost_layers = dh.ghost_layers_of_field(self.flag_interface.flag_field_name)
        method = self.lb_method
        stencil = np.array(method.stencil)

        result = np.zeros(self.dim)

        for b in dh.iterate(ghost_layers=ff_ghost_layers):
            obj_to_ind_list = b[self._index_array_name].boundary_object_to_index_list
            pdf_array = b[self._field_name]
            if boundary_obj in obj_to_ind_list:
                ind_arr = obj_to_ind_list[boundary_obj]
                indices = [ind_arr[name] for name in ('x', 'y', 'z')[:method.dim]]
                indices.append(ind_arr['dir'])
                values = 2 * pdf_array[tuple(indices)]
                forces = stencil[ind_arr['dir']] * values[:, np.newaxis]
                result += forces.sum(axis=0)
        return dh.reduce_float_sequence(list(result), 'sum')

    def _force_on_boundary(self, boundary_obj):
        dh = self._data_handling
        ff_ghost_layers = dh.ghost_layers_of_field(self.flag_interface.flag_field_name)
        method = self.lb_method
        stencil = np.array(method.stencil)
        inv_direction = np.array([method.stencil.index(inverse_direction(d))
                                 for d in method.stencil])

        result = np.zeros(self.dim)

        for b in dh.iterate(ghost_layers=ff_ghost_layers):
            obj_to_ind_list = b[self._index_array_name].boundary_object_to_index_list
            pdf_array = b[self._field_name]
            if boundary_obj in obj_to_ind_list:
                ind_arr = obj_to_ind_list[boundary_obj]
                indices = [ind_arr[name] for name in ('x', 'y', 'z')[:method.dim]]
                offsets = stencil[ind_arr['dir']]
                inv_dir = inv_direction[ind_arr['dir']]
                fluid_values = pdf_array[tuple(indices) + (ind_arr['dir'],)]
                boundary_values = pdf_array[tuple(ind + offsets[:, i] for i, ind in enumerate(indices)) + (inv_dir,)]
                values = fluid_values + boundary_values
                forces = stencil[ind_arr['dir']] * values[:, np.newaxis]
                result += forces.sum(axis=0)

        return dh.reduce_float_sequence(list(result), 'sum')

    def _create_boundary_kernel(self, symbolic_field, symbolic_index_field, boundary_obj):
        return create_lattice_boltzmann_boundary_kernel(symbolic_field, symbolic_index_field, self.lb_method,
                                                        boundary_obj, target=self._target, openmp=self._openmp)


class LbmWeightInfo(CustomCodeNode):

    # --------------------------- Functions to be used by boundaries --------------------------

    @staticmethod
    def weight_of_direction(dir_idx):
        return sp.IndexedBase(LbmWeightInfo.WEIGHTS_SYMBOL, shape=(1,))[dir_idx]

    # ---------------------------------- Internal ---------------------------------------------

    WEIGHTS_SYMBOL = TypedSymbol("weights", "double")

    def __init__(self, lb_method):
        weights = [str(w.evalf()) for w in lb_method.weights]
        w_sym = LbmWeightInfo.WEIGHTS_SYMBOL
        code = "const double %s [] = { %s };\n" % (w_sym.name, ",".join(weights))
        super(LbmWeightInfo, self).__init__(code, symbols_read=set(), symbols_defined={w_sym})


def create_lattice_boltzmann_boundary_kernel(pdf_field, index_field, lb_method, boundary_functor,
                                             target='cpu', openmp=True):
    elements = [BoundaryOffsetInfo(lb_method.stencil), LbmWeightInfo(lb_method)]
    index_arr_dtype = index_field.dtype.numpy_dtype
    dir_symbol = TypedSymbol("dir", index_arr_dtype.fields['dir'][0])
    elements += [Assignment(dir_symbol, index_field[0]('dir'))]
    elements += boundary_functor(pdf_field=pdf_field, direction_symbol=dir_symbol,
                                 lb_method=lb_method, index_field=index_field)
    return create_indexed_kernel(elements, [index_field], target=target, cpu_openmp=openmp)
