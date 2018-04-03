import sympy as sp
import abc
from lbmpy.stencils import inverse_direction
from pystencils import Field


# ------------------------------------------------ Interface -----------------------------------------------------------
from pystencils.astnodes import LoopOverCoordinate


class PdfFieldAccessor(object):
    """
    Defines how data is read and written in an LBM time step.

    Examples for PdfFieldAccessors are
         - stream pull using two fields (source/destination)
         - inplace collision access, without streaming
         - esoteric twist single field update
         -
    """
    @abc.abstractmethod
    def read(self, field, stencil):
        """Returns sequence of field accesses for all stencil values where pdfs are read from"""
        pass

    @abc.abstractmethod
    def write(self, field, stencil):
        """Returns sequence  of field accesses for all stencil values where pdfs are written to"""
        pass


# ----------------------------------------------- Implementation -------------------------------------------------------


class CollideOnlyInplaceAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        return [field(i) for i in range(len(stencil))]

    @staticmethod
    def write(field, stencil):
        return [field(i) for i in range(len(stencil))]


class StreamPullTwoFieldsAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        return [field[inverse_direction(d)](i) for i, d in enumerate(stencil)]

    @staticmethod
    def write(field, stencil):
        return [field(i) for i in range(len(stencil))]


class Pseudo2DTwoFieldsAccessor(PdfFieldAccessor):
    """Useful if a 3D simulation of a domain with size (x,y,1) is done and the dimensions with size 1 
    is periodic. In this case no periodicity exchange has to be done"""
    def __init__(self, collapsedDim):
        self._collapsedDim = collapsedDim

    def read(self, field, stencil):
        result = []
        for i, d in enumerate(stencil):
            direction = list(d)
            direction[self._collapsedDim] = 0
            result.append(field[inverse_direction(tuple(direction))](i))
        return result

    @staticmethod
    def write(field, stencil):
        return [field(i) for i in range(len(stencil))]


class PeriodicTwoFieldsAccessor(PdfFieldAccessor):
    """Access scheme that builds periodicity into the kernel, by introducing a condition on every load,
    such that at the borders the periodic value is loaded. The periodicity is specified as a tuple of booleans, one for
    each direction. The second parameter `ghost_layers` specifies the number of assumed ghost layers of the field.
    For the periodic kernel itself no ghost layers are required, however other kernels might need them. 
    """
    def __init__(self, periodicity, ghost_layers=0):
        self._periodicity = periodicity
        self._ghostLayers = ghost_layers

    def read(self, field, stencil):
        result = []
        for i, d in enumerate(stencil):
            pull_direction = inverse_direction(d)
            periodic_pull_direction = []
            for coordId, dirElement in enumerate(pull_direction):
                if not self._periodicity[coordId]:
                    periodic_pull_direction.append(dirElement)
                    continue

                lower_limit = self._ghostLayers
                upper_limit = field.spatial_shape[coordId] - 1 - self._ghostLayers
                limit_diff = upper_limit - lower_limit
                loop_counter = LoopOverCoordinate.get_loop_counter_symbol(coordId)
                if dirElement == 0:
                    periodic_pull_direction.append(0)
                elif dirElement == 1:
                    new_dir_element = sp.Piecewise((dirElement, loop_counter < upper_limit), (-limit_diff, True))
                    periodic_pull_direction.append(new_dir_element)
                elif dirElement == -1:
                    new_dir_element = sp.Piecewise((dirElement, loop_counter > lower_limit), (limit_diff, True))
                    periodic_pull_direction.append(new_dir_element)
                else:
                    raise NotImplementedError("This accessor supports only nearest neighbor stencils")
            result.append(field[tuple(periodic_pull_direction)](i))
        return result

    @staticmethod
    def write(field, stencil):
        return [field(i) for i in range(len(stencil))]


class AABBEvenTimeStepAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        return [field(i) for i in range(len(stencil))]

    @staticmethod
    def write(field, stencil):
        return [field(stencil.index(inverse_direction(d))) for d in stencil]


class AABBOddTimeStepAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        res = []
        for i, d in enumerate(stencil):
            inv_dir = inverse_direction(d)
            field_access = field[inv_dir](stencil.index(inv_dir))
            res.append(field_access)
        return

    @staticmethod
    def write(field, stencil):
        return [field[d](i) for i, d in enumerate(stencil)]


class EsotericTwistAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        result = []
        for i, direction in enumerate(stencil):
            direction = inverse_direction(direction)
            neighbor_offset = tuple([-e if e <= 0 else 0 for e in direction])
            result.append(field[neighbor_offset](i))
        return result

    @staticmethod
    def write(field, stencil):
        result = []
        for i, direction in enumerate(stencil):
            neighbor_offset = tuple([e if e >= 0 else 0 for e in direction])
            inverse_index = stencil.index(inverse_direction(direction))
            result.append(field[neighbor_offset](inverse_index))
        return result


# -------------------------------------------- Visualization -----------------------------------------------------------


def visualize_field_mapping(axes, stencil, field_mapping, color='b'):
    from lbmpy.plot2d import LbGrid
    grid = LbGrid(3, 3)
    grid.fill_with_default_arrows()
    for fieldAccess, direction in zip(field_mapping, stencil):
        field_position = stencil[fieldAccess.index[0]]
        neighbor = fieldAccess.offsets
        grid.add_arrow((1 + neighbor[0], 1 + neighbor[1]),
                       arrow_position=field_position, arrow_direction=direction, color=color)
    grid.draw(axes)


def visualize_pdf_field_accessor(pdf_field_accessor, figure=None):
    from lbmpy.stencils import get_stencil

    if figure is None:
        import matplotlib.pyplot as plt
        figure = plt.gcf()

    stencil = get_stencil('D2Q9')

    figure.patch.set_facecolor('white')

    field = Field.create_generic('f', spatial_dimensions=2, index_dimensions=1)
    pre_collision_accesses = pdf_field_accessor.read(field, stencil)
    post_collision_accesses = pdf_field_accessor.write(field, stencil)

    ax_left = figure.add_subplot(1, 2, 1)
    ax_right = figure.add_subplot(1, 2, 2)

    visualize_field_mapping(ax_left, stencil, pre_collision_accesses, color='k')
    visualize_field_mapping(ax_right, stencil, post_collision_accesses, color='r')

    ax_left.set_title("Read")
    ax_right.set_title("Write")



