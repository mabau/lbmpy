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


class PeriodicTwoFieldsAccessor(PdfFieldAccessor):
    """Access scheme that builds periodicity into the kernel.

    Introduces a condition on every load, such that at the borders the periodic value is loaded. The periodicity is specified as a tuple of booleans, one for
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
            for coord_id, dir_element in enumerate(pull_direction):
                if not self._periodicity[coord_id]:
                    periodic_pull_direction.append(dir_element)
                    continue

                lower_limit = self._ghostLayers
                upper_limit = field.spatial_shape[coord_id] - 1 - self._ghostLayers
                limit_diff = upper_limit - lower_limit
                loop_counter = LoopOverCoordinate.get_loop_counter_symbol(coord_id)
                if dir_element == 0:
                    periodic_pull_direction.append(0)
                elif dir_element == 1:
                    new_dir_element = sp.Piecewise((dir_element, loop_counter < upper_limit), (-limit_diff, True))
                    periodic_pull_direction.append(new_dir_element)
                elif dir_element == -1:
                    new_dir_element = sp.Piecewise((dir_element, loop_counter > lower_limit), (limit_diff, True))
                    periodic_pull_direction.append(new_dir_element)
                else:
                    raise NotImplementedError("This accessor supports only nearest neighbor stencils")
            result.append(field[tuple(periodic_pull_direction)](i))
        return result

    @staticmethod
    def write(field, stencil):
        return [field(i) for i in range(len(stencil))]


class AAEvenTimeStepAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        return [field(i) for i in range(len(stencil))]

    @staticmethod
    def write(field, stencil):
        return [field(stencil.index(inverse_direction(d))) for d in stencil]


class AAOddTimeStepAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        res = []
        for i, d in enumerate(stencil):
            inv_dir = inverse_direction(d)
            field_access = field[inv_dir](stencil.index(inv_dir))
            res.append(field_access)
        return res

    @staticmethod
    def write(field, stencil):
        return [field[d](i) for i, d in enumerate(stencil)]


# -------------------------------------------- Visualization -----------------------------------------------------------


def visualize_field_mapping(axes, stencil, field_mapping, color='b'):
    from lbmpy.plot2d import LbGrid
    grid = LbGrid(3, 3)
    grid.fill_with_default_arrows()
    for field_access, direction in zip(field_mapping, stencil):
        field_position = stencil[field_access.index[0]]
        neighbor = field_access.offsets
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



