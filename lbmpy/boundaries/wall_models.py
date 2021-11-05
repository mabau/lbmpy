import sympy as sp
from pystencils import Assignment
from pystencils.stencil import offset_to_direction_string

from lbmpy.advanced_streaming.indexing import MirroredStencilDirections
from lbmpy.boundaries.boundaryconditions import LbBoundary


class WallFunctionBounce(LbBoundary):
    """
    Wall function based on the bounce back idea.

    Args:
        stencil: LBM stencil which is used for the simulation
        normal_direction: optional normal direction. If the Free slip boundary is applied to a certain side in the
                          domain it is not necessary to calculate the normal direction since it can be stated for all
                          boundary cells. This reduces the memory space for the index array significantly.
        name: optional name of the boundary.
    """

    def __init__(self, stencil, normal_direction, name=None):
        """Set an optional name here, to mark boundaries, for example for force evaluations"""
        self.stencil = stencil

        if len(normal_direction) - normal_direction.count(0) != 1:
            raise ValueError("Only normal directions for straight walls are supported for example (0, 1, 0) for "
                             "a WallFunctionBounce applied to the southern boundary of the domain")

        self.mirror_axis = normal_direction.index(*[dir for dir in normal_direction if dir != 0])

        self.normal_direction = normal_direction
        self.dim = len(stencil[0])

        if name is None:
            name = f"WFB : {offset_to_direction_string([-x for x in normal_direction])}"

        super(WallFunctionBounce, self).__init__(name)

    def get_additional_code_nodes(self, lb_method):
        return [MirroredStencilDirections(self.stencil, self.mirror_axis)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        normal_direction = self.normal_direction
        mirrored_stencil_symbol = MirroredStencilDirections._mirrored_symbol(self.mirror_axis)
        mirrored_direction = inv_dir[sp.IndexedBase(mirrored_stencil_symbol, shape=(1,))[dir_symbol]]

        return Assignment(f_in(inv_dir[dir_symbol]), f_in[normal_direction](mirrored_direction))
