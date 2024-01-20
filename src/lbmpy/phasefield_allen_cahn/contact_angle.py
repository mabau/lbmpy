import math
import sympy as sp

from pystencils.astnodes import Block, Conditional, SympyAssignment

from pystencils.boundaries.boundaryhandling import BoundaryOffsetInfo
from pystencils.boundaries.boundaryconditions import Boundary

from pystencils.typing import TypedSymbol
from pystencils.typing import CastFunc


class ContactAngle(Boundary):
    r"""
    Wettability condition on solid boundaries according to equation 25 in :cite:`Fakhari2018`.

    Args:
        contact_angle: contact angle in degrees which is applied between the fluid and the solid boundary.
        interface_width: interface width of the phase field model.
        name: optional name of the boundary
        data_type: data type for temporary variables which are used.
    """

    inner_or_boundary = False
    single_link = True

    def __init__(self, contact_angle, interface_width, name=None, data_type='double'):
        self._contact_angle = contact_angle
        self._interface_width = interface_width
        self._data_type = data_type

        super(ContactAngle, self).__init__(name)

    def __call__(self, field, direction_symbol, **kwargs):

        neighbor = BoundaryOffsetInfo.offset_from_dir(direction_symbol, field.spatial_dimensions)
        dist = TypedSymbol("h", self._data_type)
        angle = TypedSymbol("a", self._data_type)
        d = CastFunc(sum([x * x for x in neighbor]), self._data_type)

        var = - dist * (4.0 / self._interface_width) * angle
        tmp = 1 + var
        else_branch = (tmp - sp.sqrt(tmp * tmp - 4.0 * var * field[neighbor])) / var - field[neighbor]
        if field.index_dimensions == 0:
            if isinstance(self._contact_angle, (int, float)):
                result = [SympyAssignment(angle, math.cos(math.radians(self._contact_angle))),
                          SympyAssignment(dist, 0.5 * sp.sqrt(d)),
                          Conditional(sp.LessThan(var * var, 0.000001),
                                      Block([SympyAssignment(field.center, field[neighbor])]),
                                      Block([SympyAssignment(field.center, else_branch)]))]
                return result
            else:
                result = [SympyAssignment(angle, sp.cos(self._contact_angle * (sp.pi / sp.Number(180)))),
                          SympyAssignment(dist, 0.5 * sp.sqrt(d)),
                          Conditional(sp.LessThan(var * var, 0.000001),
                                      Block([SympyAssignment(field.center, field[neighbor])]),
                                      Block([SympyAssignment(field.center, else_branch)]))]
                return result

        else:
            raise NotImplementedError("Contact angle only implemented for phase-fields which have a single "
                                      "value for each cell")

    def __hash__(self):
        return hash("ContactAngle")

    def __eq__(self, other):
        if not isinstance(other, ContactAngle):
            return False
        return self.__dict__ == other.__dict__
