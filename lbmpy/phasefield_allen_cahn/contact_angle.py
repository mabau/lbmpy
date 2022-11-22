import math
import sympy as sp

from pystencils.astnodes import SympyAssignment

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

        if field.index_dimensions == 0:
            if math.isclose(90, self._contact_angle, abs_tol=1e-5):
                return [SympyAssignment(field.center, field[neighbor])]

            dist = TypedSymbol("h", self._data_type)
            angle = TypedSymbol("a", self._data_type)
            tmp = TypedSymbol("tmp", self._data_type)

            result = [SympyAssignment(tmp, CastFunc(sum([x * x for x in neighbor]), self._data_type)),
                      SympyAssignment(dist, 0.5 * sp.sqrt(tmp)),
                      SympyAssignment(angle, math.cos(math.radians(self._contact_angle)))]

            var = - dist * (4.0 / self._interface_width) * angle
            tmp = 1 + var
            else_branch = (tmp - sp.sqrt(tmp * tmp - 4 * var * field[neighbor])) / var - field[neighbor]
            update = sp.Piecewise((field[neighbor], dist < 0.001), (else_branch, True))

            result.append(SympyAssignment(field.center, update))
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
