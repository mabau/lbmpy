import pystencils as ps
from pystencils.stencil import have_same_entries
from lbmpy.enums import Stencil

import sympy as sp


class LBStencil:
    r"""
    Class representing a lattice Boltzmann stencil in DxQy notation, where d is the dimension
    (length of the velocity tuples) and y is number of discrete velocities. For every dimension many different version
    of a certain stencil is available. The reason for that is to ensure comparability with the literature.
    Internally the stencil is represented as a tuple of tuples, where the ordering of the tuples plays no role.

    Args:
        stencil: Can be tuple of tuples which represents a DxQy stencil, a string like 'D2Q9' or an enum of
                 lbmpy.enums.Stencil
        ordering: the LBM literature does not use a common order of the discrete velocities, therefore here
                  different common orderings are available. All orderings lead to the same method, it just has
                  to be used consistently. Here more orderings are available to compare intermediate results with
                  the literature.
    """

    def __init__(self, stencil, ordering='walberla'):
        if isinstance(stencil, tuple):
            ordering = None
            self._stencil_entries = stencil
        elif isinstance(stencil, str):
            self._stencil_entries = _predefined_stencils(stencil, ordering)
        elif isinstance(stencil, Stencil):
            self._stencil_entries = _predefined_stencils(stencil.name, ordering)
        else:
            raise ValueError("The LBStencil can only be created with either a tuple of tuples which defines the "
                             "stencil, a string or an Enum of type lbmpy.enums.Stencil")

        valid_stencil = ps.stencil.is_valid(self._stencil_entries)
        if valid_stencil is False:
            raise ValueError("The stencil you have created is not valid. "
                             "It probably contains elements with different lengths")

        if len(set(self._stencil_entries)) < len(self._stencil_entries):
            raise ValueError("The stencil you have created is not valid. "
                             "It contains duplicated elements")

        self._ordering = ordering
        self._dim = len(self._stencil_entries[0])
        self._q = len(self._stencil_entries)

    @property
    def D(self):
        return self._dim

    @property
    def Q(self):
        return self._q

    @property
    def ordering(self):
        return self._ordering

    @property
    def name(self):
        return f"D{self.D}Q{self.Q}"

    @property
    def stencil_entries(self):
        return self._stencil_entries

    @property
    def inverse_stencil_entries(self):
        return tuple([ps.stencil.inverse_direction(d) for d in self._stencil_entries])

    def plot(self, slice=False, **kwargs):
        ps.stencil.plot(stencil=self._stencil_entries, slice=slice, **kwargs)

    def index(self, direction):
        assert len(direction) == self.D, "direction must match stencil.D"
        return self._stencil_entries.index(direction)

    def inverse_index(self, direction):
        assert len(direction) == self.D, "direction must match stencil.D"
        direction = ps.stencil.inverse_direction(direction)
        return self._stencil_entries.index(direction)

    def __getitem__(self, index):
        return self._stencil_entries[index]

    def __iter__(self):
        yield from self._stencil_entries

    def __eq__(self, other):
        return self.ordering == other.ordering and have_same_entries(self._stencil_entries, other.stencil_entries)

    def __len__(self):
        return len(self._stencil_entries)

    def __str__(self):
        return str(self._stencil_entries)

    def __hash__(self):
        return hash(self._stencil_entries)

    def _repr_html_(self):
        table = """
        <table style="border:none; width: 100%">
            <tr {nb}>
                <th {nb} >Nr.</th>
                <th {nb} >Direction Name</th>
                <th {nb} >Direction </th>
            </tr>
            {content}
        </table>
        """
        content = ""
        for i, direction in enumerate(self._stencil_entries):
            vals = {
                'nr': sp.latex(i),
                'name': sp.latex(ps.stencil.offset_to_direction_string(direction)),
                'entry': sp.latex(direction),
                'nb': 'style="border:none"'
            }
            content += """<tr {nb}>
                            <td {nb}>${nr}$</td>
                            <td {nb}>${name}$</td>
                            <td {nb}>${entry}$</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')


def _predefined_stencils(stencil: str, ordering: str):
    predefined_stencils = {
        'D2Q9': {
            'walberla': ((0, 0),
                         (0, 1), (0, -1), (-1, 0), (1, 0),
                         (-1, 1), (1, 1), (-1, -1), (1, -1),),
            'counterclockwise': ((0, 0),
                                 (1, 0), (0, 1), (-1, 0), (0, -1),
                                 (1, 1), (-1, 1), (-1, -1), (1, -1)),
            'braunschweig': ((0, 0),
                             (-1, 1), (-1, 0), (-1, -1), (0, -1),
                             (1, -1), (1, 0), (1, 1), (0, 1)),
            'uk': ((0, 0),
                   (1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, -1), (-1, 1), (1, -1),
                   ),
            'lehmann': ((0, 0),
                        (1, 0), (-1, 0), (0, 1), (0, -1),
                        (1, 1), (-1, -1), (1, -1), (-1, 1),
                        )
        },
        'D2V17': {
            'walberla': (
                (0, 0), (0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1), (-2, -2), (2, -2),
                (-2, 2),
                (2, 2), (0, -3), (-3, 0), (3, 0), (0, 3)),
        },
        'D2V37': {
            'walberla': (
                (0, 0), (0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (-2, 0),
                (2, 0),
                (0, 2), (-1, -2), (1, -2), (-2, -1), (2, -1), (-2, 1), (2, 1), (-1, 2), (1, 2), (-2, -2), (2, -2),
                (-2, 2),
                (2, 2), (0, -3), (-3, 0), (3, 0), (0, 3), (-1, -3), (1, -3), (-3, -1), (3, -1), (-3, 1), (3, 1),
                (-1, 3),
                (1, 3))
        },
        'D3Q7': {
            'walberla': ((0, 0, 0),
                         (0, 1, 0), (0, -1, 0),
                         (-1, 0, 0), (1, 0, 0),
                         (0, 0, 1), (0, 0, -1))

        },
        'D3Q15': {
            'walberla':
                ((0, 0, 0),
                 (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1),
                 (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1), (-1, 1, -1), (1, -1, -1),
                 (-1, -1, -1)),
            'premnath': ((0, 0, 0),
                         (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                         (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
                         (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)),
            'lehmann': ((0, 0, 0),
                        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                        (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1),
                        (1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1)),
        },
        'D3Q19': {
            'walberla': ((0, 0, 0),
                         (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1),
                         (-1, 1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0),
                         (0, 1, 1), (0, -1, 1), (-1, 0, 1), (1, 0, 1),
                         (0, 1, -1), (0, -1, -1), (-1, 0, -1), (1, 0, -1)),
            'counterclockwise': ((0, 0, 0),
                                 (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                                 (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1),
                                 (0, 1, 1), (0, -1, -1), (1, -1, 0), (-1, 1, 0),
                                 (1, 0, -1), (-1, 0, 1), (0, 1, -1), (0, -1, 1)),
            'braunschweig': ((0, 0, 0),
                             (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                             (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
                             (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
                             (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)),
            'premnath': ((0, 0, 0),
                         (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                         (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
                         (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                         (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1)),
            'lehmann': ((0, 0, 0),
                        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                        (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1),
                        (0, 1, 1), (0, -1, -1), (1, -1, 0), (-1, 1, 0),
                        (1, 0, -1), (-1, 0, 1), (0, 1, -1), (0, -1, 1)),
        },
        'D3Q27': {
            'walberla': ((0, 0, 0),
                         (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1),
                         (-1, 1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0),
                         (0, 1, 1), (0, -1, 1), (-1, 0, 1), (1, 0, 1),
                         (0, 1, -1), (0, -1, -1), (-1, 0, -1), (1, 0, -1),
                         (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1), (-1, 1, -1), (1, -1, -1),
                         (-1, -1, -1)),
            'premnath': ((0, 0, 0),
                         (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                         (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
                         (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                         (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
                         (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
                         (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)),
            'fakhari': ((0, 0, 0),
                        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                        (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
                        (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
                        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
                        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1),
                        (0, -1, -1)),
            'lehmann': ((0, 0, 0),
                        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                        (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1),
                        (0, 1, 1), (0, -1, -1), (1, -1, 0), (-1, 1, 0),
                        (1, 0, -1), (-1, 0, 1), (0, 1, -1), (0, -1, 1),
                        (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                        (1, -1, -1)),
        }
    }

    try:
        return predefined_stencils[stencil][ordering]
    except KeyError:
        err_msg = ""
        for stencil, ordering_names in predefined_stencils.items():
            err_msg += "  %s: %s\n" % (stencil, ", ".join(ordering_names.keys()))

        raise ValueError("No such stencil available. "
                         "Available stencils: <stencil_name>( <ordering_names> )\n" + err_msg)
