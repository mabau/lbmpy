import math

import pystencils as ps
from pystencils.boundaries.boundaryhandling import BoundaryHandling

from lbmpy.enums import Stencil
from lbmpy.phasefield_allen_cahn.contact_angle import ContactAngle
from lbmpy.stencils import LBStencil

import numpy as np


def test_contact_angle():
    stencil = LBStencil(Stencil.D2Q9)
    contact_angle = 45
    phase_value = 0.5

    domain_size = (9, 9)

    dh = ps.create_data_handling(domain_size, periodicity=(False, False))

    C = dh.add_array("C", values_per_cell=1)
    dh.fill("C", 0.0, ghost_layers=True)
    dh.fill("C", phase_value, ghost_layers=False)

    bh = BoundaryHandling(dh, C.name, stencil, target=ps.Target.CPU)
    bh.set_boundary(ContactAngle(45, 5), ps.make_slice[:, 0])
    bh()

    h = 1.0
    myA = 1.0 - 0.5 * h * (4.0 / 5) * math.cos(math.radians(contact_angle))

    phase_on_boundary = (myA - np.sqrt(myA * myA - 4.0 * (myA - 1.0) * phase_value)) / (myA - 1.0) - phase_value

    np.testing.assert_almost_equal(dh.cpu_arrays["C"][5, 0], phase_on_boundary)

    assert ContactAngle(45, 5) == ContactAngle(45, 5)
    assert ContactAngle(46, 5) != ContactAngle(45, 5)
