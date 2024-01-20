import numpy as np

from lbmpy.geometry import get_shear_flow_velocity_field
from lbmpy.scenarios import create_fully_periodic_flow


def test_builtin_periodicity():
    shape = (16, 16)
    initial_vel = get_shear_flow_velocity_field(shape, 0.05, 0.1)

    sc_ref = create_fully_periodic_flow(initial_velocity=initial_vel, relaxation_rate=1.8)
    sc_test = create_fully_periodic_flow(initial_velocity=initial_vel, relaxation_rate=1.8,
                                         optimization={'builtin_periodicity': (True, True)})
    sc_ref.run(20)
    sc_test.run(20)
    np.testing.assert_almost_equal(sc_ref.velocity[:, :], sc_test.velocity[:, :])
