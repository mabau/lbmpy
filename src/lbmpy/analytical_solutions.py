from typing import Union
from numpy.typing import NDArray


def poiseuille_flow(middle_distance: Union[float, NDArray], height,
                    ext_force_density: float, dyn_visc: float) -> Union[float, NDArray]:
    """
    Analytical solution for plane Poiseuille flow.

    Args:
        middle_distance: Distance to the middle plane of the channel.
        height: Distance between the boundaries.
        ext_force_density: Force density on the fluid normal to the boundaries.
        dyn_visc: dyn_visc

    Returns:
        A numpy array of the poiseuille profile if middle_distance is given as array otherwise of velocity of
        the position given with middle_distance
    """
    return ext_force_density * 1. / (2 * dyn_visc) * (height**2.0 / 4.0 - middle_distance**2.0)
