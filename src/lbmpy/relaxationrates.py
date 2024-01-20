import sympy as sp

from lbmpy.moments import is_bulk_moment, is_shear_moment


def relaxation_rate_from_lattice_viscosity(nu):
    r"""Computes relaxation rate from lattice viscosity: :math:`\omega = \frac{1}{3\nu_L + \frac{1}{2}}`"""
    return 2 / (6 * nu + 1)


def lattice_viscosity_from_relaxation_rate(omega):
    r"""Computes lattice viscosity from relaxation rate:
    :math:`\nu_L=\frac{1}{3}\left(\frac{1}{\omega}-\frac{1}{2}\right)`"""
    return (2 - omega) / (6 * omega)


def relaxation_rate_from_magic_number(hydrodynamic_relaxation_rate, magic_number=sp.Rational(3, 16)):
    """Computes second TRT relaxation rate from magic number."""
    omega = hydrodynamic_relaxation_rate
    return (4 - 2 * omega) / (4 * magic_number * omega + 2 - omega)


def get_shear_relaxation_rate(method):
    """
    Assumes that all shear moments are relaxed with same rate - returns this rate
    Shear moments in 3D are: x*y, x*z and y*z - in 2D its only x*y
    The shear relaxation rate determines the viscosity in hydrodynamic LBM schemes
    """
    if hasattr(method, 'shear_relaxation_rate'):
        return method.shear_relaxation_rate

    relaxation_rates = set()
    for moment, relax_info in method.relaxation_info_dict.items():
        if is_shear_moment(moment, method.dim):
            relaxation_rates.add(relax_info.relaxation_rate)
    if len(relaxation_rates) == 1:
        return relaxation_rates.pop()
    else:
        if len(relaxation_rates) > 1:
            raise ValueError("Shear moments are relaxed with different relaxation times: %s" % (relaxation_rates,))
        else:
            all_relaxation_rates = set(v.relaxation_rate for v in method.relaxation_info_dict.values())
            if len(all_relaxation_rates) == 1:
                return list(all_relaxation_rates)[0]
            raise NotImplementedError("Shear moments seem to be not relaxed separately - "
                                      "Can not determine their relaxation rate automatically")


def get_bulk_relaxation_rate(method):
    """
    The bulk moment is x^2 + y^2 + z^2, plus a constant for orthogonalization.
    If this moment does not exist, the bulk relaxation is part of the shear relaxation.
    The bulk relaxation rate determines the bulk viscosity in hydrodynamic LBM schemes.
    """
    for moment, relax_info in method.relaxation_info_dict.items():
        if is_bulk_moment(moment, method.dim):
            return relax_info.relaxation_rate
    return get_shear_relaxation_rate(method)


def relaxation_rate_scaling(omega, level_scale_factor):
    """Computes adapted omega for refinement.

    Args:
        omega: relaxation rate
        level_scale_factor: resolution of finer grid i.e. 2, 4, 8

    Returns:
        relaxation rate on refined grid
    """
    return omega / (omega / 2 + level_scale_factor * (1 - omega / 2))
