import numpy as np
from math import sinh, cosh, cos, sin, pi


def analytic_rising_speed(gravitational_acceleration, bubble_diameter, viscosity_gas):
    r"""
    Calculated the analytical rising speed of a bubble. This is the expected end rising speed.
    Args:
        gravitational_acceleration: the gravitational acceleration acting in the simulation scenario. Usually it gets
                                    calculated based on dimensionless parameters which describe the scenario
        bubble_diameter: the diameter of the bubble at the beginning of the simulation
        viscosity_gas: the viscosity of the fluid inside the bubble
    """
    result = -(gravitational_acceleration * bubble_diameter * bubble_diameter) / (12.0 * viscosity_gas)
    return result


def analytical_solution_microchannel(reference_length, length_x, length_y,
                                     kappa_top, kappa_bottom,
                                     t_h, t_c, t_0,
                                     reference_surface_tension, dynamic_viscosity_light_phase,
                                     transpose=True):
    """
    https://www.sciencedirect.com/science/article/pii/S0021999113005986
    """
    l_ref = reference_length
    sigma_t = reference_surface_tension

    kstar = kappa_top / kappa_bottom
    mp = (l_ref // 2) - 1

    w = pi / l_ref
    a = mp * w
    b = mp * w

    f = 1.0 / (kstar * sinh(b) * cosh(a) + sinh(a) * cosh(b))
    g = sinh(a) * f

    h = (sinh(a) ** 2 - a ** 2) * (sinh(b) ** 2 - b ** 2) / \
        ((sinh(b) ** 2 - b ** 2) * (sinh(2.0 * a) - 2.0 * a)
         + (sinh(a) ** 2 - a ** 2) * (sinh(2.0 * b) - 2.0 * b))

    Ca1 = sinh(a) ** 2 / (sinh(a) ** 2 - a ** 2)
    Ca2 = -1.0 * mp * a / (sinh(a) ** 2 - a ** 2)
    Ca3 = (2 * a - sinh(2 * a)) / (2.0 * (sinh(a) ** 2 - a ** 2))

    Cb1 = sinh(b) ** 2 / (sinh(b) ** 2 - b ** 2)
    Cb2 = -1.0 * mp * b / (sinh(b) ** 2 - b ** 2)
    Cb3 = (-2 * b + sinh(2 * b)) / (2.0 * (sinh(b) ** 2 - b ** 2))

    umax = -1.0 * (t_0 * sigma_t / dynamic_viscosity_light_phase) * g * h
    jj = 0
    xx = np.linspace(-l_ref - 0.5, l_ref - 0.5, length_x)
    yy = np.linspace(-mp, mp, length_y)
    u_x = np.zeros([len(xx), len(yy)])
    u_y = np.zeros([len(xx), len(yy)])
    t_a = np.zeros([len(xx), len(yy)])
    tt = t_c - t_h
    nom = kstar * t_c * mp + t_h * mp
    denom = mp + kstar * mp
    for y in yy:
        ii = 0
        for x in xx:
            swx = sin(w * x)
            cwx = cos(w * x)

            if y > 0:
                tmp1 = ((Ca1 + w * (Ca2 + Ca3 * y)) * cosh(w * y) + (Ca3 + w * Ca1 * y) * sinh(w * y))
                tmp2 = (Ca1 * y * cosh(w * y) + (Ca2 + Ca3 * y) * sinh(w * y))

                t_a[ii, jj] = (tt * y + nom) / denom + t_0 * f * sinh(a - y * w) * cwx
                u_x[ii, jj] = umax * tmp1 * swx
                u_y[ii, jj] = -w * umax * tmp2 * cwx

            elif y <= 0:
                tmp3 = (sinh(a) * cosh(w * y) - kstar * sinh(w * y) * cosh(a))
                tmp4 = ((Cb1 + w * (Cb2 + Cb3 * y)) * cosh(w * y) + (Cb3 + w * Cb1 * y) * sinh(w * y))

                t_a[ii, jj] = (kstar * tt * y + nom) / denom + t_0 * f * tmp3 * cwx
                u_x[ii, jj] = umax * tmp4 * swx
                u_y[ii, jj] = -w * umax * (Cb1 * y * cosh(w * y) + (Cb2 + Cb3 * y) * sinh(w * y)) * cwx

            ii += 1
        jj += 1
    x, y = np.meshgrid(xx, yy)
    if transpose:
        return x, y, u_x.T, u_y.T, t_a.T
    else:
        return x, y, u_x, u_y, t_a
