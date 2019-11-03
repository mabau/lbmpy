def laplacian(phi_field):
    r"""
    Get a symbolic expression for the laplacian.
    Args:
        phi_field: the phase-field on which the laplacian is applied
    """
    lp_phi = 16.0 * ((phi_field[1, 0, 0]) + (phi_field[-1, 0, 0])
                     + (phi_field[0, 1, 0]) + (phi_field[0, -1, 0])
                     + (phi_field[0, 0, 1]) + (phi_field[0, 0, -1])) \
        + 1.0 * (
                (phi_field[1, 1, 1]) + (phi_field[-1, 1, 1])
            + (phi_field[1, -1, 1]) + (phi_field[-1, -1, 1])
            + (phi_field[1, 1, -1]) + (phi_field[-1, 1, -1])
            + (phi_field[1, -1, -1]) + (phi_field[-1, -1, -1])) \
        + 4.0 * (
                (phi_field[1, 1, 0]) + (phi_field[-1, 1, 0])
            + (phi_field[1, -1, 0]) + (phi_field[-1, -1, 0])
            + (phi_field[1, 0, 1]) + (phi_field[-1, 0, 1])
            + (phi_field[1, 0, -1]) + (phi_field[-1, 0, -1])
            + (phi_field[0, 1, 1]) + (phi_field[0, -1, 1])
            + (phi_field[0, 1, -1]) + (phi_field[0, -1, -1])) \
        - 152.0 * phi_field[0, 0, 0]

    lp_phi = lp_phi / 36

    return lp_phi


def isotropic_gradient(phi_field):
    r"""
    Get a symbolic expression for the isotropic gradient of the phase-field
    Args:
        phi_field: the phase-field on which the isotropic gradient is applied
    """
    grad_phi_x = 16.00 * (phi_field[1, 0, 0] - phi_field[-1, 0, 0])\
        + (phi_field[1, 1, 1] - phi_field[-1, 1, 1] + phi_field[1, -1, 1] - phi_field[-1, -1, 1]
           + phi_field[1, 1, -1] - phi_field[-1, 1, -1] + phi_field[1, -1, -1] - phi_field[-1, -1, -1])\
        + 4.00 * (phi_field[1, 1, 0] - phi_field[-1, 1, 0] + phi_field[1, -1, 0] - phi_field[-1, -1, 0]
                  + phi_field[1, 0, 1] - phi_field[-1, 0, 1] + phi_field[1, 0, -1] - phi_field[-1, 0, -1])
    grad_phi_y = 16.00 * (phi_field[0, 1, 0] - phi_field[0, -1, 0]) \
        + (phi_field[1, 1, 1] + phi_field[-1, 1, 1] - phi_field[1, -1, 1] - phi_field[-1, -1, 1]
           + phi_field[1, 1, -1] + phi_field[-1, 1, -1] - phi_field[1, -1, -1] - phi_field[-1, -1, -1])\
        + 4.00 * (phi_field[1, 1, 0] + phi_field[-1, 1, 0] - phi_field[1, -1, 0] - phi_field[-1, -1, 0]
                  + phi_field[0, 1, 1] - phi_field[0, -1, 1] + phi_field[0, 1, -1] - phi_field[0, -1, -1])
    grad_phi_z = 16.00 * (phi_field[0, 0, 1] - phi_field[0, 0, -1]) \
        + (phi_field[1, 1, 1] + phi_field[-1, 1, 1] + phi_field[1, -1, 1] + phi_field[-1, -1, 1]
            - phi_field[1, 1, -1] - phi_field[-1, 1, -1] - phi_field[1, -1, -1] - phi_field[-1, -1, -1])\
        + 4.00 * (phi_field[1, 0, 1] + phi_field[-1, 0, 1] - phi_field[1, 0, -1] - phi_field[-1, 0, -1]
                  + phi_field[0, 1, 1] + phi_field[0, -1, 1] - phi_field[0, 1, -1] - phi_field[0, -1, -1])

    grad = [grad_phi_x / 72, grad_phi_y / 72, grad_phi_z / 72]

    return grad
