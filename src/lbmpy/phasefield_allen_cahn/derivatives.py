from pystencils.fd.derivation import FiniteDifferenceStencilDerivation

import sympy as sp


def laplacian_symbolic(field, stencil):
    r"""
    Get a symbolic expression for the laplacian of a field.
    Args:
        field: the field on which the laplacian is applied to
        stencil: stencil to derive the finite difference for the laplacian (2nd order isotropic)
    """
    lap = sp.simplify(0)
    for i in range(stencil.D):
        deriv = FiniteDifferenceStencilDerivation((i, i), stencil)
        for j in range(stencil.D):
            # assume the stencil is symmetric
            deriv.assume_symmetric(dim=j, anti_symmetric=False)

        # set weights for missing degrees of freedom in the calculation and assume the stencil is isotropic
        if stencil.D == 2 and stencil.Q == 9:
            res = deriv.get_stencil(isotropic=True)
            lap += res.apply(field.center)
        elif stencil.D == 2 and stencil.Q == 25:
            deriv.set_weight((2, 0), sp.Rational(1, 10))

            res = deriv.get_stencil(isotropic=True)
            lap += res.apply(field.center)
        elif stencil.D == 3 and stencil.Q == 15:
            deriv.set_weight((0, 0, 0), sp.Rational(-32, 27))
            res = deriv.get_stencil(isotropic=True)
            lap += res.apply(field.center)
        elif stencil.D == 3 and stencil.Q == 19:
            res = deriv.get_stencil(isotropic=True)
            lap += res.apply(field.center)
        elif stencil.D == 3 and stencil.Q == 27:
            deriv.set_weight((0, 0, 0), sp.Rational(-38, 27))
            res = deriv.get_stencil(isotropic=True)
            lap += res.apply(field.center)
        else:
            raise ValueError(f"stencil with {stencil.D} dimensions and {stencil.Q} entries is not supported")

    return lap


def isotropic_gradient_symbolic(field, stencil):
    r"""
    Get a symbolic expression for the isotropic gradient of the phase-field
    Args:
        field: the field on which the isotropic gradient is applied
        stencil: stencil to derive the finite difference for the gradient (2nd order isotropic)
    """
    deriv = FiniteDifferenceStencilDerivation((0,), stencil)

    deriv.assume_symmetric(0, anti_symmetric=True)
    deriv.assume_symmetric(1, anti_symmetric=False)
    if stencil.D == 3:
        deriv.assume_symmetric(2, anti_symmetric=False)

    # set weights for missing degrees of freedom in the calculation and assume the stencil is isotropic
    # furthermore the stencils gets rotated to get the y and z components
    if stencil.D == 2 and stencil.Q == 9:
        res = deriv.get_stencil(isotropic=True)
        grad = [res.apply(field.center), res.rotate_weights_and_apply(field.center, (0, 1)), 0]
    elif stencil.D == 2 and stencil.Q == 25:
        deriv.set_weight((2, 0), sp.Rational(1, 10))

        res = deriv.get_stencil(isotropic=True)
        grad = [res.apply(field.center), res.rotate_weights_and_apply(field.center, (0, 1)), 0]
    elif stencil.D == 3 and stencil.Q == 15:
        res = deriv.get_stencil(isotropic=True)
        grad = [res.apply(field.center),
                res.rotate_weights_and_apply(field.center, (0, 1)),
                res.rotate_weights_and_apply(field.center, (1, 2))]
    elif stencil.D == 3 and stencil.Q == 19:
        deriv.set_weight((0, 0, 0), sp.sympify(0))
        deriv.set_weight((1, 0, 0), sp.Rational(1, 6))

        res = deriv.get_stencil(isotropic=True)
        grad = [res.apply(field.center),
                res.rotate_weights_and_apply(field.center, (0, 1)),
                res.rotate_weights_and_apply(field.center, (1, 2))]
    elif stencil.D == 3 and stencil.Q == 27:
        deriv.set_weight((0, 0, 0), sp.sympify(0))
        deriv.set_weight((1, 0, 0), sp.Rational(2, 9))

        res = deriv.get_stencil(isotropic=True)
        grad = [res.apply(field.center),
                res.rotate_weights_and_apply(field.center, (0, 1)),
                res.rotate_weights_and_apply(field.center, (1, 2))]
    else:
        raise ValueError(f"stencil with {stencil.D} dimensions and {stencil.Q} entries is not supported")

    return grad
