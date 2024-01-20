import sympy as sp


def second_order_moment_tensor(function_values, stencil):
    """Returns (D x D) Matrix of second order moments of the given function where D is the dimension"""
    assert len(function_values) == stencil.Q
    return sp.Matrix(stencil.D, stencil.D, lambda i, j: sum(c[i] * c[j] * f for f, c in zip(function_values, stencil)))


def frobenius_norm(matrix, factor=1):
    """Computes the Frobenius norm of a matrix defined as the square root of the sum of squared matrix elements
    The optional factor is added inside the square root"""
    return sp.sqrt(sum(i * i for i in matrix) * factor)


def extract_shear_relaxation_rate(collision_rule, shear_relaxation_rate):
    """Searches for the shear relaxation rate in the collision equations.
       If the shear relaxation rate is assigned to a single assignment its lhs is returned.
       Otherwise, the shear relaxation rate has to be a sympy symbol, or it can not be extracted"""
    found_symbolic_shear_relaxation = True
    if isinstance(shear_relaxation_rate, (float, int)):
        found_symbolic_shear_relaxation = False
        for eq in collision_rule.all_assignments:
            if eq.rhs == shear_relaxation_rate:
                found_symbolic_shear_relaxation = True
                shear_relaxation_rate = eq.lhs

    return shear_relaxation_rate, found_symbolic_shear_relaxation
