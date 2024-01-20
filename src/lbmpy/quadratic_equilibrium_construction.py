"""
Method to construct a quadratic equilibrium using a generic quadratic ansatz according to the book of
Wolf-Gladrow, section 5.4
"""

import numpy as np
import sympy as sp

from lbmpy.maxwellian_equilibrium import compressible_to_incompressible_moment_value
from lbmpy.moments import discrete_moment
from pystencils.sympyextensions import scalar_product


def generic_equilibrium_ansatz(stencil, u=sp.symbols("u_:3")):
    """Returns a generic quadratic equilibrium with coefficients A, B, C, D according to
    Wolf Gladrow Book equation (5.4.1) """
    dim = len(stencil[0])
    u = u[:dim]

    equilibrium = []

    for direction in stencil:
        speed = np.abs(direction).sum()
        weight, linear, mix_quadratic, quadratic = get_parameter_symbols(speed)
        u_times_d = scalar_product(u, direction)
        eq = weight + linear * u_times_d + mix_quadratic * u_times_d ** 2 + quadratic * scalar_product(u, u)
        equilibrium.append(eq)
    return tuple(equilibrium)


def generic_equilibrium_ansatz_parameters(stencil):
    degrees_of_freedom = set()
    for direction in stencil:
        speed = np.abs(direction).sum()
        params = get_parameter_symbols(speed)
        degrees_of_freedom.update(params)
    degrees_of_freedom.add(sp.Symbol("p"))
    return sorted(list(degrees_of_freedom), key=lambda e: e.name)


def match_generic_equilibrium_ansatz(stencil, equilibrium, u=sp.symbols("u_:3")):
    """Given a quadratic equilibrium, the generic coefficients A,B,C,D are determined.

    Returns:
        dict that maps these coefficients to their values. If the equilibrium does not have a
        generic quadratic form, a ValueError is raised

    Example:
          >>> from lbmpy import LBStencil, Stencil
          >>> from lbmpy.maxwellian_equilibrium import discrete_maxwellian_equilibrium
          >>> stencil = LBStencil(Stencil.D2Q9)
          >>> eq = discrete_maxwellian_equilibrium(stencil)
          >>> result = match_generic_equilibrium_ansatz(stencil, eq)
          >>> result[sp.Symbol('A_0')]
          4*rho/9
          >>> result[sp.Symbol('B_1')]
          rho/(9*c_s**2)
    """
    dim = len(stencil[0])
    u = u[:dim]

    result = dict()
    for direction, actual_equilibrium in zip(stencil, equilibrium):
        speed = np.abs(direction).sum()
        a, b, c, d = get_parameter_symbols(speed)
        u_times_d = scalar_product(u, direction)
        generic_equation = a + b * u_times_d + c * u_times_d ** 2 + d * scalar_product(u, u)

        equations = sp.poly(actual_equilibrium - generic_equation, *u).coeffs()
        solve_res = sp.solve(equations, [a, b, c, d])
        if not solve_res:
            raise ValueError("This equilibrium does not match the generic quadratic standard form")
        for dof, value in solve_res.items():
            if dof in result and result[dof] != value:
                raise ValueError("This equilibrium does not match the generic quadratic standard form")
            result[dof] = value

    return result


def moment_constraint_equations(stencil, equilibrium, moment_to_value_dict, u=sp.symbols("u_:3")):
    """Returns a set of equations that have to be fulfilled for a generic equilibrium to match moment conditions
    passed in moment_to_value_dict. This dict is expected to map moment tuples to values."""
    dim = len(stencil[0])
    u = u[:dim]
    equilibrium = tuple(equilibrium)
    constraint_equations = set()
    for moment, desired_value in moment_to_value_dict.items():
        generic_moment = discrete_moment(equilibrium, moment, stencil)
        equations = sp.poly(generic_moment - desired_value, *u).coeffs()
        constraint_equations.update(equations)
    return list(constraint_equations)


def hydrodynamic_moment_values(up_to_order=3, dim=3, compressible=True):
    """Returns the values of moments that are required to approximate Navier Stokes (if up_to_order=3)"""
    from lbmpy.maxwellian_equilibrium import get_equilibrium_values_of_maxwell_boltzmann_function
    from lbmpy.moments import moments_up_to_order

    moms = moments_up_to_order(up_to_order, dim)
    c_s_sq = sp.Symbol("p") / sp.Symbol("rho")
    moment_values = get_equilibrium_values_of_maxwell_boltzmann_function(moms, dim=dim, c_s_sq=c_s_sq, order=2,
                                                                         space="moment")
    if not compressible:
        moment_values = [compressible_to_incompressible_moment_value(m, sp.Symbol("rho"), sp.symbols("u_:3")[:dim])
                         for m in moment_values]

    return {a: b.expand() for a, b in zip(moms, moment_values)}


def get_parameter_symbols(i):
    return sp.symbols("A_%d B_%d C_%d D_%d" % (i, i, i, i))
