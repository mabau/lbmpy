"""
Method to construct a quadratic equilibrium using a generic quadratic ansatz according to the book of 
Wolf-Gladrow, section 5.4
"""

import sympy as sp
import numpy as np
from pystencils.sympyextensions import scalarProduct
from lbmpy.moments import discreteMoment
from lbmpy.maxwellian_equilibrium import compressibleToIncompressibleMomentValue


def genericEquilibriumAnsatz(stencil, u=sp.symbols("u_:3")):
    """Returns a generic quadratic equilibrium with coefficients A, B, C, D according to 
    Wolf Gladrow Book equation (5.4.1) """
    dim = len(stencil[0])
    u = u[:dim]

    equilibrium = []

    for direction in stencil:
        speed = np.abs(direction).sum()
        weight, linear, mixQuadratic, quadratic = getParameterSymbols(speed)
        uTimesD = scalarProduct(u, direction)
        eq = weight + linear * uTimesD + mixQuadratic * uTimesD ** 2 + quadratic * scalarProduct(u, u)
        equilibrium.append(eq)
    return tuple(equilibrium)


def matchGenericEquilibriumAnsatz(stencil, equilibrium, u=sp.symbols("u_:3")):
    """Given a quadratic equilibrium, the generic coefficients A,B,C,D are determined. 
    Returns a dict that maps these coefficients to their values. If the equilibrium does not have a
    generic quadratic form, a ValueError is raised"""
    dim = len(stencil[0])
    u = u[:dim]

    result = dict()
    for direction, actualEquilibrium in zip(stencil, equilibrium):
        speed = np.abs(direction).sum()
        A, B, C, D = getParameterSymbols(speed)
        uTimesD = scalarProduct(u, direction)
        genericEquation = A + B * uTimesD + C * uTimesD ** 2 + D * scalarProduct(u, u)

        equations = sp.poly(actualEquilibrium - genericEquation, *u).coeffs()
        solveRes = sp.solve(equations, [A, B, C, D])
        if not solveRes:
            raise ValueError("This equilibrium does not match the generic quadratic standard form")
        for dof, value in solveRes.items():
            if dof in result and result[dof] != value:
                raise ValueError("This equilibrium does not match the generic quadratic standard form")
            result[dof] = value

    return result


def momentConstraintEquations(stencil, equilibrium, momentToValueDict, u=sp.symbols("u_:3")):
    """Returns a set of equations that have to be fulfilled for a generic equilibrium match moment conditions 
    passed in momentToValueDict. This dict is expected to map moment tuples to values."""
    dim = len(stencil[0])
    u = u[:dim]
    constraintEquations = set()
    for moment, desiredValue in momentToValueDict.items():
        genericMoment = discreteMoment(equilibrium, moment, stencil)
        equations = sp.poly(genericMoment - desiredValue, *u).coeffs()
        constraintEquations.update(equations)
    return constraintEquations


def hydrodynamicMomentValues(upToOrder=3, dim=3, compressible=True):
    """Returns the values of moments that are required to approximate Navier Stokes (if upToOrder=3)"""
    from lbmpy.maxwellian_equilibrium import getMomentsOfContinuousMaxwellianEquilibrium
    from lbmpy.moments import momentsUpToOrder

    moms = momentsUpToOrder(upToOrder, dim)
    c_s_sq = sp.Symbol("p") / sp.Symbol("rho")
    momValues = getMomentsOfContinuousMaxwellianEquilibrium(moms, dim=dim, c_s_sq=c_s_sq, order=2)
    if not compressible:
        momValues = [compressibleToIncompressibleMomentValue(m, sp.Symbol("rho"), sp.symbols("u_:3")[:dim])
                     for m in momValues]

    return {a: b.expand() for a, b in zip(moms, momValues)}


def getParameterSymbols(i):
    return sp.symbols("A_%d B_%d C_%d D_%d" % (i, i, i, i))
