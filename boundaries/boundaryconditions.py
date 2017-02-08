import sympy as sp

from lbmpy.simplificationfactory import createSimplificationStrategy
from pystencils.sympyextensions import getSymmetricPart
from lbmpy.boundaries.boundaryhandling import offsetFromDir, weightOfDirection, invDir


def noSlip(pdfField, direction, lbMethod):
    """No-Slip, simple bounce back boundary condition, enforcing zero velocity at obstacle"""
    neighbor = offsetFromDir(direction, lbMethod.dim)
    inverseDir = invDir(direction)
    return [sp.Eq(pdfField[neighbor](inverseDir), pdfField(direction))]


def ubb(pdfField, direction, lbMethod, velocity):
    """Velocity bounce back boundary condition, enforcing specified velocity at obstacle"""

    assert len(velocity) == lbMethod.dim, \
        "Dimension of velocity (%d) does not match dimension of LB method (%d)" % (len(velocity), lbMethod.dim)
    neighbor = offsetFromDir(direction, lbMethod.dim)
    inverseDir = invDir(direction)

    # TODO adapt velocity to force
    c_s_sq = sp.Rational(1, 3)
    velTerm = 2 / c_s_sq * sum([d_i * v_i for d_i, v_i in zip(neighbor, velocity)]) * weightOfDirection(direction)

    # in conserved value computation
    # rename what is currently called density to "virtualDensity"
    # provide a new quantity density, which is constant in case of incompressible models
    if lbMethod.conservedQuantityComputation._compressible:  # TODO
        cqc = lbMethod.conservedQuantityComputation
        densitySymbol = sp.Symbol("rho")
        pdfFieldAccesses = [pdfField(i) for i in range(len(lbMethod.stencil))]
        densityEquations = cqc.outputEquationsFromPdfs(pdfFieldAccesses, {'density': densitySymbol})
        densitySymbol = lbMethod.conservedQuantityComputation.definedSymbols()['density']
        result = densityEquations.allEquations
        result += [sp.Eq(pdfField[neighbor](inverseDir),
                         pdfField(direction) - velTerm * densitySymbol)]
        return result
    else:
        return [sp.Eq(pdfField[neighbor](inverseDir),
                      pdfField(direction) - velTerm)]


def fixedDensity(pdfField, direction, lbMethod, density):
    """Boundary condition that fixes the density/pressure at the obstacle"""

    def removeAsymmetricPartOfMainEquations(eqColl, dofs):
        newMainEquations = [sp.Eq(e.lhs, getSymmetricPart(e.rhs, dofs)) for e in eqColl.mainEquations]
        return eqColl.copy(newMainEquations)

    neighbor = offsetFromDir(direction, lbMethod.dim)
    inverseDir = invDir(direction)

    velocity = lbMethod.conservedQuantityComputation.definedSymbols()['velocity']
    symmetricEq = removeAsymmetricPartOfMainEquations(lbMethod.getEquilibrium(), dofs=velocity)
    substitutions = {sym: pdfField(i) for i, sym in enumerate(lbMethod.preCollisionPdfSymbols)}
    symmetricEq = symmetricEq.copyWithSubstitutionsApplied(substitutions)

    simplification = createSimplificationStrategy(lbMethod)
    symmetricEq = simplification(symmetricEq)

    densitySymbol = lbMethod.conservedQuantityComputation.definedSymbols()['density']

    conditions = [(eq_i.rhs, sp.Equality(direction, i))
                  for i, eq_i in enumerate(symmetricEq.mainEquations)] + [(0, True)]
    eq_component = sp.Piecewise(*conditions)

    subExprs = [sp.Eq(eq.lhs, density if eq.lhs == densitySymbol else eq.rhs) for eq in symmetricEq.subexpressions]
    return subExprs + [sp.Eq(pdfField[neighbor](inverseDir), 2 * eq_component - pdfField(direction))]

