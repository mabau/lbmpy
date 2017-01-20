import sympy as sp
from lbmpy.boundaries.boundaryhandling import offsetFromDir, weightOfDirection, invDir


def noSlip(pdfField, direction, latticeModel):
    neighbor = offsetFromDir(direction, latticeModel.dim)
    inverseDir = invDir(direction)
    return [sp.Eq(pdfField[neighbor](inverseDir), pdfField(direction))]


def ubb(pdfField, direction, latticeModel, velocity):
    neighbor = offsetFromDir(direction, latticeModel.dim)
    inverseDir = invDir(direction)

    velTerm = 6 * sum([d_i * v_i for d_i, v_i in zip(neighbor, velocity)]) * weightOfDirection(direction)
    return [sp.Eq(pdfField[neighbor](inverseDir),
                  pdfField(direction) - velTerm)]


def fixedDensity(pdfField, direction, latticeModel, density):
    from lbmpy_old.equilibria import standardDiscreteEquilibrium
    neighbor = offsetFromDir(direction, latticeModel.dim)
    inverseDir = invDir(direction)
    stencil = latticeModel.stencil

    if not latticeModel.compressible:
        density -= 1

    eqParams = {'stencil': stencil,
                'order': 2,
                'c_s_sq': sp.Rational(1, 3),
                'compressible': latticeModel.compressible,
                'rho': density}

    u = sp.Matrix(latticeModel.symbolicVelocity)
    symmetricEq = (standardDiscreteEquilibrium(u=u, **eqParams) + standardDiscreteEquilibrium(u=-u, **eqParams)) / 2

    subExpr1, rhoExpr, subExpr2, uExprs = getDensityVelocityExpressions(stencil,
                                                                        [pdfField(i) for i in range(len(stencil))],
                                                                        latticeModel.compressible)
    subExprs = subExpr1 + [rhoExpr] + subExpr2 + uExprs

    conditions = [(eq_i, sp.Equality(direction, i)) for i, eq_i in enumerate(symmetricEq)] + [(0, True)]
    eq_component = sp.Piecewise(*conditions)

    return subExprs + [sp.Eq(pdfField[neighbor](inverseDir),
                             2 * eq_component - pdfField(direction))]

