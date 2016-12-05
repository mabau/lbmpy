import itertools

import numpy as np
import sympy as sp

from lbmpy.densityVelocityExpressions import getDensityVelocityExpressions
from lbmpy.stencils import getStencil
from pystencils.backends.cbackend import CustomCppCode
from pystencils.types import TypedSymbol
from pystencils.field import Field
from pystencils.ast import Node, Block, SympyAssignment, LoopOverCoordinate, KernelFunction
from pystencils.transformations import moveConstantsBeforeLoop, resolveFieldAccesses, typingFromSympyInspection, \
    typeAllEquations

INV_DIR_SYMBOL = TypedSymbol("invDir", "int")
WEIGHTS_SYMBOL = TypedSymbol("weights", "double")


def offsetSymbols(dim):
    return [TypedSymbol("c_%d" % (d,), "int") for d in range(dim)]


def offsetFromDir(dirIdx, dim):
    return tuple([sp.IndexedBase(symbol, shape=(1,))[dirIdx] for symbol in offsetSymbols(dim)])


def invDir(dirIdx):
    return sp.IndexedBase(INV_DIR_SYMBOL, shape=(1,))[dirIdx]


def weightOfDirection(dirIdx):
    return sp.IndexedBase(WEIGHTS_SYMBOL, shape=(1,))[dirIdx]


class LatticeModelInfo(CustomCppCode):
    def __init__(self, latticeModel):
        stencil = latticeModel.stencil
        symbolsDefined = set(offsetSymbols(latticeModel.dim) + [INV_DIR_SYMBOL, WEIGHTS_SYMBOL])

        offsetSym = offsetSymbols(latticeModel.dim)
        code = "\n"
        for i in range(latticeModel.dim):
            offsetStr = ", ".join([str(d[i]) for d in stencil])
            code += "const int %s [] = { %s };\n" % (offsetSym[i].name, offsetStr)

        invDirs = []
        for direction in stencil:
            inverseDir = tuple([-i for i in direction])
            invDirs.append(str(stencil.index(inverseDir)))

        code += "static const int %s [] = { %s };\n" % (INV_DIR_SYMBOL.name, ", ".join(invDirs))
        weights = [str(w.evalf()) for w in latticeModel.weights]
        code += "static const double %s [] = { %s };\n" % (WEIGHTS_SYMBOL.name, ",".join(weights))
        super(LatticeModelInfo, self).__init__(code, symbolsRead=set(), symbolsDefined=symbolsDefined)


def generateBoundaryHandling(pdfField, indexArr, latticeModel, boundaryFunctor):
    dim = latticeModel.dim

    cellLoopBody = Block([])
    cellLoop = LoopOverCoordinate(cellLoopBody, coordinateToLoopOver=0, start=0, stop=indexArr.shape[0])

    indexField = Field.createFromNumpyArray("indexField", indexArr, indexDimensions=1)

    coordinateSymbols = [TypedSymbol(name, "int") for name in ['x', 'y', 'z']]
    for d in range(dim):
        cellLoopBody.append(SympyAssignment(coordinateSymbols[d], indexField[0](d)))
    dirSymbol = TypedSymbol("dir", "int")
    cellLoopBody.append(SympyAssignment(dirSymbol, indexField[0](dim)))

    boundaryEqList = boundaryFunctor(pdfField, dirSymbol, latticeModel)
    typeInfos = typingFromSympyInspection(boundaryEqList, pdfField.dtype)
    fieldsRead, fieldsWritten, assignments = typeAllEquations(boundaryEqList, typeInfos)

    for be in assignments:
        cellLoopBody.append(be)

    functionBody = Block([cellLoop])
    ast = KernelFunction(functionBody, [pdfField, indexField])

    functionBody.insertFront(LatticeModelInfo(latticeModel))
    resolveFieldAccesses(ast, set(['indexField']), fieldToFixedCoordinates={pdfField.name: coordinateSymbols[:dim]})
    moveConstantsBeforeLoop(ast)
    return ast


def createBoundaryIndexList(flagFieldArr, nrOfGhostLayers, stencil, boundaryMask, fluidMask):
    result = []

    gl = nrOfGhostLayers
    for cell in itertools.product(*[range(gl, i-gl) for i in flagFieldArr.shape]):
        if not flagFieldArr[cell] & fluidMask:
            continue
        for dirIdx, direction in enumerate(stencil):
            neighborCell = tuple([cell_i + dir_i for cell_i, dir_i in zip(cell, direction)])
            if flagFieldArr[neighborCell] & boundaryMask:
                result.append(list(cell) + [dirIdx])

    return np.array(result, dtype=np.int32)


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
    from lbmpy.equilibria import standardDiscreteEquilibrium
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

if __name__ == "__main__":
    from lbmpy.latticemodel import makeSRT
    from pystencils.cpu import generateC
    import functools
    lm = makeSRT(getStencil("D3Q19"))
    pdfField = Field.createGeneric("pdfField", lm.dim, indexDimensions=1)

    indexArr = np.array([[1, 1, 1, 1], [1, 2, 1,  1], [2, 1, 1, 1]], dtype=np.int32)

    pressureBoundary = functools.partial(fixedDensity, density=1.0)
    ast = generateBoundaryHandling(pdfField, indexArr, lm, pressureBoundary)

    print(generateC(ast))

