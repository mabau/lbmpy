import itertools

import cgen as c
import numpy as np
import sympy as sp

import pystencils.generator as gen
from lbmpy.stencils import getStencil

INV_DIR_SYMBOL = gen.TypedSymbol("invDir", "int")
WEIGHTS_SYMBOL = gen.TypedSymbol("weights", "double")


def offsetSymbols(dim):
    return [gen.TypedSymbol("c_%d" % (d,), "int") for d in range(dim)]


def offsetFromDir(dirIdx, dim):
    return tuple([sp.IndexedBase(symbol, shape=(1,))[dirIdx] for symbol in offsetSymbols(dim)])


def invDir(dirIdx):
    return sp.IndexedBase(INV_DIR_SYMBOL, shape=(1,))[dirIdx]


def weightOfDirection(dirIdx):
    return sp.IndexedBase(WEIGHTS_SYMBOL, shape=(1,))[dirIdx]


class LatticeModelInfo(gen.Node):
    def __init__(self, latticeModel):
        self._stencil = latticeModel.stencil
        self._lm = latticeModel

    @property
    def symbolsDefined(self):
        return set(offsetSymbols(self._lm.dim) + [INV_DIR_SYMBOL, WEIGHTS_SYMBOL])

    @property
    def symbolsRead(self):
        return set()

    @property
    def args(self):
        return []

    def generateC(self):
        offsetSym = offsetSymbols(self._lm.dim)
        lines = "\n"
        for i in range(self._lm.dim):
            offsetStr = ", ".join([str(d[i]) for d in self._stencil])
            lines += "const int %s [] = { %s };\n" % (offsetSym[i].name, offsetStr)

        invDirs = []
        for direction in self._stencil:
            inverseDir = tuple([-i for i in direction])
            invDirs.append(str(self._stencil.index(inverseDir)))

        lines += "static const int %s [] = { %s };\n" % (INV_DIR_SYMBOL.name, ", ".join(invDirs))
        weights = [str(w.evalf()) for w in self._lm.weights]
        lines += "static const double %s [] = { %s };\n" % (WEIGHTS_SYMBOL.name, ",".join(weights))
        return c.LiteralLines(lines)


def generateBoundaryHandling(pdfField, indexArr, latticeModel, boundaryFunctor):
    dim = latticeModel.dim

    cellLoopBody = gen.Block([])
    cellLoop = gen.LoopOverCoordinate(cellLoopBody, 0, indexArr.shape, increment=1, ghostLayers=0)

    indexField = gen.Field.createFromNumpyArray("indexField", indexArr, indexDimensions=1)

    indexField.setReadOnly()
    coordinateSymbols = [gen.TypedSymbol(name, "int") for name in ['x', 'y', 'z']]
    for d in range(dim):
        cellLoopBody.append(gen.SympyAssignment(coordinateSymbols[d], indexField[0](d)))
    dirSymbol = gen.TypedSymbol("dir", "int")
    cellLoopBody.append(gen.SympyAssignment(dirSymbol, indexField[0](dim)))

    cellLoopBody.append(boundaryFunctor(pdfField, dirSymbol, latticeModel))

    functionBody = gen.Block([cellLoop])
    ast = gen.KernelFunction(functionBody)

    functionBody.insertFront(LatticeModelInfo(latticeModel))
    gen.resolveFieldAccesses(ast, fieldToFixedCoordinates={pdfField.name: coordinateSymbols[:dim]})
    gen.moveConstantsBeforeLoop(ast)
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
    current = tuple([0] * latticeModel.dim)
    inverseDir = invDir(direction)
    return gen.SympyAssignment(pdfField[neighbor](inverseDir), pdfField[current](direction))


def ubb(pdfField, direction, latticeModel, velocity):
    neighbor = offsetFromDir(direction, latticeModel.dim)
    current = tuple([0] * latticeModel.dim)
    inverseDir = invDir(direction)

    velTerm = 6 * sum([d_i * v_i for d_i, v_i in zip(neighbor, velocity)]) * weightOfDirection(direction)
    return gen.SympyAssignment(pdfField[neighbor](inverseDir),
                               pdfField[current](direction) - velTerm)


if __name__ == "__main__":
    import lbmpy.collisionoperator as coll
    lm = coll.makeSRT(getStencil("D3Q19"))
    pdfField = gen.Field.createGeneric("pdfField", lm.dim, indexDimensions=1)

    indexArr = np.array([[1, 1, 1, 1], [1, 2, 1,  1], [2, 1, 1, 1]], dtype=np.int32)

    ast = generateBoundaryHandling(pdfField, indexArr, lm, noSlip)

    print(ast.generateC())

