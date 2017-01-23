import sympy as sp
import numpy as np
from pystencils import TypedSymbol, Field
from pystencils.backends.cbackend import CustomCppCode
from pystencils.ast import Block, SympyAssignment, LoopOverCoordinate, KernelFunction
from pystencils.transformations import moveConstantsBeforeLoop, resolveFieldAccesses, typingFromSympyInspection, \
    typeAllEquations
from pystencils.cpu import makePythonFunction as makePythonCpuFunction
from pystencils.gpucuda import makePythonFunction as makePythonGpuFunction
from lbmpy.boundaries.createindexlist import createBoundaryIndexList

INV_DIR_SYMBOL = TypedSymbol("invDir", "int")
WEIGHTS_SYMBOL = TypedSymbol("weights", "double")


class BoundaryHandling:
    def __init__(self, symbolicPdfField, domainShape, latticeModel, ghostLayers=1, target='cpu'):
        self._symbolicPdfField = symbolicPdfField
        self._shapeWithGhostLayers = [d + 2 * ghostLayers for d in domainShape]
        self._fluidFlag = 2 ** 31
        self.flagField = np.full(self._shapeWithGhostLayers, self._fluidFlag, dtype=np.int32)
        self._ghostLayers = ghostLayers
        self._latticeModel = latticeModel
        self._boundaryFunctions = []
        self._nameToIndex = {}
        self._boundarySweeps = []
        self._target = target
        if target not in ('cpu', 'gpu'):
            raise ValueError("Invalid target '%s' . Allowed values: 'cpu' or 'gpu'" % (target,))

    def addBoundary(self, boundaryFunction, name=None):
        if name is None:
            name = boundaryFunction.__name__

        self._nameToIndex[name] = len(self._boundaryFunctions)
        self._boundaryFunctions.append(boundaryFunction)

    def invalidateIndexCache(self):
        self._boundarySweeps = []

    def clear(self):
        np.fill(self._fluidFlag)
        self.invalidateIndexCache()

    def getFlag(self, name):
        return 2 ** self._nameToIndex[name]

    def setBoundary(self, function, indexExpr, clearOtherBoundaries=True):
        if hasattr(function, '__name__'):
            name = function.__name__
        elif hasattr(function, 'name'):
            name = function.name
        else:
            raise ValueError("Boundary function has to have a '__name__' or 'name' attribute")

        if function not in self._boundaryFunctions:
            self.addBoundary(function, name)

        flag = self.getFlag(name)
        if clearOtherBoundaries:
            self.flagField[indexExpr] = flag
        else:
            # clear fluid flag
            np.bitwise_and(self.flagField[indexExpr], np.invert(self._fluidFlag), self.flagField[indexExpr])
            # add new boundary flag
            np.bitwise_or(self.flagField[indexExpr], flag, self.flagField[indexExpr])

        self.invalidateIndexCache()

    def prepare(self):
        self.invalidateIndexCache()
        for boundaryIdx, boundaryFunc in enumerate(self._boundaryFunctions):
            idxField = createBoundaryIndexList(self.flagField, self._latticeModel.stencil,
                                               2 ** boundaryIdx, self._fluidFlag, self._ghostLayers)
            ast = generateBoundaryHandling(self._symbolicPdfField, idxField, self._latticeModel, boundaryFunc)

            if self._target == 'cpu':
                self._boundarySweeps.append(makePythonCpuFunction(ast, {'indexField': idxField}))
            elif self._target == 'gpu':
                self._boundarySweeps.append(makePythonGpuFunction(ast, {'indexField': idxField}))
            else:
                assert False

    def __call__(self, **kwargs):
        if len(self._boundarySweeps) == 0:
            self.prepare()
        for boundarySweep in self._boundarySweeps:
            boundarySweep(**kwargs)


# -------------------------------------- Helper Functions --------------------------------------------------------------


def offsetSymbols(dim):
    return [TypedSymbol("c_%d" % (d,), "int") for d in range(dim)]


def offsetFromDir(dirIdx, dim):
    return tuple([sp.IndexedBase(symbol, shape=(1,))[dirIdx] for symbol in offsetSymbols(dim)])


def invDir(dirIdx):
    return sp.IndexedBase(INV_DIR_SYMBOL, shape=(1,))[dirIdx]


def weightOfDirection(dirIdx):
    return sp.IndexedBase(WEIGHTS_SYMBOL, shape=(1,))[dirIdx]


# ------------------------------------- Kernel Generation --------------------------------------------------------------

class LbmMethodInfo(CustomCppCode):
    def __init__(self, lbMethod):
        stencil = lbMethod.stencil
        symbolsDefined = set(offsetSymbols(lbMethod.dim) + [INV_DIR_SYMBOL, WEIGHTS_SYMBOL])

        offsetSym = offsetSymbols(lbMethod.dim)
        code = "\n"
        for i in range(lbMethod.dim):
            offsetStr = ", ".join([str(d[i]) for d in stencil])
            code += "const int %s [] = { %s };\n" % (offsetSym[i].name, offsetStr)

        invDirs = []
        for direction in stencil:
            inverseDir = tuple([-i for i in direction])
            invDirs.append(str(stencil.index(inverseDir)))

        code += "static const int %s [] = { %s };\n" % (INV_DIR_SYMBOL.name, ", ".join(invDirs))
        weights = [str(w.evalf()) for w in lbMethod.weights]
        code += "static const double %s [] = { %s };\n" % (WEIGHTS_SYMBOL.name, ",".join(weights))
        super(LbmMethodInfo, self).__init__(code, symbolsRead=set(), symbolsDefined=symbolsDefined)


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
    if type(boundaryEqList) is tuple:
        boundaryEqList, additionalNodes = boundaryEqList
    else:
        additionalNodes = []

    typeInfos = typingFromSympyInspection(boundaryEqList, pdfField.dtype)
    fieldsRead, fieldsWritten, assignments = typeAllEquations(boundaryEqList, typeInfos)
    fieldsAccessed = fieldsRead.union(fieldsWritten) - set([indexField])

    for be in assignments:
        cellLoopBody.append(be)

    functionBody = Block([cellLoop])
    ast = KernelFunction(functionBody, [pdfField, indexField])

    if len(additionalNodes) > 0:
        loops = ast.atoms(LoopOverCoordinate)
        assert len(loops) == 1
        loop = list(loops)[0]
        for node in additionalNodes:
            loop.body.append(node)

    functionBody.insertFront(LbmMethodInfo(latticeModel))

    fixedCoordinateMapping = {f.name: coordinateSymbols[:dim] for f in fieldsAccessed}
    resolveFieldAccesses(ast, set(['indexField']), fieldToFixedCoordinates=fixedCoordinateMapping)
    moveConstantsBeforeLoop(ast)
    return ast
