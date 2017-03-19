import sympy as sp
import numpy as np
from pystencils import TypedSymbol, Field
from pystencils.backends.cbackend import CustomCppCode
from lbmpy.boundaries.createindexlist import createBoundaryIndexList

INV_DIR_SYMBOL = TypedSymbol("invDir", "int")
WEIGHTS_SYMBOL = TypedSymbol("weights", "double")


class BoundaryHandling(object):
    def __init__(self, symbolicPdfField, domainShape, lbMethod, ghostLayers=1, target='cpu'):
        self._symbolicPdfField = symbolicPdfField
        self._shapeWithGhostLayers = [d + 2 * ghostLayers for d in domainShape]
        self._fluidFlag = 2 ** 30
        self.flagField = np.full(self._shapeWithGhostLayers, self._fluidFlag, dtype=np.int32)
        self._ghostLayers = ghostLayers
        self._lbMethod = lbMethod
        self._boundaryFunctions = []
        self._nameToIndex = {}
        self._boundarySweeps = []
        self._target = target
        if target not in ('cpu', 'gpu'):
            raise ValueError("Invalid target '%s' . Allowed values: 'cpu' or 'gpu'" % (target,))

    def addBoundary(self, boundaryFunction, name=None):
        if name is None:
            name = boundaryFunction.__name__

        if name in self._nameToIndex:
            return 2 ** self._nameToIndex[name]

        newIdx = len(self._boundaryFunctions)
        self._nameToIndex[name] = newIdx
        self._boundaryFunctions.append(boundaryFunction)
        return 2 ** newIdx

    def invalidateIndexCache(self):
        self._boundarySweeps = []

    def clear(self):
        np.fill(self._fluidFlag)
        self.invalidateIndexCache()

    def getFlag(self, name):
        return 2 ** self._nameToIndex[name]

    def setBoundary(self, function, indexExpr, maskArr=None):
        if hasattr(function, '__name__'):
            name = function.__name__
        elif hasattr(function, 'name'):
            name = function.name
        else:
            raise ValueError("Boundary function has to have a '__name__' or 'name' attribute")

        if function not in self._boundaryFunctions:
            self.addBoundary(function, name)

        flag = self.getFlag(name)

        if maskArr is None:
            self.flagField[indexExpr] = flag
        else:
            flagFieldView = self.flagField[indexExpr]
            flagFieldView[maskArr] = flag

        self.invalidateIndexCache()

    def prepare(self):
        self.invalidateIndexCache()
        for boundaryIdx, boundaryFunc in enumerate(self._boundaryFunctions):
            idxField = createBoundaryIndexList(self.flagField, self._lbMethod.stencil,
                                               2 ** boundaryIdx, self._fluidFlag, self._ghostLayers)
            ast = generateBoundaryHandling(self._symbolicPdfField, idxField, self._lbMethod, boundaryFunc,
                                           target=self._target)

            if self._target == 'cpu':
                from pystencils.cpu import makePythonFunction as makePythonCpuFunction
                self._boundarySweeps.append(makePythonCpuFunction(ast, {'indexField': idxField}))
            elif self._target == 'gpu':
                from pystencils.gpucuda import makePythonFunction as makePythonGpuFunction
                import pycuda.gpuarray as gpuarray
                idxGpuField = gpuarray.to_gpu(idxField)
                self._boundarySweeps.append(makePythonGpuFunction(ast, {'indexField': idxGpuField}))
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

        code += "const int %s [] = { %s };\n" % (INV_DIR_SYMBOL.name, ", ".join(invDirs))
        weights = [str(w.evalf()) for w in lbMethod.weights]
        code += "const double %s [] = { %s };\n" % (WEIGHTS_SYMBOL.name, ",".join(weights))
        super(LbmMethodInfo, self).__init__(code, symbolsRead=set(), symbolsDefined=symbolsDefined)


def generateBoundaryHandling(pdfField, indexArr, lbMethod, boundaryFunctor, target='cpu'):
    indexField = Field.createFromNumpyArray("indexField", indexArr)

    elements = [LbmMethodInfo(lbMethod)]
    dirSymbol = TypedSymbol("dir", indexArr.dtype.fields['dir'][0])
    boundaryEqList = [sp.Eq(dirSymbol, indexField[0]('dir'))]
    boundaryEqList += boundaryFunctor(pdfField, dirSymbol, lbMethod)
    if type(boundaryEqList) is tuple:
        boundaryEqList, additionalNodes = boundaryEqList
        elements += boundaryEqList
        elements += additionalNodes
    else:
        elements += boundaryEqList

    if target == 'cpu':
        from pystencils.cpu import createIndexedKernel
        return createIndexedKernel(elements, [indexField])
    elif target == 'gpu':
        from pystencils.gpucuda import createdIndexedCUDAKernel
        return createdIndexedCUDAKernel(elements, [indexField])

