import sympy as sp
import numpy as np
from pystencils.backends.cbackend import CustomCppCode
from pystencils import TypedSymbol, Field
from pystencils.data_types import castFunc, createType

INV_DIR_SYMBOL = TypedSymbol("invDir", "int")
WEIGHTS_SYMBOL = TypedSymbol("weights", "double")


# -------------------------------------- Helper Functions --------------------------------------------------------------


def offsetSymbols(dim):
    return [TypedSymbol("c_%d" % (d,), createType(np.int64)) for d in range(dim)]


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
            code += "const int64_t %s [] = { %s };\n" % (offsetSym[i].name, offsetStr)

        invDirs = []
        for direction in stencil:
            inverseDir = tuple([-i for i in direction])
            invDirs.append(str(stencil.index(inverseDir)))

        code += "const int %s [] = { %s };\n" % (INV_DIR_SYMBOL.name, ", ".join(invDirs))
        weights = [str(w.evalf()) for w in lbMethod.weights]
        code += "const double %s [] = { %s };\n" % (WEIGHTS_SYMBOL.name, ",".join(weights))
        super(LbmMethodInfo, self).__init__(code, symbolsRead=set(), symbolsDefined=symbolsDefined)


def generateIndexBoundaryKernel(pdfField, indexArr, lbMethod, boundaryFunctor, target='cpu',
                                createInitializationKernel=False):
    indexField = Field.createFromNumpyArray("indexField", indexArr)
    return generateIndexBoundaryKernelGeneric(pdfField, indexField, indexArr.dtype, lbMethod, boundaryFunctor, target,
                                              createInitializationKernel)


def generateIndexBoundaryKernelGeneric(pdfField, indexField, indexArrDtype, lbMethod, boundaryFunctor, target='cpu',
                                       createInitializationKernel=False):

    elements = [LbmMethodInfo(lbMethod)]
    dirSymbol = TypedSymbol("dir", indexArrDtype.fields['dir'][0])
    boundaryEqList = [sp.Eq(dirSymbol, indexField[0]('dir'))]
    if createInitializationKernel:
        boundaryEqList += boundaryFunctor.additionalDataInitKernelEquations(pdfField=pdfField, directionSymbol=dirSymbol,
                                                                            lbMethod=lbMethod, indexField=indexField)
    else:
        boundaryEqList += boundaryFunctor(pdfField=pdfField, directionSymbol=dirSymbol, lbMethod=lbMethod,
                                          indexField=indexField)
    elements += boundaryEqList

    if target == 'cpu':
        from pystencils.cpu import createIndexedKernel
        return createIndexedKernel(elements, [indexField])
    elif target == 'gpu':
        from pystencils.gpucuda import createdIndexedCUDAKernel
        return createdIndexedCUDAKernel(elements, [indexField])