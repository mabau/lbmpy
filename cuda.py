import sympy as sp

from generator import TypedSymbol
from lbmpy.generator import Field, SympyAssignment, typeAllEquations, Block, KernelFunction, \
    FIELD_PTR_PREFIX, BASE_PTR_PREFIX
import cgen as c
from sympy.tensor import IndexedBase
from collections import defaultdict

BLOCK_IDX = list(sp.symbols("blockIdx.x blockIdx.y blockIdx.z"))
THREAD_IDX = list(sp.symbols("threadIdx.x threadIdx.y threadIdx.z"))


"""
GPU Access Patterns

- knows about the iteration range
- know about mapping of field indices to CUDA block and thread indices
- iterates over spatial coordinates - constructed with a specific number of coordinates
- can
"""
#class LinewiseGPUAccessPattern:
#
#    @staticmethod
#    def iterateOverFullArray(array, nrOfGhostLayers=0, indexCoordinates=0):
#        numSpatialCoordinates = len(array.shape) - indexCoordinates
#        assert numSpatialCoordinates >= 1, "At least one spatial coordinate required"
#        gl = nrOfGhostLayers
#        strides = [s / array.itemsize for s in array.strides]
#        return LinewiseGPUAccessPattern([(gl, coordSize-gl) for coordSize in array.shape[:numSpatialCoordinates]],
#                                        strides)
#
#    def __init__(self, iterationRange, strides):
#        assert len(iterationRange) <= len(strides)
#        assert len(iterationRange) <= 4, "This access pattern supports only up to 4 dimensional iteration domains"
#        self._iterationRange = iterationRange
#        self._numSpatialCoordinates = len(iterationRange)
#        self._numIndexCoordinates = len(strides)
#        self._fastestCoordinate = strides.index(min(strides))
#
#        # one thread block for the space of the fastest coordinate
#        mapping = [THREAD_IDX[0]] + BLOCK_IDX
#        mapping[0], mapping[self._fastestCoordinate] = mapping[self._fastestCoordinate], mapping[0]
#        self._mapping = mapping[:len(strides)]
#
#    def getCoordinateAccessExpression(self, spatialCoordinate, indexCoordinate=()):
#        """Returns a sympy expression for accessing the given coordinate in a field with given strides"""
#        assert len(spatialCoordinate) == self._numSpatialCoordinates
#        assert len(indexCoordinate) == self._numIndexCoordinates
#        offset = sum([stride * off[0] for stride, off in zip(self._strides[:self._numSpatialCoordinates],
#                                                             self._iterationRange)])
#        return offset + sum([stride * cudaIdx for stride, cudaIdx in zip(self._strides, self._mapping)])
#
#    def kernelCallParameters(self):
#        r = [item[1] - item[0] for item in self._iterationRange]
#        return {'block': (r[self._fastestCoordinate], 1, 1),
#                'grid': tuple(r[1:])}


def getLinewiseCoordinateAccessExpression(field, indexCoordinate):
    availableIndices = [THREAD_IDX[0]] + BLOCK_IDX
    fastestCoordinate = field.layout[-1]
    availableIndices[fastestCoordinate], availableIndices[0] = availableIndices[0], availableIndices[fastestCoordinate]
    cudaIndices = availableIndices[:field.spatialDimensions]

    offsetToCell = sum([cudaIdx*stride for cudaIdx, stride in zip(cudaIndices, field.spatialStrides)])
    indexOffset = sum([idx * indexStride for idx, indexStride in zip(indexCoordinate, field.indexStrides)])
    return sp.simplify(offsetToCell + indexOffset)


def resolveFieldAccesses(ast):
    """Substitutes FieldAccess nodes by array indexing"""

    def visitSympyExpr(expr, enclosingBlock):
        if isinstance(expr, Field.Access):
            fieldAccess = expr
            field = fieldAccess.field

            dtype = "%s * __restrict__" % field.dtype
            if field.readOnly:
                dtype = "const " + dtype

            fieldPtr = TypedSymbol("%s%s" % (FIELD_PTR_PREFIX, field.name), dtype)
            idxStr = "_".join([str(i) for i in fieldAccess.index])
            basePtr = TypedSymbol("%s%s_%s" % (BASE_PTR_PREFIX, field.name, idxStr),
                                  dtype)
            baseArr = IndexedBase(basePtr, shape=(1,))

            offset = getLinewiseCoordinateAccessExpression(field, fieldAccess.index)

            if basePtr not in enclosingBlock.symbolsDefined:
                enclosingBlock.insertFront(SympyAssignment(basePtr, fieldPtr + offset, isConst=False))

            for i in range(field.indexDimensions):
                offset += field.indexStrides[i] * fieldAccess.index[i]

            spatialOffset = sum([offset * stride for offset, stride in zip(fieldAccess.offsets, field.spatialStrides)])
            return baseArr[spatialOffset]
        else:
            newArgs = [visitSympyExpr(e, enclosingBlock) for e in expr.args]
            kwargs = {'evaluate': False} if type(expr) == sp.Add or type(expr) == sp.Mul else {}
            return expr.func(*newArgs, **kwargs) if newArgs else expr

    def visitNode(subAst, enclosingBlock):
        if isinstance(subAst, SympyAssignment):
            subAst.lhs = visitSympyExpr(subAst.lhs, enclosingBlock)
            subAst.rhs = visitSympyExpr(subAst.rhs, enclosingBlock)
        else:
            for i, a in enumerate(subAst.args):
                visitNode(a, subAst if isinstance(subAst, Block) else enclosingBlock)

    return visitNode(ast, None)


def createCUDAKernel(listOfEquations, functionName="kernel", typeForSymbol=defaultdict(lambda: "double")):
    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    for f in fieldsRead - fieldsWritten:
        f.setReadOnly()

    code = KernelFunction(Block(assignments), functionName)
    code.qualifierPrefix = "__global__ "
    code.variablesToIgnore.update(BLOCK_IDX + THREAD_IDX)
    resolveFieldAccesses(code)
    return code


if __name__ == "__main__":
    import sympy as sp
    from lbmpy.stencils import getStencil
    from lbmpy.collisionoperator import makeSRT
    from lbmpy.lbmgenerator import createLbmEquations

    latticeModel = makeSRT(getStencil("D2Q9"), order=2, compressible=False)
    r = createLbmEquations(latticeModel, doCSE=True)
    kernel = createCUDAKernel(r)
    print(kernel.generateC())

    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    mod = SourceModule(str(kernel.generateC()))
    func = mod.get_function("kernel")