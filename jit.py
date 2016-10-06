from ctypes import cdll, Structure, c_long, c_double, c_float, POINTER, byref, sizeof
import numpy as np
import lbmpy.generator as gen
import subprocess
import os
from tempfile import TemporaryDirectory

CONFIG = {
    'compiler': 'g++',
    'flags': '-Ofast -DNDEBUG -fPIC -shared -march=native -fopenmp',
}


def ctypeFromString(typename, includePointers=True):
    import ctypes as ct

    typename = typename.replace("*", " * ")
    typeComponents = typename.split()

    basicTypeMap = {
        'double': ct.c_double,
        'float': ct.c_float,
        'int': ct.c_int,
        'long': ct.c_long,
    }

    resultType = None
    for typeComponent in typeComponents:
        typeComponent = typeComponent.strip()
        if typeComponent == "const" or typeComponent == "restrict" or typeComponent == "volatile":
            continue
        if typeComponent in basicTypeMap:
            resultType = basicTypeMap[typeComponent]
        elif typeComponent == "*" and includePointers:
            assert resultType is not None
            resultType = ct.POINTER(resultType)

    return resultType


def ctypeFromNumpyType(numpyType):
    typeMap = {
        np.dtype('float64'): c_double,
        np.dtype('float32'): c_float,
    }
    return typeMap[numpyType]


def compileAndLoad(kernelFunctionNode):
    with TemporaryDirectory() as tmpDir:
        srcFile = os.path.join(tmpDir, 'source.c')
        with open(srcFile, 'w') as sourceFile:
            print('extern "C" { ', file=sourceFile)
            print(kernelFunctionNode.generateC(), file=sourceFile)
            print('}', file=sourceFile)

        compilerCmd = [CONFIG['compiler']] + CONFIG['flags'].split()
        libFile = os.path.join(tmpDir, "jit.so")
        compilerCmd += [srcFile, '-o', libFile]
        subprocess.call(compilerCmd)
        loadedJitLib = cdll.LoadLibrary(libFile)

    return loadedJitLib


def buildCTypeArgumentList(kernelFunctionNode, argumentDict):
    ctArguments = []
    for arg in kernelFunctionNode.parameters:
        if arg.isFieldArgument:
            field = argumentDict[arg.fieldName]
            if arg.isFieldPtrArgument:
                ctArguments.append(field.ctypes.data_as(ctypeFromString(arg.dtype)))
            elif arg.isFieldShapeArgument:
                dataType = ctypeFromString(gen.SHAPE_DTYPE, includePointers=False)
                ctArguments.append(field.ctypes.shape_as(dataType))
            elif arg.isFieldStrideArgument:
                dataType = ctypeFromString(gen.STRIDE_DTYPE, includePointers=False)
                baseFieldType = ctypeFromNumpyType(field.dtype)
                strides = field.ctypes.strides_as(dataType)
                for i in range(len(field.shape)):
                    assert strides[i] % sizeof(baseFieldType) == 0
                    strides[i] //= sizeof(baseFieldType)
                ctArguments.append(strides)
            else:
                assert False
        else:
            param = argumentDict[arg.name]
            expectedType = ctypeFromString(arg.dtype)
            ctArguments.append(expectedType(param))
    return ctArguments


def makePythonFunction(kernelFunctionNode, argumentDict):
    # build up list of CType arguments
    args = buildCTypeArgumentList(kernelFunctionNode, argumentDict)
    func = compileAndLoad(kernelFunctionNode)[kernelFunctionNode.functionName]
    func.restype = None
    return lambda: func(*args)


def makeWalberlaSourceDestinationSweep(kernelFunctionNode, sourceFieldName='src', destinationFieldName='dst'):
    from waLBerla import field

    swapFields = {}

    func = compileAndLoad(kernelFunctionNode)[kernelFunctionNode.functionName]
    func.restype = None

    # Counting the number of domain loops to get dimensionality
    dim = len(kernelFunctionNode.atoms(gen.LoopOverCoordinate))

    def f(**kwargs):
        src = kwargs[sourceFieldName]
        sizeInfo = (src.size, src.allocSize, src.layout)
        if sizeInfo not in swapFields:
            swapFields[sizeInfo] = src.cloneUninitialized()
        dst = swapFields[sizeInfo]

        kwargs[sourceFieldName] = field.toArray(src, withGhostLayers=True)
        kwargs[destinationFieldName] = field.toArray(dst, withGhostLayers=True)

        # Since waLBerla does not really support 2D domains a small hack is required here
        if dim == 2:
            assert kwargs[sourceFieldName].shape[2] in [1, 3]
            assert kwargs[destinationFieldName].shape[2] in [1, 3]
            kwargs[sourceFieldName] = kwargs[sourceFieldName][:, :, 1, :]
            kwargs[destinationFieldName] = kwargs[destinationFieldName][:, :, 1, :]

        args = buildCTypeArgumentList(kernelFunctionNode, kwargs)
        func(*args)
        src.swapDataPointers(dst)

    return f



