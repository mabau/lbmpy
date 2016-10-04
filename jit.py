from ctypes import cdll, Structure, c_long, c_double, c_float, POINTER, byref, sizeof
import numpy as np
import lbmpy.generator as gen
import subprocess
import os
from tempfile import TemporaryDirectory

CONFIG = {
    'compiler': 'g++',
    'flags': '-Ofast -DNDEBUG -fPIC -shared -march=native',
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


def makePythonFunction(kernelFunctionNode, argumentDict):

    # build up list of ctype arguments
    ctypeArguments = []
    for arg in kernelFunctionNode.parameters:
        if arg.isFieldArgument:
            field = argumentDict[arg.fieldName]
            if arg.isFieldPtrArgument:
                ctypeArguments.append(field.ctypes.data_as(ctypeFromString(arg.dtype)))
            elif arg.isFieldShapeArgument:
                dtype = ctypeFromString(gen.SHAPE_DTYPE, includePointers=False)
                ctypeArguments.append(field.ctypes.shape_as(dtype))
            elif arg.isFieldStrideArgument:
                dtype = ctypeFromString(gen.STRIDE_DTYPE, includePointers=False)
                baseFieldType = ctypeFromNumpyType(field.dtype)
                strides = field.ctypes.strides_as(dtype)
                for i in range(len(field.shape)):
                    assert strides[i] % sizeof(baseFieldType) == 0
                    strides[i] //= sizeof(baseFieldType)
                ctypeArguments.append(strides)
            else:
                assert False
        else:
            param = argumentDict[arg.name]
            expectedType = ctypeFromString(arg.dtype)
            ctypeArguments.append(expectedType(param))

    dll = compileAndLoad(kernelFunctionNode)
    cfunc = dll[kernelFunctionNode.functionName]
    cfunc.restype = None

    return lambda: cfunc(*ctypeArguments)


def compileAndLoad(kernelFunctionNode):

    loadedJitLib = None
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


def fieldFromArr(arr):
    typeMap = {
        np.dtype('float64'): c_double,
        np.dtype('float32'): c_float,
    }

    baseFieldType = typeMap[arr.dtype]

    class Field(Structure):
        _fields_ = [
            ("data", POINTER(baseFieldType)),
            ("dim", c_long),
            ("strides", POINTER(c_long)),
            ("shape", POINTER(c_long)),
        ]

    data = arr.ctypes.data_as(POINTER(baseFieldType))
    dim = c_long(len(arr.shape))
    strides = arr.ctypes.strides_as(c_long)
    for i in range(len(arr.shape)):
        assert strides[i] % sizeof(baseFieldType) == 0
        strides[i] //= sizeof(baseFieldType)
    shape = arr.ctypes.shape_as(c_long)
    return Field(data, dim, strides, shape)


