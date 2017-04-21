import sympy as sp
import numpy as np

from lbmpy.stencils import getStencil
from pystencils import TypedSymbol, Field
from pystencils.backends.cbackend import CustomCppCode
from lbmpy.boundaries.createindexlist import createBoundaryIndexList
from pystencils.slicing import normalizeSlice, makeSlice

INV_DIR_SYMBOL = TypedSymbol("invDir", "int")
WEIGHTS_SYMBOL = TypedSymbol("weights", "double")


class BoundaryHandling(object):
    class BoundaryInfo(object):
        def __init__(self, name, flag, function, kernel, ast):
            self.name = name
            self.flag = flag
            self.function = function
            self.kernel = kernel
            self.ast = ast

    def __init__(self, pdfField, domainShape, lbMethod, ghostLayers=1, target='cpu', openMP=True):
        """
        Class for managing boundary kernels

        :param pdfField: either pdf numpy array (including ghost layers), or pystencils.Field
        :param domainShape: domain size without ghost layers
        :param lbMethod: lattice Boltzmann method
        :param ghostLayers: number of ghost layers
        :param target: either 'cpu' or 'gpu'
        """
        if isinstance(pdfField, np.ndarray):
            symbolicPdfField = Field.createFromNumpyArray('pdfs', pdfField, indexDimensions=1)
            assert pdfField.shape[:-1] == tuple(d + 2*ghostLayers for d in domainShape)
        elif isinstance(pdfField, Field):
            symbolicPdfField = pdfField
        else:
            raise ValueError("pdfField has to be either a numpy array or a pystencils.Field")

        self._symbolicPdfField = symbolicPdfField
        self._shapeWithGhostLayers = [d + 2 * ghostLayers for d in domainShape]
        self._fluidFlag = 2 ** 0
        self.flagField = np.full(self._shapeWithGhostLayers, self._fluidFlag, dtype=np.int32)
        self._ghostLayers = ghostLayers
        self._lbMethod = lbMethod

        self._boundaryInfos = []
        self._nameToBoundary = {}
        self._periodicityKernels = []

        self._dirty = False
        self._periodicity = [False, False, False]
        self._target = target
        self.openMP = openMP
        if target not in ('cpu', 'gpu'):
            raise ValueError("Invalid target '%s' . Allowed values: 'cpu' or 'gpu'" % (target,))

    @property
    def periodicity(self):
        """List that indicates for x,y (z) coordinate if domain is periodic in that direction"""
        return self._periodicity

    @property
    def fluidFlag(self):
        """Flag that is set where the lattice Boltzmann update should happen"""
        return self._fluidFlag

    def getFlag(self, name):
        """Flag that represents the boundary with given name. Raises KeyError if no such boundary exists."""
        return self._nameToBoundary[name].flag

    def getBoundaryNames(self):
        """List of names of all registered boundary conditions"""
        return [b.name for b in self._boundaryInfos]

    def setPeriodicity(self, x=False, y=False, z=False):
        """Enable periodic boundary conditions at the border of the domain"""
        for d in (x, y, z):
            assert isinstance(d, bool)

        self._periodicity = [x, y, z]
        self._compilePeriodicityKernels()

    def hasBoundary(self, name):
        """Returns boolean indicating if a boundary with that name exists"""
        return name in self._nameToBoundary

    def setBoundary(self, function, indexExpr=None, maskArr=None, name=None):
        """
        Sets boundary in a rectangular region (slice)

        :param function: boundary function or the string 'fluid' to remove boundary conditions
        :param indexExpr: slice expression, where boundary should be set, see :mod:`pystencils.slicing`
        :param maskArr: optional boolean (masked) array specifying where the boundary should be set
        :param name: name of the boundary
        """
        if indexExpr is None:
            indexExpr = [slice(None, None, None)] * len(self.flagField.shape)
        if function == 'fluid':
            flag = self._fluidFlag
        else:
            if name is None:
                if hasattr(function, '__name__'):
                    name = function.__name__
                elif hasattr(function, 'name'):
                    name = function.name
                else:
                    raise ValueError("Boundary function has to have a '__name__' or 'name' attribute "
                                     "if name is not specified")

            if not self.hasBoundary(name):
                self.addBoundary(function, name)

            flag = self.getFlag(name)
            assert flag != self._fluidFlag

        indexExpr = normalizeSlice(indexExpr, self._shapeWithGhostLayers)
        if maskArr is None:
            self.flagField[indexExpr] = flag
        else:
            flagFieldView = self.flagField[indexExpr]
            flagFieldView[maskArr] = flag
        self._dirty = True

    def addBoundary(self, boundaryFunction, name=None):
        """
        Adds a boundary condition, i.e. reserves a flog in the flag field and returns that flag
        If a boundary with that name already exists, the existing flag is returned.
        This flag can be logicalled or'ed to the boundaryHandling.flagField

        :param boundaryFunction: boundary condition function, see :mod:`lbmpy.boundaries.boundaryconditions`
        :param name: boundaries with different name are considered different. If not given
                     ```boundaryFunction.__name__`` is used
        """
        if name is None:
            name = boundaryFunction.__name__

        if self.hasBoundary(name):
            return self._boundaryInfos[name].flag

        newIdx = len(self._boundaryInfos) + 1  # +1 because 2**0 is reserved for fluid flag
        boundaryInfo = self.BoundaryInfo(name, 2 ** newIdx, boundaryFunction, None, None)
        self._boundaryInfos.append(boundaryInfo)
        self._nameToBoundary[name] = boundaryInfo
        self._dirty = True
        return boundaryInfo.flag

    def indices(self, dx=1.0, includeGhostLayers=False):
        if not includeGhostLayers:
            params = [np.arange(start=-1, stop=s-1) * dx for s in self.flagField.shape]
        else:
            params = [np.arange(s) * dx for s in self.flagField.shape]
        return np.meshgrid(*params, indexing='ij')

    def __call__(self, **kwargs):
        """Run the boundary handling, all keyword args are passed through to the boundary sweeps"""
        if self._dirty:
            self.prepare()
        for boundary in self._boundaryInfos:
            boundary.kernel(**kwargs)
        for k in self._periodicityKernels:
            k(**kwargs)

    def clear(self):
        """Removes all boundaries and fills the domain with fluid"""
        self.flagField.fill(self._fluidFlag)
        self._dirty = False
        self._boundaryInfos = []
        self._nameToBoundary = {}

    def prepare(self):
        """Fills data structures to speed up the boundary handling and compiles all boundary kernels.
        This is automatically done when first called. With this function this can be triggered before running."""
        for boundary in self._boundaryInfos:
            assert boundary.flag != self._fluidFlag
            idxField = createBoundaryIndexList(self.flagField, self._lbMethod.stencil,
                                               boundary.flag, self._fluidFlag, self._ghostLayers)
            ast = generateBoundaryHandling(self._symbolicPdfField, idxField, self._lbMethod, boundary.function,
                                           target=self._target)
            boundary.ast = ast
            if self._target == 'cpu':
                from pystencils.cpu import makePythonFunction as makePythonCpuFunction, addOpenMP
                addOpenMP(ast, numThreads=self.openMP)
                boundary.kernel = makePythonCpuFunction(ast, {'indexField': idxField})
            elif self._target == 'gpu':
                from pystencils.gpucuda import makePythonFunction as makePythonGpuFunction
                import pycuda.gpuarray as gpuarray
                idxGpuField = gpuarray.to_gpu(idxField)
                boundary.kernel = makePythonGpuFunction(ast, {'indexField': idxGpuField})
            else:
                assert False
        self._dirty = False

    def invalidateIndexCache(self):
        """Invalidates the cache for optimization data structures. When setting boundaries the cache is automatically
        invalidated, so there is no need to call this function manually, as long as the flag field is not manually
        modified."""
        self._dirty = True

    def _compilePeriodicityKernels(self):
        self._periodicityKernels = []
        dim = len(self.flagField.shape)
        if dim == 2:
            stencil = getStencil("D2Q9")
        elif dim == 3:
            stencil = getStencil("D3Q27")
        else:
            assert False

        filteredStencil = []
        for direction in stencil:
            useDirection = True
            if direction == (0, 0) or direction == (0, 0, 0):
                useDirection = False
            for component, periodicity in zip(direction, self._periodicity):
                if not periodicity and component != 0:
                    useDirection = False
            if useDirection:
                filteredStencil.append(direction)

        if len(filteredStencil) > 0:
            if self._target == 'cpu':
                from pystencils.slicing import getPeriodicBoundaryFunctor
                self._periodicityKernels.append(getPeriodicBoundaryFunctor(filteredStencil, ghostLayers=1))
            elif self._target == 'gpu':
                from pystencils.gpucuda.periodicity import getPeriodicBoundaryFunctor
                self._periodicityKernels.append(getPeriodicBoundaryFunctor(filteredStencil, self.flagField.shape,
                                                                           indexDimensions=1,
                                                                           indexDimShape=len(self._lbMethod.stencil),
                                                                           ghostLayers=1))
            else:
                assert False


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

