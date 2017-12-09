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
        def __init__(self, flag, object, kernel, ast):
            self.flag = flag
            self.object = object
            self.kernel = kernel
            self.ast = ast

    def __init__(self, pdfField, domainShape, lbMethod, ghostLayers=1, target='cpu', openMP=True):
        """
        Class for managing boundary kernels

        :param pdfField: pdf numpy array including ghost layers
        :param domainShape: domain size without ghost layers
        :param lbMethod: lattice Boltzmann method
        :param ghostLayers: number of ghost layers
        :param target: either 'cpu' or 'gpu'
        """
        symbolicPdfField = Field.createFromNumpyArray('pdfs', pdfField, indexDimensions=1)
        assert pdfField.shape[:-1] == tuple(d + 2*ghostLayers for d in domainShape)

        self._symbolicPdfField = symbolicPdfField
        self._pdfField = pdfField
        self._shapeWithGhostLayers = [d + 2 * ghostLayers for d in domainShape]
        self._fluidFlag = 2 ** 0
        self.flagField = np.full(self._shapeWithGhostLayers, self._fluidFlag, dtype=np.int32)
        self._ghostLayers = ghostLayers
        self._lbMethod = lbMethod

        self._boundaryInfos = {}
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

    def getFlag(self, boundaryObject):
        """Flag that represents the given boundary."""
        return self._boundaryInfos[boundaryObject].flag

    def getBoundaries(self):
        return [b.object for b in self._boundaryInfos.values()]

    def setPeriodicity(self, x=False, y=False, z=False):
        """Enable periodic boundary conditions at the border of the domain"""
        for d in (x, y, z):
            assert isinstance(d, bool)

        self._periodicity = [x, y, z]
        self._compilePeriodicityKernels()

    def hasBoundary(self, boundaryObject):
        """Returns boolean indicating if a boundary with that name exists"""
        return boundaryObject in self._boundaryInfos

    def setBoundary(self, boundaryObject, indexExpr=None, maskArr=None):
        """
        Sets boundary in a rectangular region (slice)

        :param boundaryObject: boundary condition object or the string 'fluid' to remove boundary conditions
        :param indexExpr: slice expression, where boundary should be set, see :mod:`pystencils.slicing`
        :param maskArr: optional boolean (masked) array specifying where the boundary should be set
        """
        if indexExpr is None:
            indexExpr = [slice(None, None, None)] * len(self.flagField.shape)
        if boundaryObject == 'fluid':
            flag = self._fluidFlag
        else:
            flag = self.addBoundary(boundaryObject)
            assert flag != self._fluidFlag

        indexExpr = normalizeSlice(indexExpr, self._shapeWithGhostLayers)
        if maskArr is None:
            self.flagField[indexExpr] = flag
        else:
            flagFieldView = self.flagField[indexExpr]
            flagFieldView[maskArr] = flag
        self._dirty = True

    def addBoundary(self, boundaryObject):
        """
        Adds a boundary condition, i.e. reserves a flog in the flag field and returns that flag
        If the boundary already exists, the existing flag is returned.
        This flag can be logicalled or'ed to the boundaryHandling.flagField

        :param boundaryObject: boundary condition object, see :mod:`lbmpy.boundaries.boundaryconditions`
        """
        if boundaryObject in self._boundaryInfos:
            return self._boundaryInfos[boundaryObject].flag

        newIdx = len(self._boundaryInfos) + 1  # +1 because 2**0 is reserved for fluid flag
        boundaryInfo = self.BoundaryInfo(2 ** newIdx, boundaryObject, None, None)
        self._boundaryInfos[boundaryObject] = boundaryInfo
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
        for boundary in self._boundaryInfos.values():
            boundary.kernel(**kwargs)
        for k in self._periodicityKernels:
            k(**kwargs)

    def clear(self):
        """Removes all boundaries and fills the domain with fluid"""
        self.flagField.fill(self._fluidFlag)
        self._dirty = False
        self._boundaryInfos = {}

    def prepare(self):
        """Fills data structures to speed up the boundary handling and compiles all boundary kernels.
        This is automatically done when first called. With this function this can be triggered before running."""
        for boundary in self._boundaryInfos.values():
            assert boundary.flag != self._fluidFlag
            idxArray = createBoundaryIndexList(self.flagField, self._lbMethod.stencil,
                                               boundary.flag, self._fluidFlag, self._ghostLayers)

            dim = self._lbMethod.dim

            if boundary.object.additionalData:
                coordinateNames = ["x", "y", "z"][:dim]
                indexArrDtype = np.dtype([(name, np.int32) for name in coordinateNames] +
                                         [('dir', np.int32)] +
                                         [(i[0], i[1].numpyDtype) for i in boundary.object.additionalData])
                extendedIdxField = np.empty(len(idxArray), dtype=indexArrDtype)
                for prop in coordinateNames + ['dir']:
                    extendedIdxField[prop] = idxArray[prop]

                idxArray = extendedIdxField
                if boundary.object.additionalDataInitKernelEquations:
                    initKernelAst = generateIndexBoundaryKernel(self._symbolicPdfField, idxArray, self._lbMethod,
                                                                boundary.object, target='cpu',
                                                                createInitializationKernel=True)
                    from pystencils.cpu import makePythonFunction as makePythonCpuFunction
                    initKernel = makePythonCpuFunction(initKernelAst, {'indexField': idxArray, 'pdfs': self._pdfField})
                    initKernel()

            ast = generateIndexBoundaryKernel(self._symbolicPdfField, idxArray, self._lbMethod, boundary.object,
                                              target=self._target)
            boundary.ast = ast
            if self._target == 'cpu':
                from pystencils.cpu import makePythonFunction as makePythonCpuFunction, addOpenMP
                addOpenMP(ast, numThreads=self.openMP)
                boundary.kernel = makePythonCpuFunction(ast, {'indexField': idxArray})
            elif self._target == 'gpu':
                from pystencils.gpucuda import makePythonFunction as makePythonGpuFunction
                import pycuda.gpuarray as gpuarray
                idxGpuField = gpuarray.to_gpu(idxArray)
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


def generateIndexBoundaryKernel(pdfField, indexArr, lbMethod, boundaryFunctor, target='cpu',
                                createInitializationKernel=False):
    indexField = Field.createFromNumpyArray("indexField", indexArr)

    elements = [LbmMethodInfo(lbMethod)]
    dirSymbol = TypedSymbol("dir", indexArr.dtype.fields['dir'][0])
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

