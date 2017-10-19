import numpy as np

from pystencils.field import Field
from pystencils.slicing import normalizeSlice
from lbmpy.boundaries.boundary_kernel import generateIndexBoundaryKernel
from lbmpy.boundaries.createindexlist import createBoundaryIndexList


class FlagFieldInterface(object):

    def getFlag(self, boundaryObject):
        raise NotImplementedError()

    def getName(self, boundaryObject):
        raise NotImplementedError()

    @property
    def array(self):
        raise NotImplementedError()

    @property
    def boundaryObjects(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    @staticmethod
    def _makeBoundaryName(boundaryObject, existingNames):
        baseName = boundaryObject.name
        name = baseName
        counter = 1
        while name in existingNames:
            name = "%s_%d" % (baseName, counter)
            counter += 1

        return name


class GenericBoundaryHandling(object):

    class BoundaryInfo(object):
        def __init__(self, kernel, ast, indexArray, idxArrayForExecution):
            self.kernel = kernel
            self.ast = ast
            self.indexArray = indexArray
            self.idxArrayForExecution = idxArrayForExecution  # is different for GPU kernels

    def __init__(self, flagFieldInterface, pdfField, lbMethod, ghostLayers=1, target='cpu', openMP=True):
        """
        :param flagFieldInterface: implementation of FlagFieldInterface
        :param pdfField: numpy array
        :param lbMethod:
        :param target: 'cpu' or 'gpu'
        :param openMP:
        """
        self._flagFieldInterface = flagFieldInterface
        self._pdfField = pdfField
        self._lbMethod = lbMethod
        self._target = target
        self._openMP = openMP
        self.ghostLayers = ghostLayers
        self._dirty = False

        self._boundaryInfos = {}  # mapping of boundary object to boundary info
        self._symbolicPdfField = Field.createFromNumpyArray('pdfs', pdfField, indexDimensions=1)

        if target not in ('cpu', 'gpu'):
            raise ValueError("Invalid target '%s' . Allowed values: 'cpu' or 'gpu'" % (target,))

    def getBoundaryNameToFlagDict(self):
        result = {self._flagFieldInterface.getName(o): self._flagFieldInterface.getFlag(o) for o in self._boundaryInfos}
        result['fluid'] = self._flagFieldInterface.getFlag('fluid')
        return result

    def hasBoundary(self, boundaryObject):
        """Returns boolean indicating if this boundary exists in that handling"""
        return boundaryObject in self._boundaryInfos

    def __call__(self, **kwargs):
        """
        Runs boundary handling
        :param kwargs: keyword arguments passed to boundary kernel
        :return:
        """
        if self._dirty:
            self.prepare()
        for boundary in self._boundaryInfos.values():
            if boundary.kernel:
                boundary.kernel(indexField=boundary.idxArrayForExecution, **kwargs)

    def getBoundaryIndexArray(self, boundaryObject):
        return self._boundaryInfos[boundaryObject].indexArray

    def clear(self):
        """Removes all boundaries and fills the domain with fluid"""
        self._flagFieldInterface.clear()
        self._dirty = False
        self._boundaryInfos = {}

    def _invalidateCache(self):
        self._dirty = True

    def triggerReinitializationOfBoundaryData(self, **kwargs):
        if self._dirty:
            self.prepare()
            return
        else:
            for boundaryObject, boundaryInfo in self._boundaryInfos.items():
                self.__boundaryDataInitialization(boundaryInfo.indexArray, boundaryObject, **kwargs)
                if self._target == 'gpu':
                    import pycuda.gpuarray as gpuarray
                    boundaryInfo.idxArrayForExecution = gpuarray.to_gpu(boundaryInfo.indexArray)

    def __boundaryDataInitialization(self, idxArray, boundaryObject, **kwargs):
        if boundaryObject.additionalDataInitCallback:
            #TODO x,y,z coordinates should be transformed here
            boundaryObject.additionalDataInitCallback(idxArray, **kwargs)

        if boundaryObject.additionalDataInitKernelEquations:
            initKernelAst = generateIndexBoundaryKernel(self._symbolicPdfField, idxArray, self._lbMethod,
                                                        boundaryObject, target='cpu',
                                                        createInitializationKernel=True)
            from pystencils.cpu import makePythonFunction as makePythonCpuFunction
            initKernel = makePythonCpuFunction(initKernelAst, {'indexField': idxArray, 'pdfs': self._pdfField})
            initKernel()

    def prepare(self):
        """Compiles boundary kernels according to flag field. When setting boundaries the cache is automatically
        invalidated, so there is no need to call this function manually, as long as the flag field is not manually
        modified."""
        if not self._dirty:
            return

        dim = self._lbMethod.dim
        fluidFlag = self._flagFieldInterface.getFlag('fluid')

        for boundaryObject in self._flagFieldInterface.boundaryObjects:
            boundaryFlag = self._flagFieldInterface.getFlag(boundaryObject)
            idxArray = createBoundaryIndexList(self.flagField, self._lbMethod.stencil,
                                               boundaryFlag, fluidFlag, self.ghostLayers)
            if len(idxArray) == 0:
                continue

            if boundaryObject.additionalData:
                coordinateNames = ["x", "y", "z"][:dim]
                indexArrDtype = np.dtype([(name, np.int32) for name in coordinateNames] +
                                         [('dir', np.int32)] +
                                         [(i[0], i[1].numpyDtype) for i in boundaryObject.additionalData])
                extendedIdxField = np.empty(len(idxArray), dtype=indexArrDtype)
                for prop in coordinateNames + ['dir']:
                    extendedIdxField[prop] = idxArray[prop]

                idxArray = extendedIdxField
                self.__boundaryDataInitialization(idxArray, boundaryObject)

            ast = generateIndexBoundaryKernel(self._symbolicPdfField, idxArray, self._lbMethod, boundaryObject,
                                              target=self._target)

            if self._target == 'cpu':
                from pystencils.cpu import makePythonFunction as makePythonCpuFunction, addOpenMP
                addOpenMP(ast, numThreads=self._openMP)
                idxArrayForExecution = idxArray
                kernel = makePythonCpuFunction(ast)
            elif self._target == 'gpu':
                from pystencils.gpucuda import makePythonFunction as makePythonGpuFunction
                import pycuda.gpuarray as gpuarray
                idxArrayForExecution = gpuarray.to_gpu(idxArray)
                kernel = makePythonGpuFunction(ast)
            else:
                assert False

            boundaryInfo = GenericBoundaryHandling.BoundaryInfo(kernel, ast, idxArray, idxArrayForExecution)
            self._boundaryInfos[boundaryObject] = boundaryInfo

        self._dirty = False

    def reserveFlag(self, boundaryObject):
        self._flagFieldInterface.getFlag(boundaryObject)

    def setBoundary(self, boundaryObject, indexExpr=None, maskArr=None):
        """
        Sets boundary in a rectangular region (slice)

        :param boundaryObject: boundary condition object or the string 'fluid' to remove boundary conditions
        :param indexExpr: slice expression, where boundary should be set, see :mod:`pystencils.slicing`
        :param maskArr: optional boolean (masked) array specifying where the boundary should be set
        """
        if indexExpr is None:
            indexExpr = [slice(None, None, None)] * len(self.flagField.shape)

        flag = self._flagFieldInterface.getFlag(boundaryObject)
        flagField = self._flagFieldInterface.array
        indexExpr = normalizeSlice(indexExpr, flagField.shape)

        if maskArr is None:
            flagField[indexExpr] = flag
        else:
            flagFieldView = flagField[indexExpr].squeeze()
            maskArr = maskArr.squeeze()
            flagFieldView[maskArr] = flag
        self._dirty = True

    def getMask(self, boundaryObject):
        return np.bitwise_and(self._flagFieldInterface.array, self._flagFieldInterface.getFlag(boundaryObject))

    @property
    def flagField(self):
        return self._flagFieldInterface.array
