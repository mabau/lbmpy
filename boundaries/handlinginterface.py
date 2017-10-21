import numpy as np

from pystencils.field import Field
from pystencils.slicing import normalizeSlice, shiftSlice
from lbmpy.boundaries.boundary_kernel import generateIndexBoundaryKernel
from lbmpy.boundaries.createindexlist import createBoundaryIndexList
from pystencils.cache import memorycache


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


class BoundaryDataSetter:

    def __init__(self, indexArray, offset, stencil, ghostLayers):
        self.indexArray = indexArray
        self.offset = offset
        self.stencil = np.array(stencil)
        arrFieldNames = indexArray.dtype.names
        self.dim = 3 if 'z' in arrFieldNames else 2
        assert 'x' in arrFieldNames and 'y' in arrFieldNames and 'dir' in arrFieldNames, str(arrFieldNames)
        self.boundaryDataNames = set(self.indexArray.dtype.names) - set(['x', 'y', 'z', 'dir'])
        self.coordMap = {0: 'x', 1: 'y', 2: 'z'}
        self.ghostLayers = ghostLayers

    def fluidCellPositions(self, coord):
        assert coord < self.dim
        return self.indexArray[self.coordMap[coord]] + self.offset[coord] - self.ghostLayers

    @memorycache()
    def linkOffsets(self):
        return self.stencil[self.indexArray['dir']]

    @memorycache()
    def linkPositions(self, coord):
        return self.fluidCellPositions(coord) + 0.5 * self.linkOffsets()[:, coord]

    @memorycache()
    def boundaryCellPositions(self, coord):
        return self.fluidCellPositions(coord) + self.linkOffsets()[:, coord]

    def __setitem__(self, key, value):
        if key not in self.boundaryDataNames:
            raise KeyError("Invalid boundary data name %s. Allowed are %s" % (key, self.boundaryDataNames))
        self.indexArray[key] = value

    def __getitem__(self, item):
        if item not in self.boundaryDataNames:
            raise KeyError("Invalid boundary data name %s. Allowed are %s" % (item, self.boundaryDataNames))
        return self.indexArray[item]


class GenericBoundaryHandling(object):

    class BoundaryInfo(object):
        def __init__(self, kernel, ast, indexArray, idxArrayForExecution, boundaryDataSetter):
            self.kernel = kernel
            self.ast = ast
            self.indexArray = indexArray
            self.idxArrayForExecution = idxArrayForExecution  # is different for GPU kernels
            self.boundaryDataSetter = boundaryDataSetter

    def __init__(self, flagFieldInterface, pdfField, lbMethod, offset=None, ghostLayers=1, target='cpu', openMP=True):
        """
        :param flagFieldInterface: implementation of FlagFieldInterface
        :param pdfField: numpy array
        :param lbMethod:
        :param offset: offset that is added to all coordinates when calling callback functions to set up geometry
                       or boundary data. This is used for waLBerla simulations to pass the block offset  
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
        self.offset = offset if offset else (0,) * lbMethod.dim
        self._boundaryInfos = {}  # mapping of boundary object to boundary info
        self._symbolicPdfField = Field.createFromNumpyArray('pdfs', pdfField, indexDimensions=1)
        self.dim = lbMethod.dim

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

    def triggerReinitializationOfBoundaryData(self, **kwargs):
        if self._dirty:
            self.prepare()
            return
        else:
            for boundaryObject, boundaryInfo in self._boundaryInfos.items():
                self.__boundaryDataInitialization(boundaryInfo, boundaryObject, **kwargs)
                if self._target == 'gpu':
                    import pycuda.gpuarray as gpuarray
                    boundaryInfo.idxArrayForExecution = gpuarray.to_gpu(boundaryInfo.indexArray)

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

            boundaryDataSetter = BoundaryDataSetter(idxArray, self.offset, self._lbMethod.stencil, self.ghostLayers)
            boundaryInfo = GenericBoundaryHandling.BoundaryInfo(kernel, ast, idxArray,
                                                                idxArrayForExecution, boundaryDataSetter)
            self._boundaryInfos[boundaryObject] = boundaryInfo

            if boundaryObject.additionalData:
                self.__boundaryDataInitialization(boundaryInfo, boundaryObject)

        self._dirty = False

    def reserveFlag(self, boundaryObject):
        self._flagFieldInterface.getFlag(boundaryObject)

    def setBoundary(self, boundaryObject, indexExpr=None, maskCallback=None, includeGhostLayers=True):
        """
        Sets boundary using either a rectangular slice, a boolean mask or a combination of both
        
        :param boundaryObject: instance of a boundary object that should be set 
        :param indexExpr: a slice object (can be created with makeSlice[]) that selects a part of the domain where
                          the boundary should be set. If none, the complete domain is selected which makes only sense
                          if a maskCallback is passed. The slice can have ':' placeholders, which are interpreted 
                          depending on the 'includeGhostLayers' parameter i.e. if it is True, the slice extends
                          into the ghost layers
        :param maskCallback: callback function getting x,y (z) parameters of the cell midpoints and returning a 
                             boolean mask with True entries where boundary cells should be set. 
                             The x, y, z arrays have 2D/3D shape such that they can be used directly 
                             to create the boolean return array. i.e return x < 10 sets boundaries in cells with
                             midpoint x coordinate smaller than 10.
        :param includeGhostLayers: if this parameter is False, boundaries can not be set in the ghost
                                   layer, because index 0 is the first inner layer and -1 is interpreted in the Python
                                   way as maximum. If this parameter is True, the lower ghost layers have index 0, and
                                   placeholders ':' in index expressions extend into the ghost layers.
        """
        if indexExpr is None:
            indexExpr = [slice(None, None, None)] * len(self.flagField.shape)
        if not includeGhostLayers:
            domainSize = [i - 2 * self.ghostLayers for i in self._flagFieldInterface.array.shape]
            indexExpr = normalizeSlice(indexExpr, domainSize)
            indexExpr = shiftSlice(indexExpr, self.ghostLayers)
        else:
            indexExpr = normalizeSlice(indexExpr, self._flagFieldInterface.array.shape)

        mask = None
        if maskCallback is not None:
            gridParams = []
            for s, offset in zip(indexExpr, self.offset):
                if isinstance(s, slice):
                    gridParams.append(np.arange(s.start, s.stop) + offset + 0.5 - self.ghostLayers)
                else:
                    gridParams.append(s + offset + 0.5 - self.ghostLayers)
            indices = np.meshgrid(*gridParams, indexing='ij')
            mask = maskCallback(*indices)
        return self._setBoundaryWithMaskArray(boundaryObject, indexExpr, mask)

    def _setBoundaryWithMaskArray(self, boundaryObject, indexExpr=None, maskArr=None):
        """
        Sets boundary in a rectangular region (slice)

        :param boundaryObject: boundary condition object or the string 'fluid' to remove boundary conditions
        :param indexExpr: slice expression, where boundary should be set. ghost layers are expected to have coord=0
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

    def _invalidateCache(self):
        self._dirty = True

    def __boundaryDataInitialization(self, boundaryInfo, boundaryObject, **kwargs):
        if boundaryObject.additionalDataInitCallback:
            boundaryObject.additionalDataInitCallback(boundaryInfo.boundaryDataSetter, **kwargs)

        if boundaryObject.additionalDataInitKernelEquations:
            initKernelAst = generateIndexBoundaryKernel(self._symbolicPdfField, boundaryInfo.indexArray, self._lbMethod,
                                                        boundaryObject, target='cpu',
                                                        createInitializationKernel=True)
            from pystencils.cpu import makePythonFunction as makePythonCpuFunction
            initKernel = makePythonCpuFunction(initKernelAst, {'indexField': boundaryInfo.indexArray,
                                                               'pdfs': self._pdfField})
            initKernel()
