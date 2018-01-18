import numpy as np

from lbmpy.boundaries.boundary_kernel import generateIndexBoundaryKernelGeneric
from pystencils import Field
from lbmpy.boundaries.createindexlist import createBoundaryIndexArray, numpyDataTypeForBoundaryObject
from pystencils.cache import memorycache


class BoundaryDataSetter:

    def __init__(self, indexArray, offset, stencil, ghostLayers, pdfArray):
        self.indexArray = indexArray
        self.offset = offset
        self.stencil = np.array(stencil)
        self.pdfArray = pdfArray.view()
        self.pdfArray.flags.writeable = False

        arrFieldNames = indexArray.dtype.names
        self.dim = 3 if 'z' in arrFieldNames else 2
        assert 'x' in arrFieldNames and 'y' in arrFieldNames and 'dir' in arrFieldNames, str(arrFieldNames)
        self.boundaryDataNames = set(self.indexArray.dtype.names) - set(['x', 'y', 'z', 'dir'])
        self.coordMap = {0: 'x', 1: 'y', 2: 'z'}
        self.ghostLayers = ghostLayers

    def fluidCellPositions(self, coord):
        assert coord < self.dim
        return self.indexArray[self.coordMap[coord]] + self.offset[coord] - self.ghostLayers + 0.5

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


class BoundaryHandling:

    def __init__(self, lbMethod, dataHandling, pdfFieldName, name="boundaryHandling", target='cpu', openMP=True):
        assert dataHandling.hasData(pdfFieldName)

        self._lbMethod = lbMethod
        self._dataHandling = dataHandling
        self._pdfFieldName = pdfFieldName
        self._flagFieldName = name + "Flags"
        self._indexArrayName = name + "IndexArrays"
        self._target = target
        self._openMP = openMP
        self._boundaryObjectToBoundaryInfo = {}
        self._fluidFlag = 1 << 0
        self._nextFreeFlag = 1

        self._dirty = True

        # Add flag field to data handling if it does not yet exist
        if dataHandling.hasData(self._flagFieldName) or dataHandling.hasData(self._indexArrayName):
            raise ValueError("There is already a boundary handling registered at the data handling."
                             "If you want to add multiple handlings, choose a different name.")

        gpu = self._target == 'gpu'
        dataHandling.addArray(self._flagFieldName, dtype=np.int32, cpu=True, gpu=gpu)
        dataHandling.addCustomClass(self._indexArrayName, self.IndexFieldBlockData, cpu=True, gpu=gpu)

        ffGhostLayers = self._dataHandling.ghostLayersOfField(self._flagFieldName)
        for b in self._dataHandling.iterate(ghostLayers=ffGhostLayers):
            b[self._flagFieldName].fill(self._fluidFlag)

    @property
    def dim(self):
        return self._lbMethod.dim

    @property
    def lbMethod(self):
        return self._lbMethod

    @property
    def boundaryObjects(self):
        return tuple(self._boundaryObjectToName.keys())

    def setBoundary(self, boundaryObject, sliceObj=None, maskCallback=None, ghostLayers=True):
        """
        Sets boundary using either a rectangular slice, a boolean mask or a combination of both

        :param boundaryObject: instance of a boundary object that should be set
        :param sliceObj: a slice object (can be created with makeSlice[]) that selects a part of the domain where
                          the boundary should be set. If none, the complete domain is selected which makes only sense
                          if a maskCallback is passed. The slice can have ':' placeholders, which are interpreted
                          depending on the 'includeGhostLayers' parameter i.e. if it is True, the slice extends
                          into the ghost layers
        :param maskCallback: callback function getting x,y (z) parameters of the cell midpoints and returning a
                             boolean mask with True entries where boundary cells should be set.
                             The x, y, z arrays have 2D/3D shape such that they can be used directly
                             to create the boolean return array. i.e return x < 10 sets boundaries in cells with
                             midpoint x coordinate smaller than 10.
        :param ghostLayers see DataHandling.iterate()
        """
        flag = self._getFlagForBoundary(boundaryObject)

        for b in self._dataHandling.iterate(sliceObj=sliceObj, ghostLayers=ghostLayers):
            flagArr = b[self._flagFieldName]
            if maskCallback is not None:
                mask = maskCallback(*b.midpointArrays)
                flagArr[mask] = flag
            else:
                flagArr.fill(flag)

        self._dirty = True

    def getBoundaryNameToFlagDict(self):
        result = {bObj.name: bInfo.flag for bObj, bInfo in self._boundaryObjectToBoundaryInfo.values()}
        result['fluid'] = self._fluidFlag
        return result

    def prepare(self):
        if not self._dirty:
            return
        self._createIndexFields()
        self._dirty = False

    def triggerReinitializationOfBoundaryData(self, **kwargs):
        if self._dirty:
            self.prepare()
        else:
            ffGhostLayers = self._dataHandling.ghostLayersOfField(self._flagFieldName)
            for b in self._dataHandling.iterate(ghostLayers=ffGhostLayers):
                for bObj, setter in b[self._indexArrayName].boundaryObjectToDataSetter.items():
                    self._boundaryDataInitialization(bObj, setter, **kwargs)

    def __call__(self, **kwargs):
        if self._dirty:
            self.prepare()

        for b in self._dataHandling.iterate():
            for bInfo in self._boundaryObjectToBoundaryInfo.values():
                idxArr = b[self._indexArrayName].boundaryObjectToIndexList[bInfo.boundaryObject]
                kwargs[self._pdfFieldName] = b[self._pdfFieldName]
                bInfo.kernel(indexField=idxArr, **kwargs)

    # ------------------------------ Implementation Details ------------------------------------------------------------

    def _getFlagForBoundary(self, boundaryObject):
        if boundaryObject not in self._boundaryObjectToBoundaryInfo:
            symbolicIndexField = Field.createGeneric('indexField', spatialDimensions=1,
                                                     dtype=numpyDataTypeForBoundaryObject(boundaryObject, self.dim))

            spPdfField = self._dataHandling.fields[self._pdfFieldName]
            ast = generateIndexBoundaryKernelGeneric(spPdfField, symbolicIndexField, self._lbMethod,
                                                     boundaryObject, target=self._target, openMP=self._openMP)

            boundaryInfo = self.BoundaryInfo(boundaryObject, flag=1 << self._nextFreeFlag, kernel=ast.compile())

            self._nextFreeFlag += 1
            self._boundaryObjectToBoundaryInfo[boundaryObject] = boundaryInfo
        return self._boundaryObjectToBoundaryInfo[boundaryObject].flag

    def _createIndexFields(self):
        dh = self._dataHandling
        ffGhostLayers = dh.ghostLayersOfField(self._flagFieldName)
        for b in dh.iterate(ghostLayers=ffGhostLayers):
            flagArr = b[self._flagFieldName]
            for bInfo in self._boundaryObjectToBoundaryInfo.values():
                idxArr = createBoundaryIndexArray(flagArr, self.lbMethod.stencil, bInfo.flag, self._fluidFlag,
                                                  bInfo.boundaryObject, dh.ghostLayersOfField(self._flagFieldName))

                pdfArr = b[self._pdfFieldName]
                # TODO test that offset is used correctly here
                boundaryDataSetter = BoundaryDataSetter(idxArr, b.offset, self.lbMethod.stencil, ffGhostLayers, pdfArr)
                indexArrayBD = b[self._indexArrayName]
                indexArrayBD.boundaryObjectToIndexList[bInfo.boundaryObject] = idxArr
                indexArrayBD.boundaryObjectToDataSetter[bInfo.boundaryObject] = boundaryDataSetter
                self._boundaryDataInitialization(bInfo.boundaryObject, boundaryDataSetter)

    def _boundaryDataInitialization(self, boundaryObject, boundaryDataSetter, **kwargs):
        if boundaryObject.additionalDataInitCallback:
            boundaryObject.additionalDataInitCallback(boundaryDataSetter, **kwargs)
        if self._target == 'gpu':
            self._dataHandling.toGpu(self._indexArrayName)

    class BoundaryInfo(object):
        def __init__(self, boundaryObject, flag, kernel):
            self.boundaryObject = boundaryObject
            self.flag = flag
            self.kernel = kernel

    class IndexFieldBlockData:
        def __init__(self):
            self.boundaryObjectToIndexList = {}
            self.boundaryObjectToDataSetter = {}

        @staticmethod
        def toCpu(gpuVersion, cpuVersion):
            gpuVersion = gpuVersion.boundaryObjectToIndexList
            cpuVersion = cpuVersion.boundaryObjectToIndexList
            for obj, cpuArr in cpuVersion.values():
                gpuVersion[obj].get(cpuArr)

        @staticmethod
        def toGpu(gpuVersion, cpuVersion):
            from pycuda import gpuarray
            gpuVersion = gpuVersion.boundaryObjectToIndexList
            cpuVersion = cpuVersion.boundaryObjectToIndexList
            for obj, cpuArr in cpuVersion.values():
                if obj not in gpuVersion:
                    gpuVersion[obj] = gpuarray.to_gpu(cpuArr)
                else:
                    gpuVersion[obj].set(cpuArr)