import numpy as np

from lbmpy.stencils import inverseDirection
from pystencils.cache import memorycache
from pystencils import Field
from lbmpy.boundaries import NoSlip
from lbmpy.boundaries.boundary_kernel import generateIndexBoundaryKernelGeneric
from lbmpy.boundaries.createindexlist import createBoundaryIndexArray, numpyDataTypeForBoundaryObject


class BoundaryHandling:

    def __init__(self, lbMethod, dataHandling, pdfFieldName, name="boundaryHandling", target='cpu', openMP=True):
        assert dataHandling.hasData(pdfFieldName)

        self._lbMethod = lbMethod
        self._dataHandling = dataHandling
        self._pdfArrayName = pdfFieldName
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
        dataHandling.addArray(self._flagFieldName, dtype=np.uint32, cpu=True, gpu=False)
        dataHandling.addCustomClass(self._indexArrayName, self.IndexFieldBlockData, cpu=True, gpu=gpu)

        ffGhostLayers = self._dataHandling.ghostLayersOfField(self._flagFieldName)
        for b in self._dataHandling.iterate(ghostLayers=ffGhostLayers):
            b[self._flagFieldName].fill(self._fluidFlag)

    @property
    def shape(self):
        return self._dataHandling.shape

    @property
    def dim(self):
        return self._lbMethod.dim

    @property
    def lbMethod(self):
        return self._lbMethod

    @property
    def dataHandling(self):
        return self._dataHandling

    @property
    def boundaryObjects(self):
        return tuple(self._boundaryObjectToName.keys())

    @property
    def flagArrayName(self):
        return self._flagFieldName

    def getBoundaryNameToFlagDict(self):
        result = {bObj.name: bInfo.flag for bObj, bInfo in self._boundaryObjectToBoundaryInfo.items()}
        result['fluid'] = self._fluidFlag
        return result

    def getMask(self, sliceObj, boundaryObj, inverse=False):
        if isinstance(boundaryObj, str) and boundaryObj.lower() == 'fluid':
            flag = self._fluidFlag
        else:
            flag = self._boundaryObjectToBoundaryInfo[boundaryObj].flag

        arr = self.dataHandling.gatherArray(self.flagArrayName, sliceObj)
        if arr is None:
            return None
        else:
            result = np.bitwise_and(arr, flag)
            if inverse:
                result = np.logical_not(result)
            return result

    def setBoundary(self, boundaryObject, sliceObj=None, maskCallback=None, ghostLayers=True, innerGhostLayers=True):
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
        if isinstance(boundaryObject, str) and boundaryObject.lower() == 'fluid':
            flag = self._fluidFlag
        else:
            flag = self._getFlagForBoundary(boundaryObject)

        for b in self._dataHandling.iterate(sliceObj, ghostLayers=ghostLayers, innerGhostLayers=innerGhostLayers):
            flagArr = b[self._flagFieldName]
            if maskCallback is not None:
                mask = maskCallback(*b.midpointArrays)
                flagArr[mask] = flag
            else:
                flagArr.fill(flag)

        self._dirty = True

    def forceOnBoundary(self, boundaryObj):
        if isinstance(boundaryObj, NoSlip):
            return self._forceOnNoSlip(boundaryObj)
        else:
            self.__call__()
            return self._forceOnBoundary(boundaryObj)

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

        for b in self._dataHandling.iterate(gpu=self._target == 'gpu'):
            for bObj, idxArr in b[self._indexArrayName].boundaryObjectToIndexList.items():
                kwargs[self._pdfArrayName] = b[self._pdfArrayName]
                self._boundaryObjectToBoundaryInfo[bObj].kernel(indexField=idxArr, **kwargs)

    def geometryToVTK(self, fileName='geometry', boundaries='all', ghostLayers=False):
        """
        Writes a VTK field where each cell with the given boundary is marked with 1, other cells are 0
        This can be used to display the simulation geometry in Paraview
        :param fileName: vtk filename
        :param boundaries: boundary object, or special string 'fluid' for fluid cells or special string 'all' for all
                         boundary conditions.
                         can also  be a sequence, to write multiple boundaries to VTK file
        :param ghostLayers: number of ghost layers to write, or True for all, False for none
        """
        if boundaries == 'all':
            boundaries = list(self._boundaryObjectToBoundaryInfo.keys()) + ['fluid']
        elif not hasattr(boundaries, "__len__"):
            boundaries = [boundaries]

        masksToName = {}
        for b in boundaries:
            if b == 'fluid':
                masksToName[self._fluidFlag] = 'fluid'
            else:
                masksToName[self._boundaryObjectToBoundaryInfo[b].flag] = b.name

        writer = self.dataHandling.vtkWriterFlags(fileName, self._flagFieldName, masksToName, ghostLayers=ghostLayers)
        writer(1)

    # ------------------------------ Implementation Details ------------------------------------------------------------

    def _forceOnNoSlip(self, boundaryObj):
        dh = self._dataHandling
        ffGhostLayers = dh.ghostLayersOfField(self._flagFieldName)
        method = self._lbMethod
        stencil = np.array(method.stencil)

        result = np.zeros(self.dim)

        for b in dh.iterate(ghostLayers=ffGhostLayers):
            objToIndList = b[self._indexArrayName].boundaryObjectToIndexList
            pdfArray = b[self._pdfArrayName]
            if boundaryObj in objToIndList:
                indArr = objToIndList[boundaryObj]
                indices = [indArr[name] for name in ('x', 'y', 'z')[:method.dim]]
                indices.append(indArr['dir'])
                values = 2 * pdfArray[tuple(indices)]
                forces = stencil[indArr['dir']] * values[:, np.newaxis]
                result += forces.sum(axis=0)
        return dh.reduceFloatSequence(list(result), 'sum')

    def _forceOnBoundary(self, boundaryObj):
        dh = self._dataHandling
        ffGhostLayers = dh.ghostLayersOfField(self._flagFieldName)
        method = self._lbMethod
        stencil = np.array(method.stencil)
        invDirection = np.array([method.stencil.index(inverseDirection(d))
                                 for d in method.stencil])

        result = np.zeros(self.dim)

        for b in dh.iterate(ghostLayers=ffGhostLayers):
            objToIndList = b[self._indexArrayName].boundaryObjectToIndexList
            pdfArray = b[self._pdfArrayName]
            if boundaryObj in objToIndList:
                indArr = objToIndList[boundaryObj]
                indices = [indArr[name] for name in ('x', 'y', 'z')[:method.dim]]
                offsets = stencil[indArr['dir']]
                invDir = invDirection[indArr['dir']]
                fluidValues = pdfArray[tuple(indices) + (indArr['dir'],)]
                boundaryValues = pdfArray[tuple(ind + offsets[:, i] for i, ind in enumerate(indices)) + (invDir,)]
                values = fluidValues + boundaryValues
                forces = stencil[indArr['dir']] * values[:, np.newaxis]
                result += forces.sum(axis=0)

        return dh.reduceFloatSequence(list(result), 'sum')

    def _getFlagForBoundary(self, boundaryObject):
        if boundaryObject not in self._boundaryObjectToBoundaryInfo:
            symbolicIndexField = Field.createGeneric('indexField', spatialDimensions=1,
                                                     dtype=numpyDataTypeForBoundaryObject(boundaryObject, self.dim))

            spPdfField = self._dataHandling.fields[self._pdfArrayName]
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
            pdfArr = b[self._pdfArrayName]
            indexArrayBD = b[self._indexArrayName]
            indexArrayBD.clear()
            for bInfo in self._boundaryObjectToBoundaryInfo.values():
                idxArr = createBoundaryIndexArray(flagArr, self.lbMethod.stencil, bInfo.flag, self._fluidFlag,
                                                  bInfo.boundaryObject, dh.ghostLayersOfField(self._flagFieldName))
                if idxArr.size == 0:
                    continue

                # TODO test boundary data setter (especially offset)
                boundaryDataSetter = BoundaryDataSetter(idxArr, b.offset, self.lbMethod.stencil, ffGhostLayers, pdfArr)
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
        def __init__(self, *args, **kwargs):
            self.boundaryObjectToIndexList = {}
            self.boundaryObjectToDataSetter = {}

        def clear(self):
            self.boundaryObjectToIndexList.clear()
            self.boundaryObjectToDataSetter.clear()

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
            for obj, cpuArr in cpuVersion.items():
                if obj not in gpuVersion:
                    gpuVersion[obj] = gpuarray.to_gpu(cpuArr)
                else:
                    gpuVersion[obj].set(cpuArr)


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