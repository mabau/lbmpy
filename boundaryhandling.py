import numpy as np
from lbmpy.boundaries import createBoundaryIndexList, generateBoundaryHandling
from pystencils.cpu import makePythonFunction


class BoundaryHandling:
    def __init__(self, symbolicPdfField, domainShape, latticeModel, ghostLayers=1):
        self._symbolicPdfField = symbolicPdfField
        self._shapeWithGhostLayers = [d + 2 * ghostLayers for d in domainShape]
        self._fluidFlag = 2 ** 31
        self.flagField = np.full(self._shapeWithGhostLayers, self._fluidFlag, dtype=np.int32)
        self._ghostLayers = ghostLayers
        self._latticeModel = latticeModel
        self._boundaryFunctions = []
        self._nameToIndex = {}
        self._boundarySweeps = []

    def addBoundary(self, boundaryFunction, name=None):
        if name is None:
            name = boundaryFunction.__name__

        self._nameToIndex[name] = len(self._boundaryFunctions)
        self._boundaryFunctions.append(boundaryFunction)

    def invalidateIndexCache(self):
        self._boundarySweeps = []

    def clear(self):
        np.fill(self._fluidFlag)
        self.invalidateIndexCache()

    def getFlag(self, name):
        return 2 ** self._nameToIndex[name]

    def setBoundary(self, name, indexExpr):
        if not isinstance(name, str):
            name = name.__name__

        flag = self.getFlag(name)
        self.flagField[indexExpr] = flag
        self.invalidateIndexCache()

    def prepare(self):
        self.invalidateIndexCache()
        for boundaryIdx, boundaryFunc in enumerate(self._boundaryFunctions):
            idxField = createBoundaryIndexList(self.flagField, self._ghostLayers, self._latticeModel.stencil,
                                               2 ** boundaryIdx, self._fluidFlag)
            ast = generateBoundaryHandling(self._symbolicPdfField, idxField, self._latticeModel, boundaryFunc)
            self._boundarySweeps.append(makePythonFunction(ast, {'indexField': idxField}))

    def __call__(self, pdfField):
        assert tuple(self._shapeWithGhostLayers) == tuple(pdfField.shape[:-1])
        if len(self._boundarySweeps) == 0:
            self.prepare()
        for boundarySweep in self._boundarySweeps:
            boundarySweep(pdfs=pdfField)
