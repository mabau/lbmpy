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

    def setBoundary(self, name, indexExpr, clearOtherBoundaries=True):
        if not isinstance(name, str):
            function = name
            if hasattr(function, '__name__'):
                name = function.__name__
            else:
                name = function.name

            if function not in self._boundaryFunctions:
                self.addBoundary(function, name)

        flag = self.getFlag(name)
        if clearOtherBoundaries:
            self.flagField[indexExpr] = flag
        else:
            # clear fluid flag
            np.bitwise_and(self.flagField[indexExpr], np.invert(self._fluidFlag), self.flagField[indexExpr])
            # add new boundary flag
            np.bitwise_or(self.flagField[indexExpr], flag, self.flagField[indexExpr])

        self.invalidateIndexCache()

    def prepare(self):
        self.invalidateIndexCache()
        for boundaryIdx, boundaryFunc in enumerate(self._boundaryFunctions):
            idxField = createBoundaryIndexList(self.flagField, self._ghostLayers, self._latticeModel.stencil,
                                               2 ** boundaryIdx, self._fluidFlag)
            ast = generateBoundaryHandling(self._symbolicPdfField, idxField, self._latticeModel, boundaryFunc)
            self._boundarySweeps.append(makePythonFunction(ast, {'indexField': idxField}))

    def __call__(self, **kwargs):
        if len(self._boundarySweeps) == 0:
            self.prepare()
        for boundarySweep in self._boundarySweeps:
            boundarySweep(**kwargs)
