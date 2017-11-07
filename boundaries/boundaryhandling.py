import numpy as np
from lbmpy.boundaries.handlinginterface import GenericBoundaryHandling, FlagFieldInterface
from lbmpy.boundaries.periodicityhandling import PeriodicityHandling


class BoundaryHandling(PeriodicityHandling, GenericBoundaryHandling):  # important: first periodicity, then boundary

    def __init__(self, pdfField, domainShape, lbMethod, ghostLayers=1, target='cpu', openMP=True, flagDtype=np.uint32):
        shapeWithGl = [a + 2 * ghostLayers for a in domainShape]
        self.domainShape = domainShape
        self.domainShapeWithGhostLayers = shapeWithGl
        flagInterface = NumpyFlagFieldInterface(shapeWithGl, flagDtype)

        GenericBoundaryHandling.__init__(self, flagInterface, pdfField, lbMethod, None, ghostLayers, target, openMP)
        PeriodicityHandling.__init__(self, list(domainShape) + [len(lbMethod.stencil)], target=target)

    def __call__(self, *args, **kwargs):
        for cls in BoundaryHandling.__bases__:
            cls.__call__(self, *args, **kwargs)

    def prepare(self):
        for cls in BoundaryHandling.__bases__:
            cls.prepare(self)

# ----------------------------------------------------------------------------------------------------------------------


class NumpyFlagFieldInterface(FlagFieldInterface):

    def __init__(self, shape, dtype=np.uint32):
        self._array = np.ones(shape, dtype=dtype)
        self._boundaryObjectToName = {}
        self._boundaryObjectToFlag = {'fluid': dtype(2**0)}
        self._nextFreeExponent = 1

    @property
    def array(self):
        return self._array

    def getFlag(self, boundaryObject):
        if boundaryObject not in self._boundaryObjectToFlag:
            self._boundaryObjectToFlag[boundaryObject] = 2 ** self._nextFreeExponent
            self._nextFreeExponent += 1
            name = self._makeBoundaryName(boundaryObject, self._boundaryObjectToName.values())
            self._boundaryObjectToName[boundaryObject] = name

        return self._boundaryObjectToFlag[boundaryObject]

    def getName(self, boundaryObject):
        return self._boundaryObjectToName[boundaryObject]

    @property
    def boundaryObjects(self):
        return self._boundaryObjectToName.keys()

    def clear(self):
        self._array.fill(0)
        self._boundaryObjectToName = {}
        self._boundaryObjectToFlag = {'fluid': np.dtype(self._array.dtype)(2**0)}
        self._nextFreeExponent = 1

