import numpy as np
import sympy as sp
from lbmpy.stencils import inverseDirection
from pystencils import TypedSymbol, createIndexedKernel
from pystencils.backends.cbackend import CustomCppCode
from pystencils.boundaries import BoundaryHandling
from pystencils.boundaries.boundaryhandling import BoundaryOffsetInfo


class LatticeBoltzmannBoundaryHandling(BoundaryHandling):

    def __init__(self, lbMethod, dataHandling, pdfFieldName, name="boundaryHandling", flagInterface=None,
                 target='cpu', openMP=True):
        self.lbMethod = lbMethod
        super(LatticeBoltzmannBoundaryHandling, self).__init__(dataHandling, pdfFieldName, lbMethod.stencil,
                                                               name, flagInterface, target, openMP)

    def forceOnBoundary(self, boundaryObj):
        from lbmpy.boundaries import NoSlip
        if isinstance(boundaryObj, NoSlip):
            return self._forceOnNoSlip(boundaryObj)
        else:
            self.__call__()
            return self._forceOnBoundary(boundaryObj)

    # ------------------------------ Implementation Details ------------------------------------------------------------

    def _forceOnNoSlip(self, boundaryObj):
        dh = self._dataHandling
        ffGhostLayers = dh.ghostLayersOfField(self.flagInterface.flagFieldName)
        method = self.lbMethod
        stencil = np.array(method.stencil)

        result = np.zeros(self.dim)

        for b in dh.iterate(ghostLayers=ffGhostLayers):
            objToIndList = b[self._indexArrayName].boundaryObjectToIndexList
            pdfArray = b[self._fieldName]
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
        ffGhostLayers = dh.ghostLayersOfField(self.flagInterface.flagFieldName)
        method = self.lbMethod
        stencil = np.array(method.stencil)
        invDirection = np.array([method.stencil.index(inverseDirection(d))
                                 for d in method.stencil])

        result = np.zeros(self.dim)

        for b in dh.iterate(ghostLayers=ffGhostLayers):
            objToIndList = b[self._indexArrayName].boundaryObjectToIndexList
            pdfArray = b[self._fieldName]
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

    def _createBoundaryKernel(self, symbolicField, symbolicIndexField, boundaryObject):
        return createLatticeBoltzmannBoundaryKernel(symbolicField, symbolicIndexField, self.lbMethod,
                                                    boundaryObject, target=self._target, openMP=self._openMP)


class LbmWeightInfo(CustomCppCode):

    # --------------------------- Functions to be used by boundaries --------------------------

    @staticmethod
    def weightOfDirection(dirIdx):
        return sp.IndexedBase(LbmWeightInfo.WEIGHTS_SYMBOL, shape=(1,))[dirIdx]

    # ---------------------------------- Internal ---------------------------------------------

    WEIGHTS_SYMBOL = TypedSymbol("weights", "double")

    def __init__(self, lbMethod):
        weights = [str(w.evalf()) for w in lbMethod.weights]
        code = "const double %s [] = { %s };\n" % (LbmWeightInfo.WEIGHTS_SYMBOL.name, ",".join(weights))
        super(LbmWeightInfo, self).__init__(code, symbolsRead=set(),
                                            symbolsDefined=set([LbmWeightInfo.WEIGHTS_SYMBOL]))


def createLatticeBoltzmannBoundaryKernel(pdfField, indexField, lbMethod, boundaryFunctor, target='cpu', openMP=True):
    elements = [BoundaryOffsetInfo(lbMethod.stencil), LbmWeightInfo(lbMethod)]
    indexArrDtype = indexField.dtype.numpyDtype
    dirSymbol = TypedSymbol("dir", indexArrDtype.fields['dir'][0])
    elements += [sp.Eq(dirSymbol, indexField[0]('dir'))]
    elements += boundaryFunctor(pdfField=pdfField, directionSymbol=dirSymbol, lbMethod=lbMethod, indexField=indexField)
    return createIndexedKernel(elements, [indexField], target=target, cpuOpenMP=openMP)
