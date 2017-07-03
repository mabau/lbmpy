import numpy as np
import waLBerla as wlb
from lbmpy.boundaries.handlinginterface import GenericBoundaryHandling, FlagFieldInterface
from lbmpy.parallel.blockiteration import slicedBlockIteration
from pystencils.slicing import normalizeSlice


class BoundaryHandling(object):
    def __init__(self, blocks, lbMethod, pdfFieldId, flagFieldId, boundaryId='boundary', target='cpu', openMP=True):

        def addBoundaryHandling(block, *args, **kwargs):
            flagFieldInterface = WalberlaFlagFieldInterface(block, flagFieldId)
            pdfField = wlb.field.toArray(block[pdfFieldId], withGhostLayers=True)
            ghostLayers = block[pdfFieldId].nrOfGhostLayers
            return GenericBoundaryHandling(flagFieldInterface, pdfField, lbMethod, ghostLayers=ghostLayers,
                                           target=target, openMP=openMP)

        self._boundaryId = boundaryId
        self._pdfFieldId = pdfFieldId
        self._blocks = blocks
        self.dim = lbMethod.dim
        blocks.addBlockData(boundaryId, addBoundaryHandling)

    def __call__(self, **kwargs):
        for block in self._blocks:
            kwargs['pdfs'] = wlb.field.toArray(block[self._pdfFieldId], withGhostLayers=True)
            block[self._boundaryId](**kwargs)

    def prepare(self):
        for block in self._blocks:
            block[self._boundaryId].prepare()

    def setBoundary(self, boundaryObject, indexExpr=None, maskCallback=None, sliceNormalizationGhostLayers=1):
        if indexExpr is None:
            indexExpr = [slice(None, None, None)] * self.dim

        if len(self._blocks) == 0:
            return

        gl = self._blocks[0][self._boundaryId].ghostLayers
        ngl = sliceNormalizationGhostLayers
        for block in self._blocks:
            block[self._boundaryId].reserveFlag(boundaryObject)

        for block, indexArrs, localSlice in slicedBlockIteration(self._blocks, indexExpr, gl, ngl):
            mask = maskCallback(*indexArrs) if maskCallback else None
            block[self._boundaryId].setBoundary(boundaryObject, indexExpr=localSlice, maskArr=mask)

        #domainCellBB = self._blocks.getDomainCellBB()
        #domainExtent = [s + 2 * sliceNormalizationGhostLayers for s in domainCellBB.size]
        #indexExpr = normalizeSlice(indexExpr, domainExtent)
        #targetCellBB = wlb.CellInterval.fromSlice(indexExpr)
        #targetCellBB.shift(*[a - sliceNormalizationGhostLayers for a in domainCellBB.min])
        #
        #for block in self._blocks:
        #    boundaryHandling = block[self._boundaryId]
        #    ghostLayers = boundaryHandling.ghostLayers
        #    intersection = self._blocks.getBlockCellBB(block).getExpanded(ghostLayers)
        #    intersection.intersect(targetCellBB)
        #    if not intersection.empty():
        #        if maskCallback is not None:
        #            meshGridParams = [offset + 0.5 + np.arange(width)
        #                              for offset, width in zip(intersection.min, intersection.size)]
        #            indexArr = np.meshgrid(*meshGridParams, indexing='ij')
        #            mask = maskCallback(*indexArr)
        #        else:
        #            mask = None
        #        localTargetBB = self._blocks.transformGlobalToLocal(block, intersection)
        #        localTargetBB.shift(ghostLayers, ghostLayers, ghostLayers)
        #        block[self._boundaryId].setBoundary(boundaryObject, indexExpr=localTargetBB.toSlice(), maskArr=mask)


# ----------------------------------------------------------------------------------------------------------------------


class WalberlaFlagFieldInterface(FlagFieldInterface):
    def __init__(self, block, flagFieldId):
        self._block = block
        self._flagFieldId = flagFieldId
        self._flagArray = wlb.field.toArray(block[self._flagFieldId], withGhostLayers=True)
        assert self._flagArray.shape[3] == 1
        self._flagArray = self._flagArray[..., 0]

        self._fluidFlag = self._block[self._flagFieldId].registerFlag('fluid')
        self._flagArray.fill(self._fluidFlag)
        self._boundaryObjectToName = {}

    def getFlag(self, boundaryObject):
        if boundaryObject == 'fluid':
            return self._fluidFlag
        if boundaryObject not in self._boundaryObjectToName:
            name = self._makeBoundaryName(boundaryObject, self._boundaryObjectToName.values())
            self._boundaryObjectToName[boundaryObject] = name
            return self._block[self._flagFieldId].registerFlag(name)
        else:
            return self._block[self._flagFieldId].flag(self._boundaryObjectToName[boundaryObject])

    def getName(self, boundaryObject):
        return self._boundaryObjectToName[boundaryObject]

    @property
    def array(self):
        return self._flagArray

    @property
    def boundaryObjects(self):
        return self._boundaryObjectToName.keys()

    def clear(self):
        self._flagArray.fill(self._fluidFlag)
        raise NotImplementedError()


if __name__ == '__main__':
    from lbmpy.creationfunctions import createLatticeBoltzmannMethod
    from lbmpy.boundaries.boundaryconditions import NoSlip
    from pystencils.slicing import makeSlice

    blocks = wlb.createUniformBlockGrid(cellsPerBlock=(4, 4, 4), blocks=(2, 2, 2), oneBlockPerProcess=False)

    lbMethod = createLatticeBoltzmannMethod(stencil='D3Q19', method='srt', relaxationRate=1.8)

    wlb.field.addFlagFieldToStorage(blocks, 'flagField', nrOfBits=8, ghostLayers=1)
    wlb.field.addToStorage(blocks, 'pdfField', float, fSize=len(lbMethod.stencil), ghostLayers=1)

    bh = BoundaryHandling(blocks, lbMethod, 'pdfField', 'flagField')


    def maskCallback(x, y, z):
        midPoint = (4, 4, 4)
        radius = 2.0

        resultMask = (x - midPoint[0]) ** 2 + \
                     (y - midPoint[1]) ** 2 + \
                     (z - midPoint[2]) ** 2 <= radius ** 2
        return resultMask


    bh.setBoundary(NoSlip(), makeSlice[:, :, :], maskCallback, sliceNormalizationGhostLayers=0)
    vtkOutput = wlb.vtk.makeOutput(blocks, "vtkOutput")
    flagFieldWriter = wlb.field.createVTKWriter(blocks, "flagField")
    vtkOutput.addCellDataWriter(flagFieldWriter)
    vtkOutput.write(1)