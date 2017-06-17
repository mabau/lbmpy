import numpy as np
import waLBerla as wlb
from pystencils.slicing import normalizeSlice


def slicedBlockIteration(blocks, indexExpr=None, ghostLayers=1, sliceNormalizationGhostLayers=1):
    if indexExpr is None:
        indexExpr = [slice(None, None, None)] * 3

    domainCellBB = blocks.getDomainCellBB()
    domainExtent = [s + 2 * sliceNormalizationGhostLayers for s in domainCellBB.size]
    indexExpr = normalizeSlice(indexExpr, domainExtent)
    targetCellBB = wlb.CellInterval.fromSlice(indexExpr)
    targetCellBB.shift(*[a - sliceNormalizationGhostLayers for a in domainCellBB.min])

    for block in blocks:
        intersection = blocks.getBlockCellBB(block).getExpanded(ghostLayers)
        intersection.intersect(targetCellBB)
        if intersection.empty():
            continue

        meshGridParams = [offset + 0.5 + np.arange(width)
                          for offset, width in zip(intersection.min, intersection.size)]

        indexArrays = np.meshgrid(*meshGridParams, indexing='ij')

        localTargetBB = blocks.transformGlobalToLocal(block, intersection)
        localTargetBB.shift(ghostLayers, ghostLayers, ghostLayers)
        localSlice = localTargetBB.toSlice()
        yield block, indexArrays, localSlice
