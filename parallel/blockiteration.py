import numpy as np
import waLBerla as wlb
from pystencils.slicing import normalizeSlice


def slicedBlockIteration(blocks, sliceObj=None, ghostLayers=1, sliceNormalizationGhostLayers=1,
                         withIndexArrays=False):
    """
    Iterates of all blocks that have an intersection with the given slice object.
    For these blocks a slice object in local block coordinates is given
    
    :param blocks: waLBerla block data structure
    :param sliceObj: a slice (i.e. rectangular subregion), can be created with makeSlice[]
    :param ghostLayers: how many ghost layers are included in the local slice and the optional index arrays
    :param sliceNormalizationGhostLayers: slices can have relative coordinates e.g. makeSlice[0.2, :, :]
                                          when computing absolute values, the domain size is needed. This parameter 
                                          specifies how many ghost layers are taken into account for this operation.
    :param withIndexArrays: if true index arrays [x,y,z] are yielded as last parameters. These arrays contain the
                            cell midpoints in global coordinates
    """
    if sliceObj is None:
        sliceObj = [sliceObj(None, None, None)] * 3

    domainCellBB = blocks.getDomainCellBB()
    domainExtent = [s + 2 * sliceNormalizationGhostLayers for s in domainCellBB.size]
    sliceObj = normalizeSlice(sliceObj, domainExtent)
    targetCellBB = wlb.CellInterval.fromSlice(sliceObj)
    targetCellBB.shift(*[a - sliceNormalizationGhostLayers for a in domainCellBB.min])

    for block in blocks:
        intersection = blocks.getBlockCellBB(block).getExpanded(ghostLayers)
        intersection.intersect(targetCellBB)
        if intersection.empty():
            continue

        localTargetBB = blocks.transformGlobalToLocal(block, intersection)
        localTargetBB.shift(ghostLayers, ghostLayers, ghostLayers)
        localSlice = localTargetBB.toSlice()

        if withIndexArrays:
            meshGridParams = [offset + 0.5 + np.arange(width)
                              for offset, width in zip(intersection.min, intersection.size)]
            indexArrays = np.meshgrid(*meshGridParams, indexing='ij')
            yield block, localSlice, indexArrays
        else:
            yield block, localSlice

