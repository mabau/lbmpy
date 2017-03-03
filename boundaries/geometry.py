import numpy as np
import scipy.misc
from pystencils.slicing import makeSlice, normalizeSlice, shiftSlice


class BlackAndWhiteImageBoundary:

    def __init__(self, imagePath, boundaryFunction, targetSlice=makeSlice[:, :, :]):
        self.imgArr = scipy.misc.imread(imagePath, flatten=True).astype(int)
        self.imgArr = np.rot90(self.imgArr, 3)

        self._boundaryFunction = boundaryFunction
        self._targetSlice = targetSlice

    def __call__(self, boundaryHandling, method, domainSize, **kwargs):
        normalizedSlice = normalizeSlice(self._targetSlice, domainSize)
        normalizedSlice = shiftSlice(normalizedSlice, 1)
        targetSize = [s.stop - s.start for s in normalizedSlice]
        img = scipy.misc.imresize(self.imgArr, size=targetSize)
        img[img <= 254] = 0
        img[img > 254] = 1
        boundaryHandling.setBoundary(self._boundaryFunction, normalizedSlice, maskArr=(img == 0))

