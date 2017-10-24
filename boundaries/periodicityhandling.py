from lbmpy.stencils import getStencil


class PeriodicityHandling(object):
    def __init__(self, fieldShape, periodicity=(False, False, False)):
        self._spatialShape = fieldShape[:-1]
        self._indexShape = fieldShape[-1]
        self._periodicity = list(periodicity)
        self._periodicityDirty = False
        self._periodicityKernels = []

    @property
    def periodicity(self):
        """List that indicates for x,y (z) coordinate if domain is periodic in that direction"""
        return self._periodicity

    def setPeriodicity(self, x=False, y=False, z=False):
        """Enable periodic boundary conditions at the border of the domain"""
        for d in (x, y, z):
            assert isinstance(d, bool)

        self._periodicity = [x, y, z]
        self._periodicityDirty = True

    def __call__(self, **kwargs):
        if self._periodicityDirty:
            self.prepare()
        for k in self._periodicityKernels:
            k(**kwargs)

    def prepare(self):
        if not self._periodicityDirty:
            return

        self._periodicityKernels = []
        dim = len(self.flagField.shape)
        if dim == 2:
            stencil = getStencil("D2Q9")
        elif dim == 3:
            stencil = getStencil("D3Q27")
        else:
            assert False

        filteredStencil = []
        for direction in stencil:
            useDirection = True
            if direction == (0, 0) or direction == (0, 0, 0):
                useDirection = False
            for component, periodicity in zip(direction, self._periodicity):
                if not periodicity and component != 0:
                    useDirection = False
            if useDirection:
                filteredStencil.append(direction)

        if len(filteredStencil) > 0:
            if self._target == 'cpu':
                from pystencils.slicing import getPeriodicBoundaryFunctor
                self._periodicityKernels.append(getPeriodicBoundaryFunctor(filteredStencil, ghostLayers=1))
            elif self._target == 'gpu':
                from pystencils.gpucuda.periodicity import getPeriodicBoundaryFunctor
                self._periodicityKernels.append(getPeriodicBoundaryFunctor(filteredStencil, self._spatialShape,
                                                                           indexDimensions=1,
                                                                           indexDimShape=self._indexShape,
                                                                           ghostLayers=1))
            else:
                assert False

        self._periodicityDirty = False
