import sympy as sp
import abc
from lbmpy.stencils import inverseDirection
from pystencils import Field


# ------------------------------------------------ Interface -----------------------------------------------------------
from pystencils.astnodes import LoopOverCoordinate


class PdfFieldAccessor(object):
    """
    Defines how data is read and written in an LBM time step.

    Examples for PdfFieldAccessors are
         - stream pull using two fields (source/destination)
         - inplace collision access, without streaming
         - esoteric twist single field update
         -
    """
    @abc.abstractmethod
    def read(self, field, stencil):
        """Returns sequence of field accesses for all stencil values where pdfs are read from"""
        pass

    @abc.abstractmethod
    def write(self, field, stencil):
        """Returns sequence  of field accesses for all stencil values where pdfs are written to"""
        pass


# ----------------------------------------------- Implementation -------------------------------------------------------


class CollideOnlyInplaceAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        return [field(i) for i in range(len(stencil))]

    @staticmethod
    def write(field, stencil):
        return [field(i) for i in range(len(stencil))]


class StreamPullTwoFieldsAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        return [field[inverseDirection(d)](i) for i, d in enumerate(stencil)]

    @staticmethod
    def write(field, stencil):
        return [field(i) for i in range(len(stencil))]


class Pseudo2DTwoFieldsAccessor(PdfFieldAccessor):
    """Useful if a 3D simulation of a domain with size (x,y,1) is done and the dimensions with size 1 
    is periodic. In this case no periodicity exchange has to be done"""
    def __init__(self, collapsedDim):
        self._collapsedDim = collapsedDim

    def read(self, field, stencil):
        result = []
        for i, d in enumerate(stencil):
            direction = list(d)
            direction[self._collapsedDim] = 0
            result.append(field[inverseDirection(tuple(direction))](i))
        return result

    @staticmethod
    def write(field, stencil):
        return [field(i) for i in range(len(stencil))]


class PeriodicTwoFieldsAccessor(PdfFieldAccessor):
    """Access scheme that builds periodicity into the kernel, by introducing a condition on every load,
    such that at the borders the periodic value is loaded. The periodicity is specified as a tuple of booleans, one for
    each direction. The second parameter `ghostLayers` specifies the number of assumed ghost layers of the field. 
    For the periodic kernel itself no ghost layers are required, however other kernels might need them. 
    """
    def __init__(self, periodicity, ghostLayers=0):
        self._periodicity = periodicity
        self._ghostLayers = ghostLayers

    def read(self, field, stencil):
        result = []
        for i, d in enumerate(stencil):
            pullDirection = inverseDirection(d)
            periodicPullDirection = []
            for coordId, dirElement in enumerate(pullDirection):
                if not self._periodicity[coordId]:
                    periodicPullDirection.append(dirElement)
                    continue

                lowerLimit = self._ghostLayers
                upperLimit = field.spatialShape[coordId] - 1 - self._ghostLayers
                limitDiff = upperLimit - lowerLimit
                loopCounter = LoopOverCoordinate.get_loop_counter_symbol(coordId)
                if dirElement == 0:
                    periodicPullDirection.append(0)
                elif dirElement == 1:
                    newDirElement = sp.Piecewise((dirElement, loopCounter < upperLimit), (-limitDiff, True))
                    periodicPullDirection.append(newDirElement)
                elif dirElement == -1:
                    newDirElement = sp.Piecewise((dirElement, loopCounter > lowerLimit), (limitDiff, True))
                    periodicPullDirection.append(newDirElement)
                else:
                    raise NotImplementedError("This accessor supports only nearest neighbor stencils")
            result.append(field[tuple(periodicPullDirection)](i))
        return result

    @staticmethod
    def write(field, stencil):
        return [field(i) for i in range(len(stencil))]


class AABBEvenTimeStepAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        return [field(i) for i in range(len(stencil))]

    @staticmethod
    def write(field, stencil):
        return [field(stencil.index(inverseDirection(d))) for d in stencil]


class AABBOddTimeStepAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        res = []
        for i, d in enumerate(stencil):
            invDir = inverseDirection(d)
            fieldAccess = field[invDir](stencil.index(invDir))
            res.append(fieldAccess)
        return

    @staticmethod
    def write(field, stencil):
        return [field[d](i) for i, d in enumerate(stencil)]


class EsotericTwistAccessor(PdfFieldAccessor):
    @staticmethod
    def read(field, stencil):
        result = []
        for i, direction in enumerate(stencil):
            direction = inverseDirection(direction)
            neighborOffset = tuple([-e if e <= 0 else 0 for e in direction])
            result.append(field[neighborOffset](i))
        return result

    @staticmethod
    def write(field, stencil):
        result = []
        for i, direction in enumerate(stencil):
            neighborOffset = tuple([e if e >= 0 else 0 for e in direction])
            inverseIndex = stencil.index(inverseDirection(direction))
            result.append(field[neighborOffset](inverseIndex))
        return result


# -------------------------------------------- Visualization -----------------------------------------------------------


def visualizeFieldMapping(axes, stencil, fieldMapping, color='b'):
    from lbmpy.gridvisualization import Grid
    grid = Grid(3, 3)
    grid.fillWithDefaultArrows()
    for fieldAccess, direction in zip(fieldMapping, stencil):
        fieldPosition = stencil[fieldAccess.index[0]]
        neighbor = fieldAccess.offsets
        grid.addArrow((1 + neighbor[0], 1 + neighbor[1]),
                      arrowPosition=fieldPosition, arrowDirection=direction, color=color)
    grid.draw(axes)


def visualizePdfFieldAccessor(pdfFieldAccessor, figure=None):
    from lbmpy.stencils import getStencil

    if figure is None:
        import matplotlib.pyplot as plt
        figure = plt.gcf()

    stencil = getStencil('D2Q9')

    figure.patch.set_facecolor('white')

    field = Field.createGeneric('f', spatialDimensions=2, indexDimensions=1)
    preCollisionAccesses = pdfFieldAccessor.read(field, stencil)
    postCollisionAccesses = pdfFieldAccessor.write(field, stencil)

    axLeft = figure.add_subplot(1, 2, 1)
    axRight = figure.add_subplot(1, 2, 2)

    visualizeFieldMapping(axLeft, stencil, preCollisionAccesses, color='k')
    visualizeFieldMapping(axRight, stencil, postCollisionAccesses, color='r')

    axLeft.set_title("Read")
    axRight.set_title("Write")



