from lbmpy.stencils import inverseDirection
from pystencils import Field
import abc


# ------------------------------------------------ Interface -----------------------------------------------------------


class PdfFieldAccessor:
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



