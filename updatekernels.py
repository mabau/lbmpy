import numpy as np
import sympy as sp
from pystencils import Field
from pystencils.field import createNumpyArrayWithLayout, layoutStringToTuple
from pystencils.sympyextensions import fastSubs
from lbmpy.fieldaccess import StreamPullTwoFieldsAccessor, Pseudo2DTwoFieldsAccessor, PeriodicTwoFieldsAccessor


# -------------------------------------------- LBM Kernel Creation -----------------------------------------------------


def createLBMKernel(collisionRule, inputField, outputField, accessor):
    """
    Replaces the pre- and post collision symbols in the collision rule by field accesses.

    :param collisionRule:  instance of LbmCollisionRule, defining the collision step
    :param inputField: field used for reading pdf values
    :param outputField: field used for writing pdf values (may be the same as input field for certain accessors)
    :param accessor: instance of PdfFieldAccessor, defining where to read and write values
                     to create e.g. a fused stream-collide kernel
    :return: LbmCollisionRule where pre- and post collision symbols have been replaced
    """
    method = collisionRule.method
    preCollisionSymbols = method.preCollisionPdfSymbols
    postCollisionSymbols = method.postCollisionPdfSymbols
    substitutions = {}

    inputAccesses = accessor.read(inputField, method.stencil)
    outputAccesses = accessor.write(outputField, method.stencil)

    for (idx, offset), inputAccess, outputAccess in zip(enumerate(method.stencil), inputAccesses, outputAccesses):
        substitutions[preCollisionSymbols[idx]] = inputAccess
        substitutions[postCollisionSymbols[idx]] = outputAccess

    result = collisionRule.copyWithSubstitutionsApplied(substitutions)

    if 'splitGroups' in result.simplificationHints:
        newSplitGroups = []
        for splitGroup in result.simplificationHints['splitGroups']:
            newSplitGroups.append([fastSubs(e, substitutions) for e in splitGroup])
        result.simplificationHints['splitGroups'] = newSplitGroups

    return result


def createStreamPullKernel(collisionRule, numpyField=None, srcFieldName="src", dstFieldName="dst",
                           genericLayout='numpy', genericFieldType=np.float64,
                           builtinPeriodicity=(False, False, False)):
    """
    Implements a stream-pull scheme, where values are read from source and written to destination field
    :param collisionRule: a collision rule created by lbm method
    :param numpyField: optional numpy field for PDFs. Used to create a kernel of fixed loop bounds and strides
                       if None, a generic kernel is created
    :param srcFieldName: name of the pdf source field
    :param dstFieldName: name of the pdf destination field
    :param genericLayout: if no numpyField is given to determine the layout, a variable sized field with the given
                          genericLayout is used
    :param genericFieldType: if no numpyField is given, this data type is used for the fields
    :param builtinPeriodicity: periodicity that should be built into the kernel
    :return: lbm update rule, where generic pdf references are replaced by field accesses
    """
    dim = collisionRule.method.dim
    if numpyField is not None:
        assert len(numpyField.shape) == dim + 1, "Field dimension mismatch: dimension is %s, should be %d" % \
                                                 (len(numpyField.shape), dim + 1)

    if numpyField is None:
        src = Field.createGeneric(srcFieldName, dim, indexDimensions=1, layout=genericLayout, dtype=genericFieldType)
        dst = Field.createGeneric(dstFieldName, dim, indexDimensions=1, layout=genericLayout, dtype=genericFieldType)
    else:
        src = Field.createFromNumpyArray(srcFieldName, numpyField, indexDimensions=1)
        dst = Field.createFromNumpyArray(dstFieldName, numpyField, indexDimensions=1)

    accessor = StreamPullTwoFieldsAccessor

    if any(builtinPeriodicity):
        accessor = PeriodicTwoFieldsAccessor(builtinPeriodicity, ghostLayers=1)
    return createLBMKernel(collisionRule, src, dst, accessor)


# ---------------------------------- Pdf array creation for various layouts --------------------------------------------


def createPdfArray(size, numDirections, ghostLayers=1, layout='fzyx'):
    """
    Creates an empty numpy array for a pdf field with the specified memory layout.

    Examples:
        >>> createPdfArray((3, 4, 5), 9, layout='zyxf', ghostLayers=0).shape
        (3, 4, 5, 9)
        >>> createPdfArray((3, 4, 5), 9, layout='zyxf', ghostLayers=0).strides
        (72, 216, 864, 8)
        >>> createPdfArray((3, 4), 9, layout='zyxf', ghostLayers=1).shape
        (5, 6, 9)
        >>> createPdfArray((3, 4), 9, layout='zyxf', ghostLayers=1).strides
        (72, 360, 8)
    """
    sizeWithGl = [s + 2 * ghostLayers for s in size]
    dim = len(size)
    if isinstance(layout, str):
        layout = layoutStringToTuple(dim+1)
    return createNumpyArrayWithLayout(sizeWithGl + [numDirections], layout)


# ------------------------------------------- Add output fields to kernel ----------------------------------------------


def addOutputFieldForConservedQuantities(collisionRule, conservedQuantitiesToOutputFieldDict):
    method = collisionRule.method
    cqc = method.conservedQuantityComputation.outputEquationsFromPdfs(method.preCollisionPdfSymbols,
                                                                      conservedQuantitiesToOutputFieldDict)
    return collisionRule.merge(cqc)


def writeQuantitiesToField(collisionRule, symbols, outputField):
    if not hasattr(symbols, "__len__"):
        symbols = [symbols]
    eqs = [sp.Eq(outputField(i), s) for i, s in enumerate(symbols)]
    return collisionRule.copy(collisionRule.mainEquations + eqs)
