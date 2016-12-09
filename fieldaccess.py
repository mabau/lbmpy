import sympy as sp
import numpy as np
from lbmpy.latticemodel import LbmCollisionRule
from pystencils import Field


# --------------------------------- Field Access patterns --------------------------------------------------------------

def streamPullSourceDestination(lbmCollisionRule, srcField, dstField):
    lm = lbmCollisionRule.latticeModel

    pdfSrcSymbols = lm.pdfSymbols
    pdfDstSymbols = lm.pdfDestinationSymbols
    substitutions = {}

    for idx, offset in enumerate(lm.stencil):
        inverseIdx = tuple([-d for d in offset])
        substitutions[pdfSrcSymbols[idx]] = srcField[inverseIdx](idx)
        substitutions[pdfDstSymbols[idx]] = dstField(idx)

    return lbmCollisionRule.newWithSubstitutions(substitutions)


def streamPullWithSourceAndDestinationFields(lbmCollisionRule, numpyField=None, srcFieldName="src", dstFieldName="dst",
                                             genericLayout='numpy', genericFieldType=np.float64):
    """
    Implements a stream-pull scheme, where values are read from source and written to destination field
    :param lbmCollisionRule: a collision rule created by the lattice model
    :param numpyField: optional numpy field for PDFs. Used to create a kernel of fixed loop bounds and strides
                       if None, a generic kernel is created
    :param srcFieldName: name of the pdf source field
    :param dstFieldName: name of the pdf destination field
    :param genericLayout: if no numpyField is given to determine the layout, a variable sized field with the given
                          genericLayout is used
    :param genericFieldType: if no numpyField is given, this data type is used for the fields
    :return: new lbm collision rule, where generic pdf references are replaced by field accesses
    """
    lm = lbmCollisionRule.latticeModel
    if numpyField is not None:
        assert len(numpyField.shape) == lm.dim + 1

    if numpyField is None:
        src = Field.createGeneric(srcFieldName, lm.dim, indexDimensions=1, layout=genericLayout, dtype=genericFieldType)
        dst = Field.createGeneric(dstFieldName, lm.dim, indexDimensions=1, layout=genericLayout, dtype=genericFieldType)
    else:
        src = Field.createFromNumpyArray(srcFieldName, numpyField, indexDimensions=1)
        dst = Field.createFromNumpyArray(dstFieldName, numpyField, indexDimensions=1)
    return streamPullSourceDestination(lbmCollisionRule, src, dst)


# ---------------------------- Macroscopic value I/O to fields ---------------------------------------------------------


def addVelocityFieldOutput(lbmCollisionRule, velocityField, shiftForceIfNecessary=True):
    lm = lbmCollisionRule.latticeModel
    rho = lm.symbolicDensity
    u = lm.symbolicVelocity

    if hasattr(lm.forceModel, "macroscopicVelocity") and shiftForceIfNecessary:
        macroscopicVelocity = lm.forceModel.macroscopicVelocity(lm, u, rho)
    else:
        macroscopicVelocity = u

    newEqs = [sp.Eq(velocityField(i), u_i) for i, u_i in enumerate(macroscopicVelocity)]
    return LbmCollisionRule(lbmCollisionRule.updateEquations+newEqs, lbmCollisionRule.subexpressions,
                            lm, lbmCollisionRule.updateEquationDirections)


def addScalarOutput(lbmCollisionRule, symbol, outputField):
    eq = sp.Eq(outputField(0), symbol)
    return lbmCollisionRule.newWithSubexpressions(lbmCollisionRule.updateEquations, [eq])


def addDensityFieldOutput(lbmCollisionRule, densityField):
    lm = lbmCollisionRule.latticeModel
    newEqs = [sp.Eq(densityField(0), lm.symbolicDensity)]
    return LbmCollisionRule(lbmCollisionRule.updateEquations+newEqs, lbmCollisionRule.subexpressions, lm,
                            lbmCollisionRule.updateEquationDirections)
