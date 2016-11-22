from collections import defaultdict
import sympy as sp
import numpy as np
from pystencils.field import Field
from lbmpy.fieldaccess import streamPullWithSourceAndDestinationFields
from lbmpy.latticemodel import MomentRelaxationLatticeModel
from lbmpy.simplifications import createDefaultMomentSpaceSimplificationStrategy
import lbmpy.simplifications as simplifications


def createStreamCollideUpdateRule(lm, numpyField=None, srcFieldName="src", dstFieldName="dst",
                                  doCSE=False, genericLayout='numpy', genericFieldType=np.float64):
    """
    Creates a list of LBM update equations
    :param lm: instance of lattice model
    :param numpyField: optional numpy field for PDFs. Used to create a kernel of fixed loop bounds and strides
                       if None, a generic kernel is created
    :param srcFieldName: name of the pdf source field
    :param dstFieldName: name of the pdf destination field
    :param doCSE: if True, common subexpression elimination is done for pdfs in opposing directions
    :param genericLayout: if no numpyField is given to determine the layout, a variable sized field with the given
                          genericLayout is used
    :param genericFieldType: if no numpyField is given, this data type is used for the fields
    :return: list of sympy equations
    """
    if isinstance(lm, MomentRelaxationLatticeModel):
        simplificationStrategy = createDefaultMomentSpaceSimplificationStrategy()
        if doCSE:
            simplificationStrategy.addSimplificationRule(simplifications.cseInOpposingDirections)
    else:
        simplificationStrategy = simplifications.Strategy()
        simplificationStrategy.addSimplificationRule(simplifications.subexpressionSubstitutionInExistingSubexpressions)
        simplificationStrategy.addSimplificationRule(simplifications.sympyCSE)

    collisionRule = lm.getCollisionRule()
    collisionRule = simplificationStrategy(collisionRule)
    collisionRule = streamPullWithSourceAndDestinationFields(collisionRule, numpyField, srcFieldName, dstFieldName,
                                                             genericLayout, genericFieldType)
    return collisionRule


def createLbmSplitGroups(lm, equations):
    f_eq_common = sp.Symbol('f_eq_common')
    rho = sp.Symbol('rho')

    result = [list(lm.symbolicVelocity), ]

    if lm.compressible:
        result[0].append(rho)

    directionGroups = defaultdict(list)

    for eq in equations:
        if f_eq_common in eq.rhs.atoms(sp.Symbol) and f_eq_common not in result[0]:
            result[0].append(f_eq_common)

        if not type(eq.lhs) is Field.Access:
            continue

        idx = eq.lhs.index[0]
        if idx == 0:
            result[0].append(eq.lhs)
            continue

        direction = lm.stencil[idx]
        inverseDir = tuple([-i for i in direction])

        if inverseDir in directionGroups:
            directionGroups[inverseDir].append(eq.lhs)
        else:
            directionGroups[direction].append(eq.lhs)
    result += directionGroups.values()

    if f_eq_common not in result[0]:
        result[0].append(rho)

    return result
