import sympy as sp
from pystencils.equationcollection import EquationCollection


def getEquationsForZerothAndFirstOrderMoment(stencil, symbolicPdfs, symbolicZerothMoment, symbolicFirstMoments):
    """
    Returns an equation system that computes the zeroth and first order moments with the least amount of operations

    The first equation of the system is equivalent to

    .. math :

        \rho = \sum_{d \in S} f_d
        \u_j = \sum_{d \in S} f_d u_jd

    :param stencil: called :math:`S` above
    :param symbolicPdfs: called :math:`f` above
    :param symbolicZerothMoment:  called :math:`\rho` above
    :param symbolicFirstMoments: called :math:`u` above
    """
    def filterOutPlusTerms(expr):
        result = 0
        for term in expr.args:
            if not type(term) is sp.Mul:
                result += term
        return result

    dim = len(stencil[0])

    subexpressions = []
    pdfSum = sum(symbolicPdfs)
    u = [0] * dim
    for f, offset in zip(symbolicPdfs, stencil):
        for i in range(dim):
            u[i] += f * int(offset[i])

    plusTerms = [set(filterOutPlusTerms(u_i).args) for u_i in u]
    for i in range(dim):
        rhs = plusTerms[i]
        for j in range(i):
            rhs -= plusTerms[j]
        eq = sp.Eq(sp.Symbol("vel%dTerm" % (i,)), sum(rhs))
        subexpressions.append(eq)

    for subexpression in subexpressions:
        pdfSum = pdfSum.subs(subexpression.rhs, subexpression.lhs)

    for i in range(dim):
        u[i] = u[i].subs(subexpressions[i].rhs, subexpressions[i].lhs)

    equations = []
    equations += [sp.Eq(symbolicZerothMoment, pdfSum)]
    equations += [sp.Eq(u_i_sym, u_i) for u_i_sym, u_i in zip(symbolicFirstMoments, u)]

    return EquationCollection(equations, subexpressions)


def shiftMomentsForEquilibrium(dim, equationCollection, forceModel):
    """
    :param equationCollection: assumes that equations[1:dim+1] are the velocities
    :param forceModel:
    :return:
    """
    pass


def shiftMomentsForMacroscopicValues(equationCollection, forceModel):
    """
    :param equationCollection: assumes that equations[1:dim+1] are the velocities
    :param forceModel:
    :return:
    """
    pass


def densityVelocityExpressionsForEquilibrium(stencil, symbolicPdfs, compressible,
                                             symbolicDensity, symbolicVelocities):
    """
    Returns an equation collection, containing equations to compute density, velocity and pressure from pdf values
    """

    eqColl = getEquationsForZerothAndFirstOrderMoment(stencil, symbolicPdfs, symbolicDensity, symbolicVelocities)
    if compressible:
        eqColl = divideFirstOrderMomentsByRho(eqColl)
    if forceModel is not None:
        shiftMomentsForEquilibrium(len(stencil[0]), eqColl, forceModel)
    return eqColl


def densityVelocityExpressionsForOutput(stencil, symbolicPdfs, compressible,
                                        symbolicDensity, symbolicVelocities):
    """
    Returns an equation collection, containing equations to compute density, velocity and pressure from pdf values
    """

    momentEqCollection = getEquationsForZerothAndFirstOrderMoment(stencil, symbolicPdfs,
                                                                  symbolicDensity, symbolicVelocities)
    eqColl = getEquationsForZerothAndFirstOrderMoment(stencil, symbolicPdfs, symbolicDensity, symbolicVelocities)
    if compressible:
        eqColl = divideFirstOrderMomentsByRho(eqColl)
        addDensityOffset()
    if forceModel is not None:
        shiftMomentsForEquilibrium(len(stencil[0]), eqColl, forceModel)
    return eqColl