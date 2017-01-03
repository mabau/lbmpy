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


def divideFirstOrderMomentsByRho(equationCollection, dim):
    """
    Assumes that the equations of the passed equation collection are the following
        - rho = f_0  + f_1 + ...
        - u_0 = ...
        - u_1 = ...
    Returns a new equation collection where the u terms (first order moments) are divided by rho.
    The dim parameter specifies the number of first order moments. All subsequent equations are just copied over.
    """
    oldEqs = equationCollection.mainEquations
    rho = oldEqs[0].lhs
    rhoInv = sp.Symbol("rhoInv")
    newSubExpression = sp.Eq(rhoInv, 1 / rho)
    newFirstOrderMomentEq = [sp.Eq(eq.lhs, eq.rhs * rhoInv) for eq in oldEqs[1:dim+1]]
    newEqs = oldEqs[0] + newFirstOrderMomentEq + oldEqs[dim+1:]
    return equationCollection.createNewWithAdditionalSubexpressions(newEqs, newSubExpression)


def addDensityOffset(equationCollection, offset=sp.Rational(1, 1)):
    """
    Assumes that first equation is the density (zeroth moment). Changes the density equations by adding offset to it.
    """
    oldEqs = equationCollection.mainEquations
    newDensity = sp.Eq(oldEqs[0].lhs, oldEqs[1].rhs + offset)
    return equationCollection.createNewWithAdditionalSubexpressions([newDensity] + oldEqs[1:], [])


def applyForceModelShift(shiftMemberName, dim, equationCollection, forceModel, compressible):
    """
    Modifies the first order moment equations in equationCollection according to the force model shift.
    It is applied if force model has a method named shiftMemberName. The equations 1: dim+1 of the passed
    equation collection are assumed to be the velocity equations.
    """
    if forceModel is not None and hasattr(forceModel, shiftMemberName):
        oldEqs = equationCollection.mainEquations
        density = oldEqs[0].lhs if compressible else sp.Rational(1, 1)
        oldVelEqs = oldEqs[1:dim + 1]
        shiftFunc = getattr(forceModel, shiftMemberName)
        shiftedVels = shiftFunc([eq.rhs for eq in oldVelEqs], density)
        shiftedVelocityEqs = [sp.Eq(oldEq.lhs, shiftedVel) for oldEq, shiftedVel in zip(oldVelEqs, shiftedVels)]
        newEqs = [oldEqs[0]] + shiftedVelocityEqs + oldEqs[dim + 1:]
        return equationCollection.createNewWithAdditionalSubexpressions(newEqs, [])
    else:
        return equationCollection


# --------------------------- Density / Velocity definitions with force shift ------------------------------------------


def densityVelocityExpressionsForEquilibrium(stencil, symbolicPdfs, compressible, symbolicDensity,
                                             symbolicVelocities, forceModel=None):
    """
    Returns an equation collection, containing equations to compute density, velocity and pressure from pdf values
    """

    eqColl = getEquationsForZerothAndFirstOrderMoment(stencil, symbolicPdfs, symbolicDensity, symbolicVelocities)
    if compressible:
        eqColl = divideFirstOrderMomentsByRho(eqColl)
    if forceModel is not None:
        eqColl = applyForceModelShift('equilibriumVelocity', len(stencil[0]), eqColl, forceModel, compressible)
    return eqColl


def densityVelocityExpressionsForOutput(stencil, symbolicPdfs, compressible, symbolicDensity,
                                        symbolicVelocities, forceModel=None):
    """
    Returns an equation collection, containing equations to compute density, velocity and pressure from pdf values
    """

    eqColl = getEquationsForZerothAndFirstOrderMoment(stencil, symbolicPdfs, symbolicDensity, symbolicVelocities)
    if compressible:
        eqColl = divideFirstOrderMomentsByRho(eqColl)
        eqColl = addDensityOffset(eqColl)
    if forceModel is not None:
        eqColl = applyForceModelShift('macroscopicVelocity', len(stencil[0]), eqColl, forceModel, compressible)
    return eqColl
