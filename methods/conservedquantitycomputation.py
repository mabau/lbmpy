import abc
import sympy as sp
from pystencils.equationcollection import EquationCollection


class AbstractConservedQuantityComputation(metaclass=abc.ABCMeta):
    """

    This class defines how conserved quantities are computed as functions of the pdfs.
    Conserved quantities are used for output and as input to the equilibrium in the collision step

    Depending on the method they might also be computed slightly different, e.g. due to a force model.

    An additional method describes how to get the conserved quantities for the equilibrium for initialization.
    In most cases the inputs can be used directly, but for some methods they have to be altered slightly.
    For example in zero centered hydrodynamic schemes with force model, the density has
    to be decreased by one, and the given velocity has to be shifted dependent on the force.

    .. image:: moment_shift.svg

    """

    @abc.abstractproperty
    def conservedQuantities(self):
        """
        Dict, mapping names (symbol) to dimensionality (int)
        For example: {'density' : 1, 'velocity' : 3}
        The naming strings can be used in :func:`outputEquationsFromPdfs`
        and :func:`equilibriumInputEquationsFromInitValues`
        """

    def definedSymbols(self, order='all'):
        """
        Returns a dict, mapping names of conserved quantities to their symbols
        """

    @abc.abstractproperty
    def defaultValues(self):
        """
        Returns a dict of symbol to default value, where "default" means that
        the equilibrium simplifies to the weights if these values are inserted.
        Hydrodynamic example: rho=1, u_i = 0
        """

    @abc.abstractmethod
    def equilibriumInputEquationsFromPdfs(self, pdfs):
        """
        Returns an equation collection that defines all necessary quantities to compute the equilibrium as functions
        of the pdfs.
        For hydrodynamic LBM schemes this is usually the density and velocity.

        :param pdfs: values or symbols for the pdf values
        """

    @abc.abstractmethod
    def outputEquationsFromPdfs(self, pdfs, outputQuantityNames):
        """
        Returns an equation collection that defines conserved quantities for output. These conserved quantities might
        be slightly different that the ones used as input for the equilibrium e.g. due to a force model.

        :param pdfs: values for the pdf entries
        :param outputQuantityNames: list of conserved quantity names, defining which parameters should be written out.
                                    See :func:`conservedQuantities`
        """

    @abc.abstractmethod
    def equilibriumInputEquationsFromInitValues(self, **kwargs):
        """
        Returns an equation collection that defines all necessary quantities to compute the equilibrium as function of
        given conserved quantities. Parameters can be names that are given by
        symbol names of :func:`conservedQuantities`.
        For all parameters not specified each implementation should use sensible defaults. For example hydrodynamic
        schemes use density=1 and velocity=0.
        """


class DensityVelocityComputation(AbstractConservedQuantityComputation):
    def __init__(self, stencil, compressible, forceModel=None,
                 zerothOrderMomentSymbol=sp.Symbol("rho"),
                 firstOrderMomentSymbols=sp.symbols("u_:3")):
        dim = len(stencil[0])
        self._stencil = stencil
        self._compressible = compressible
        self._forceModel = forceModel
        self._symbolOrder0 = zerothOrderMomentSymbol
        self._symbolsOrder1 = firstOrderMomentSymbols[:dim]

    @property
    def conservedQuantities(self):
        return {'density': 1,
                'velocity': 3}

    def definedSymbols(self, order='all'):
        if order == 'all':
            return {'density': self._symbolOrder0,
                    'velocity': self._symbolsOrder1}
        elif order == 0:
            return 'density', self._symbolOrder0
        elif order == 1:
            return 'velocity', self._symbolsOrder1
        else:
            return None

    @property
    def defaultValues(self):
        result = {self._symbolOrder0: 1}
        for s in self._symbolsOrder1:
            result[s] = 0
        return result

    def equilibriumInputEquationsFromPdfs(self, pdfs):
        dim = len(self._stencil[0])
        eqColl = getEquationsForZerothAndFirstOrderMoment(self._stencil, pdfs, self._symbolOrder0,
                                                          self._symbolsOrder1[:dim])
        if self._compressible:
            eqColl = divideFirstOrderMomentsByRho(eqColl, dim)

        eqColl = applyForceModelShift('equilibriumVelocityShift', dim, eqColl, self._forceModel, self._compressible)
        return eqColl

    def equilibriumInputEquationsFromInitValues(self, density=1, velocity=[0, 0, 0]):
        dim = len(self._stencil[0])
        zerothOrderMoment = density
        firstOrderMoments = velocity[:dim]
        velOffset = [0] * dim
        if self._compressible:
            if self._forceModel and hasattr(self._forceModel, 'macroscopicVelocityShift'):
                velOffset = self._forceModel.macroscopicVelocityShift(zerothOrderMoment)
        else:
            if self._forceModel and hasattr(self._forceModel, 'macroscopicVelocityShift'):
                velOffset = self._forceModel.macroscopicVelocityShift(sp.Rational(1, 1))
            zerothOrderMoment -= sp.Rational(1, 1)
        firstOrderMoments = [a - b for a, b in zip(firstOrderMoments, velOffset)]

        eqs = [sp.Eq(self._symbolOrder0, zerothOrderMoment)]
        eqs += [sp.Eq(l, r) for l, r in zip(self._symbolsOrder1, firstOrderMoments)]
        return EquationCollection(eqs, [])

    def outputEquationsFromPdfs(self, pdfs, outputQuantityNames):
        outputQuantityNames = set(outputQuantityNames)

        dim = len(self._stencil[0])
        eqColl = getEquationsForZerothAndFirstOrderMoment(self._stencil, pdfs, self._symbolOrder0,
                                                          self._symbolsOrder1[:dim])
        if self._compressible:
            eqColl = divideFirstOrderMomentsByRho(eqColl, dim)
        else:
            eqColl = addDensityOffset(eqColl)

        eqColl = applyForceModelShift('macroscopicVelocityShift', dim, eqColl, self._forceModel, self._compressible)

        nameToSymbol = {'density': self._symbolOrder0,
                        'velocity': self._symbolsOrder1}
        return eqColl.extract({nameToSymbol[e] for e in outputQuantityNames})

    def __repr__(self):
        return "ConservedValueComputation for %s" % (", " .join(self.conservedQuantities.keys()),)


# -----------------------------------------  Helper functions ----------------------------------------------------------


def getEquationsForZerothAndFirstOrderMoment(stencil, symbolicPdfs, symbolicZerothMoment, symbolicFirstMoments):
    """
    Returns an equation system that computes the zeroth and first order moments with the least amount of operations

    The first equation of the system is equivalent to

    .. math :

        \rho = \sum_{d \in S} f_d
        u_j = \sum_{d \in S} f_d u_jd

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
    newEqs = [oldEqs[0]] + newFirstOrderMomentEq + oldEqs[dim+1:]
    return equationCollection.newWithAdditionalSubexpressions(newEqs, [newSubExpression])


def addDensityOffset(equationCollection, offset=sp.Rational(1, 1)):
    """
    Assumes that first equation is the density (zeroth moment). Changes the density equations by adding offset to it.
    """
    oldEqs = equationCollection.mainEquations
    newDensity = sp.Eq(oldEqs[0].lhs, oldEqs[0].rhs + offset)
    return equationCollection.newWithAdditionalSubexpressions([newDensity] + oldEqs[1:], [])


def applyForceModelShift(shiftMemberName, dim, equationCollection, forceModel, compressible, reverse=False):
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
        velOffsets = shiftFunc(density)
        if reverse:
            velOffsets = [-v for v in velOffsets]
        shiftedVelocityEqs = [sp.Eq(oldEq.lhs, oldEq.rhs + offset) for oldEq, offset in zip(oldVelEqs, velOffsets)]
        newEqs = [oldEqs[0]] + shiftedVelocityEqs + oldEqs[dim + 1:]
        return equationCollection.newWithAdditionalSubexpressions(newEqs, [])
    else:
        return equationCollection




