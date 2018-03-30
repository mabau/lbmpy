import abc
import sympy as sp
from collections import OrderedDict
from pystencils.assignment_collection import AssignmentCollection
from pystencils.field import Field, Assignment


class AbstractConservedQuantityComputation(abc.ABCMeta('ABC', (object,), {})):
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

    @property
    @abc.abstractmethod
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

    @property
    @abc.abstractmethod
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
    def outputEquationsFromPdfs(self, pdfs, outputQuantityNamesToSymbols):
        """
        Returns an equation collection that defines conserved quantities for output. These conserved quantities might
        be slightly different that the ones used as input for the equilibrium e.g. due to a force model.

        :param pdfs: values for the pdf entries
        :param outputQuantityNamesToSymbols: dict mapping of conserved quantity names (See :func:`conservedQuantities`)
                                            to symbols or field accesses where they should be written to
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
                'velocity': len(self._stencil[0])}

    @property
    def compressible(self):
        return self._compressible

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
    def zeroCenteredPdfs(self):
        return not self._compressible

    @property
    def zerothOrderMomentSymbol(self):
        return self._symbolOrder0

    @property
    def firstOrderMomentSymbols(self):
        return self._symbolsOrder1

    @property
    def defaultValues(self):
        result = {self._symbolOrder0: 1}
        for s in self._symbolsOrder1:
            result[s] = 0
        return result

    def equilibriumInputEquationsFromPdfs(self, pdfs):
        dim = len(self._stencil[0])
        eq_coll = getEquationsForZerothAndFirstOrderMoment(self._stencil, pdfs, self._symbolOrder0,
                                                          self._symbolsOrder1[:dim])
        if self._compressible:
            eq_coll = divideFirstOrderMomentsByRho(eq_coll, dim)

        eq_coll = applyForceModelShift('equilibriumVelocityShift', dim, eq_coll, self._forceModel, self._compressible)
        return eq_coll

    def equilibriumInputEquationsFromInitValues(self, density=1, velocity=(0, 0, 0)):
        dim = len(self._stencil[0])
        zerothOrderMoment = density
        first_order_moments = velocity[:dim]
        vel_offset = [0] * dim

        if self._compressible:
            if self._forceModel and hasattr(self._forceModel, 'macroscopicVelocityShift'):
                vel_offset = self._forceModel.macroscopicVelocityShift(zerothOrderMoment)
        else:
            if self._forceModel and hasattr(self._forceModel, 'macroscopicVelocityShift'):
                vel_offset = self._forceModel.macroscopicVelocityShift(sp.Rational(1, 1))
            zerothOrderMoment -= sp.Rational(1, 1)
        eqs = [Assignment(self._symbolOrder0, zerothOrderMoment)]

        first_order_moments = [a - b for a, b in zip(first_order_moments, vel_offset)]
        eqs += [Assignment(l, r) for l, r in zip(self._symbolsOrder1, first_order_moments)]

        return AssignmentCollection(eqs, [])

    def outputEquationsFromPdfs(self, pdfs, outputQuantityNamesToSymbols):
        dim = len(self._stencil[0])

        ac = getEquationsForZerothAndFirstOrderMoment(self._stencil, pdfs, self._symbolOrder0, self._symbolsOrder1)

        if self._compressible:
            ac = divideFirstOrderMomentsByRho(ac, dim)
        else:
            ac = addDensityOffset(ac)

        ac = applyForceModelShift('macroscopicVelocityShift', dim, ac, self._forceModel, self._compressible)

        main_assignments = []
        eqs = OrderedDict([(eq.lhs, eq.rhs) for eq in ac.allEquations])

        if 'density' in outputQuantityNamesToSymbols:
            density_output_symbol = outputQuantityNamesToSymbols['density']
            if isinstance(density_output_symbol, Field):
                density_output_symbol = density_output_symbol()
            if density_output_symbol != self._symbolOrder0:
                main_assignments.append(Assignment(density_output_symbol, self._symbolOrder0))
            else:
                main_assignments.append(Assignment(self._symbolOrder0, eqs[self._symbolOrder0]))
                del eqs[self._symbolOrder0]
        if 'velocity' in outputQuantityNamesToSymbols:
            vel_output_symbols = outputQuantityNamesToSymbols['velocity']
            if isinstance(vel_output_symbols, Field):
                field = vel_output_symbols
                vel_output_symbols = [field(i) for i in range(len(self._symbolsOrder1))]
            if tuple(vel_output_symbols) != tuple(self._symbolsOrder1):
                main_assignments += [Assignment(a, b) for a, b in zip(vel_output_symbols, self._symbolsOrder1)]
            else:
                for u_i in self._symbolsOrder1:
                    main_assignments.append(Assignment(u_i, eqs[u_i]))
                    del eqs[u_i]
        if 'momentumDensity' in outputQuantityNamesToSymbols:
            # get zeroth and first moments again - force-shift them if necessary
            # and add their values directly to the main equations assuming that subexpressions are already in
            # main equation collection
            # Is not optimal when velocity and momentumDensity are calculated together, but this is usually not the case
            momentum_density_output_symbols = outputQuantityNamesToSymbols['momentumDensity']
            mom_density_eq_coll = getEquationsForZerothAndFirstOrderMoment(self._stencil, pdfs,
                                                                        self._symbolOrder0, self._symbolsOrder1)
            mom_density_eq_coll = applyForceModelShift('macroscopicVelocityShift', dim, mom_density_eq_coll,
                                                    self._forceModel, self._compressible)
            for sym, val in zip(momentum_density_output_symbols, mom_density_eq_coll.mainAssignments[1:]):
                main_assignments.append(Assignment(sym, val.rhs))

        ac = ac.copy(main_assignments, [Assignment(a, b) for a, b in eqs.items()])
        return ac.newWithoutUnusedSubexpressions()

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
    def filter_out_plus_terms(expr):
        result = 0
        for term in expr.args:
            if not type(term) is sp.Mul:
                result += term
        return result

    dim = len(stencil[0])

    subexpressions = []
    pdf_sum = sum(symbolicPdfs)
    u = [0] * dim
    for f, offset in zip(symbolicPdfs, stencil):
        for i in range(dim):
            u[i] += f * int(offset[i])

    plus_terms = [set(filter_out_plus_terms(u_i).args) for u_i in u]
    for i in range(dim):
        rhs = plus_terms[i]
        for j in range(i):
            rhs -= plus_terms[j]
        eq = Assignment(sp.Symbol("vel%dTerm" % (i,)), sum(rhs))
        subexpressions.append(eq)

    for subexpression in subexpressions:
        pdf_sum = pdf_sum.subs(subexpression.rhs, subexpression.lhs)

    for i in range(dim):
        u[i] = u[i].subs(subexpressions[i].rhs, subexpressions[i].lhs)

    equations = []
    equations += [Assignment(symbolicZerothMoment, pdf_sum)]
    equations += [Assignment(u_i_sym, u_i) for u_i_sym, u_i in zip(symbolicFirstMoments, u)]

    return AssignmentCollection(equations, subexpressions)


def divideFirstOrderMomentsByRho(assignment_collection, dim):
    """
    Assumes that the equations of the passed equation collection are the following
        - rho = f_0  + f_1 + ...
        - u_0 = ...
        - u_1 = ...
    Returns a new equation collection where the u terms (first order moments) are divided by rho.
    The dim parameter specifies the number of first order moments. All subsequent equations are just copied over.
    """
    oldEqs = assignment_collection.mainAssignments
    rho = oldEqs[0].lhs
    new_first_order_moment_eq = [Assignment(eq.lhs, eq.rhs / rho) for eq in oldEqs[1:dim+1]]
    new_eqs = [oldEqs[0]] + new_first_order_moment_eq + oldEqs[dim+1:]
    return assignment_collection.copy(new_eqs)


def addDensityOffset(assignment_collection, offset=sp.Rational(1, 1)):
    """
    Assumes that first equation is the density (zeroth moment). Changes the density equations by adding offset to it.
    """
    oldEqs = assignment_collection.mainAssignments
    newDensity = Assignment(oldEqs[0].lhs, oldEqs[0].rhs + offset)
    return assignment_collection.copy([newDensity] + oldEqs[1:])


def applyForceModelShift(shiftMemberName, dim, assignment_collection, forceModel, compressible, reverse=False):
    """
    Modifies the first order moment equations in assignment collection according to the force model shift.
    It is applied if force model has a method named shiftMemberName. The equations 1: dim+1 of the passed
    equation collection are assumed to be the velocity equations.
    """
    if forceModel is not None and hasattr(forceModel, shiftMemberName):
        old_eqs = assignment_collection.mainAssignments
        density = old_eqs[0].lhs if compressible else sp.Rational(1, 1)
        old_vel_eqs = old_eqs[1:dim + 1]
        shift_func = getattr(forceModel, shiftMemberName)
        vel_offsets = shift_func(density)
        if reverse:
            vel_offsets = [-v for v in vel_offsets]
        shifted_velocity_eqs = [Assignment(oldEq.lhs, oldEq.rhs + offset)
                                for oldEq, offset in zip(old_vel_eqs, vel_offsets)]
        new_eqs = [old_eqs[0]] + shifted_velocity_eqs + old_eqs[dim + 1:]
        return assignment_collection.copy(new_eqs)
    else:
        return assignment_collection
