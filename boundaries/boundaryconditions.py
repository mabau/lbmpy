import sympy as sp
from lbmpy.simplificationfactory import createSimplificationStrategy
from pystencils.astnodes import SympyAssignment
from pystencils.sympyextensions import getSymmetricPart
from pystencils import Field
from lbmpy.boundaries.boundaryhandling import BoundaryOffsetInfo, LbmWeightInfo
from pystencils.data_types import createType


class Boundary(object):
    """Base class all boundaries should derive from"""

    def __init__(self, name=None):
        self._name = name

    def __call__(self, pdfField, directionSymbol, lbMethod, indexField):
        """
        This function defines the boundary behavior and must therefore be implemented by all boundaries.
        Here the boundary is defined as a list of sympy equations, from which a boundary kernel is generated.
        :param pdfField: pystencils field describing the pdf. The current cell is cell next to the boundary,
                         which is influenced by the boundary cell i.e. has a link from the boundary cell to
                         itself.
        :param directionSymbol: a sympy symbol that can be used as index to the pdfField. It describes
                                the direction pointing from the fluid to the boundary cell 
        :param lbMethod: an instance of the LB method used. Use this to adapt the boundary to the method 
                         (e.g. compressiblity)
        :param indexField: the boundary index field that can be used to retrieve and update boundary data
        :return: list of sympy equations
        """
        raise NotImplementedError("Boundary class has to overwrite __call__")

    @property
    def additionalData(self):
        """Return a list of (name, type) tuples for additional data items required in this boundary
        These data items can either be initialized in separate kernel see additionalDataKernelInit or by 
        Python callbacks - see additionalDataCallback """
        return []

    @property
    def additionalDataInitCallback(self):
        """Return a callback function called with a boundary data setter object and returning a dict of
        data-name to data for each element that should be initialized"""
        return None

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return type(self).__name__

    @name.setter
    def name(self, newValue):
        self._name = newValue


class NoSlip(Boundary):

    def __init__(self, name=None):
        """Set an optional name here, to mark boundaries, for example for force evaluations"""
        super(NoSlip, self).__init__(name)

    """No-Slip, (half-way) simple bounce back boundary condition, enforcing zero velocity at obstacle"""
    def __call__(self, pdfField, directionSymbol, lbMethod, **kwargs):
        neighbor = BoundaryOffsetInfo.offsetFromDir(directionSymbol, lbMethod.dim)
        inverseDir = BoundaryOffsetInfo.invDir(directionSymbol)
        return [sp.Eq(pdfField[neighbor](inverseDir), pdfField(directionSymbol))]

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal (as long as name is equal)
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, NoSlip):
            return False
        return self.name == other.name


class UBB(Boundary):

    """Velocity bounce back boundary condition, enforcing specified velocity at obstacle"""

    def __init__(self, velocity, adaptVelocityToForce=False, dim=None, name=None):
        """
        
        :param velocity: can either be a constant, an access into a field, or a callback function.
                         The callback functions gets a numpy record array with members, 'x','y','z', 'dir' (direction) 
                         and 'velocity' which has to be set to the desired velocity of the corresponding link
        :param adaptVelocityToForce:
        """
        super(UBB, self).__init__(name)
        self._velocity = velocity
        self._adaptVelocityToForce = adaptVelocityToForce
        if callable(self._velocity) and not dim:
            raise ValueError("When using a velocity callback the dimension has to be specified with the dim parameter")
        elif not callable(self._velocity):
            dim = len(velocity)
        self.dim = dim

    @property
    def additionalData(self):
        if callable(self._velocity):
            return [('vel_%d' % (i,), createType("double")) for i in range(self.dim)]
        else:
            return []

    @property
    def additionalDataInitCallback(self):
        if callable(self._velocity):
            return self._velocity

    def __call__(self, pdfField, directionSymbol, lbMethod, indexField, **kwargs):
        velFromIdxField = callable(self._velocity)
        vel = [indexField('vel_%d' % (i,)) for i in range(self.dim)] if velFromIdxField else self._velocity
        direction = directionSymbol

        assert self.dim == lbMethod.dim, "Dimension of UBB (%d) does not match dimension of method (%d)" \
                                         % (self.dim, lbMethod.dim)

        neighbor = BoundaryOffsetInfo.offsetFromDir(direction, lbMethod.dim)
        inverseDir = BoundaryOffsetInfo.invDir(direction)

        velocity = tuple(v_i.getShifted(*neighbor) if isinstance(v_i, Field.Access) and not velFromIdxField else v_i
                         for v_i in vel)

        if self._adaptVelocityToForce:
            cqc = lbMethod.conservedQuantityComputation
            shiftedVelEqs = cqc.equilibriumInputEquationsFromInitValues(velocity=velocity)
            velocity = [eq.rhs for eq in shiftedVelEqs.extract(cqc.firstOrderMomentSymbols).mainEquations]

        c_s_sq = sp.Rational(1, 3)
        weightOfDirection = LbmWeightInfo.weightOfDirection
        velTerm = 2 / c_s_sq * sum([d_i * v_i for d_i, v_i in zip(neighbor, velocity)]) * weightOfDirection(direction)

        # Better alternative: in conserved value computation
        # rename what is currently called density to "virtualDensity"
        # provide a new quantity density, which is constant in case of incompressible models
        if not lbMethod.conservedQuantityComputation.zeroCenteredPdfs:
            cqc = lbMethod.conservedQuantityComputation
            densitySymbol = sp.Symbol("rho")
            pdfFieldAccesses = [pdfField(i) for i in range(len(lbMethod.stencil))]
            densityEquations = cqc.outputEquationsFromPdfs(pdfFieldAccesses, {'density': densitySymbol})
            densitySymbol = lbMethod.conservedQuantityComputation.definedSymbols()['density']
            result = densityEquations.allEquations
            result += [sp.Eq(pdfField[neighbor](inverseDir),
                             pdfField(direction) - velTerm * densitySymbol)]
            return result
        else:
            return [sp.Eq(pdfField[neighbor](inverseDir),
                          pdfField(direction) - velTerm)]


class FixedDensity(Boundary):

    def __init__(self, density, name=None):
        super(FixedDensity, self).__init__(name)
        self._density = density

    def __call__(self, pdfField, directionSymbol, lbMethod, **kwargs):
        """Boundary condition that fixes the density/pressure at the obstacle"""

        def removeAsymmetricPartOfMainEquations(eqColl, dofs):
            newMainEquations = [sp.Eq(e.lhs, getSymmetricPart(e.rhs, dofs)) for e in eqColl.mainEquations]
            return eqColl.copy(newMainEquations)

        neighbor = BoundaryOffsetInfo.offsetFromDir(directionSymbol, lbMethod.dim)
        inverseDir = BoundaryOffsetInfo.invDir(directionSymbol)

        cqc = lbMethod.conservedQuantityComputation
        velocity = cqc.definedSymbols()['velocity']
        symmetricEq = removeAsymmetricPartOfMainEquations(lbMethod.getEquilibrium(), dofs=velocity)
        substitutions = {sym: pdfField(i) for i, sym in enumerate(lbMethod.preCollisionPdfSymbols)}
        symmetricEq = symmetricEq.copyWithSubstitutionsApplied(substitutions)

        simplification = createSimplificationStrategy(lbMethod)
        symmetricEq = simplification(symmetricEq)

        densitySymbol = cqc.definedSymbols()['density']

        density = self._density
        densityEq = cqc.equilibriumInputEquationsFromInitValues(density=density).insertSubexpressions().mainEquations[0]
        assert densityEq.lhs == densitySymbol
        transformedDensity = densityEq.rhs

        conditions = [(eq_i.rhs, sp.Equality(directionSymbol, i))
                      for i, eq_i in enumerate(symmetricEq.mainEquations)] + [(0, True)]
        eq_component = sp.Piecewise(*conditions)

        subExprs = [sp.Eq(eq.lhs, transformedDensity if eq.lhs == densitySymbol else eq.rhs)
                    for eq in symmetricEq.subexpressions]

        return subExprs + [SympyAssignment(pdfField[neighbor](inverseDir),
                                           2 * eq_component - pdfField(directionSymbol))]


class NeumannByCopy(Boundary):
    def __call__(self, pdfField, directionSymbol, lbMethod, **kwargs):
        neighbor = BoundaryOffsetInfo.offsetFromDir(directionSymbol, lbMethod.dim)
        return [sp.Eq(pdfField[neighbor](directionSymbol), pdfField(directionSymbol))]

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal
        return hash("NeumannByCopy")

    def __eq__(self, other):
        return type(other) == NeumannByCopy

