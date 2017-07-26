import sympy as sp

from lbmpy.simplificationfactory import createSimplificationStrategy
from pystencils.sympyextensions import getSymmetricPart
from pystencils import Field
from lbmpy.boundaries.boundary_kernel import offsetFromDir, weightOfDirection, invDir
from pystencils.types import createTypeFromString


class Boundary(object):
    """Base class all boundaries should derive from"""

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
        return []

    @property
    def name(self):
        return type(self).__name__

    def additionalDataInit(self, idxArray):
        return []


class NoSlip(Boundary):
    """No-Slip, (half-way) simple bounce back boundary condition, enforcing zero velocity at obstacle"""
    def __call__(self, pdfField, directionSymbol, lbMethod, **kwargs):
        neighbor = offsetFromDir(directionSymbol, lbMethod.dim)
        inverseDir = invDir(directionSymbol)
        return [sp.Eq(pdfField[neighbor](inverseDir), pdfField(directionSymbol))]

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal
        return hash("NoSlip")

    def __eq__(self, other):
        return type(other) == NoSlip


class NoSlipFullWay(Boundary):
    """Full-way bounce back"""

    @property
    def additionalData(self):
        return [('lastValue', createTypeFromString("double"))]

    def additionalDataInit(self, pdfField, directionSymbol, indexField, **kwargs):
        return [sp.Eq(indexField('lastValue'), pdfField(directionSymbol))]

    def __call__(self, pdfField, directionSymbol, lbMethod, indexField, **kwargs):
        neighbor = offsetFromDir(directionSymbol, lbMethod.dim)
        inverseDir = invDir(directionSymbol)
        return [sp.Eq(pdfField[neighbor](inverseDir), indexField('lastValue')),
                sp.Eq(indexField('lastValue'), pdfField(directionSymbol))]

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal
        return hash("NoSlipFullWay")

    def __eq__(self, other):
        return type(other) == NoSlipFullWay


class UBB(Boundary):

    """Velocity bounce back boundary condition, enforcing specified velocity at obstacle"""

    def __init__(self, velocity, adaptVelocityToForce=False):
        self._velocity = velocity
        self._adaptVelocityToForce = adaptVelocityToForce

    def __call__(self, pdfField, directionSymbol, lbMethod, **kwargs):
        vel = self._velocity
        direction = directionSymbol

        assert len(vel) == lbMethod.dim, \
            "Dimension of velocity (%d) does not match dimension of LB method (%d)" % (len(vel), lbMethod.dim)
        neighbor = offsetFromDir(direction, lbMethod.dim)
        inverseDir = invDir(direction)

        velocity = tuple(v_i.getShifted(*neighbor) if isinstance(v_i, Field.Access) else v_i for v_i in vel)

        if self._adaptVelocityToForce:
            cqc = lbMethod.conservedQuantityComputation
            shiftedVelEqs = cqc.equilibriumInputEquationsFromInitValues(velocity=velocity)
            velocity = [eq.rhs for eq in shiftedVelEqs.extract(cqc.firstOrderMomentSymbols).mainEquations]

        c_s_sq = sp.Rational(1, 3)
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

    def __init__(self, density):
        self._density = density

    def __call__(self, pdfField, directionSymbol, lbMethod, **kwargs):
        """Boundary condition that fixes the density/pressure at the obstacle"""

        def removeAsymmetricPartOfMainEquations(eqColl, dofs):
            newMainEquations = [sp.Eq(e.lhs, getSymmetricPart(e.rhs, dofs)) for e in eqColl.mainEquations]
            return eqColl.copy(newMainEquations)

        neighbor = offsetFromDir(directionSymbol, lbMethod.dim)
        inverseDir = invDir(directionSymbol)

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
        return subExprs + [sp.Eq(pdfField[neighbor](inverseDir), 2 * eq_component - pdfField(directionSymbol))]


class NeumannByCopy(Boundary):
    def __call__(self, pdfField, directionSymbol, lbMethod, **kwargs):
        neighbor = offsetFromDir(directionSymbol, lbMethod.dim)
        return [sp.Eq(pdfField[neighbor](directionSymbol), pdfField(directionSymbol))]

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal
        return hash("NeumannByCopy")

    def __eq__(self, other):
        return type(other) == NeumannByCopy
