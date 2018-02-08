import sympy as sp
from lbmpy.chapman_enskog.derivative import expandUsingLinearity, Diff
from pystencils.equationcollection.simplifications import sympyCseOnEquationList
from pystencils.sympyextensions import multidimensionalSummation as multiSum

orderParameterSymbolName = "phi"
surfaceTensionSymbolName = "tau"
interfaceWidthSymbol = sp.Symbol("alpha")


def functionalDerivative(functional, v, constants=None):
    """
    Computes functional derivative of functional with respect to v using Euler-Lagrange equation

    .. math ::

        \frac{\delta F}{\delta v} =
                \frac{\partial F}{\partial v} - \nabla \cdot \frac{\partial F}{\partial \nabla v}

    - assumes that gradients are represented by Diff() node (from Chapman Enskog module)    
    - Diff(Diff(r)) represents the divergence of r
    - the constants parameter is a list with symbols not affected by the derivative. This is used for simplification
      of the derivative terms.
    """
    functional = expandUsingLinearity(functional, constants=constants)
    diffs = functional.atoms(Diff)

    diffV = Diff(v)
    assert diffV in diffs  # not necessary in general, but for this use case this should be true

    nonDiffPart = functional.subs({d: 0 for d in diffs})

    partialF_partialV = sp.diff(nonDiffPart, v)

    dummy = sp.Dummy()
    partialF_partialGradV = functional.subs(diffV, dummy).diff(dummy).subs(dummy, diffV)

    result = partialF_partialV - Diff(partialF_partialGradV)
    return expandUsingLinearity(result, constants=constants)


def coshIntegral(f, var):
    """Integrates a function f that has exactly one cosh term, from -oo to oo, by
    substituting a new helper variable for the cosh argument"""
    coshTerm = list(f.atoms(sp.cosh))
    assert len(coshTerm) == 1
    integral = sp.Integral(f, var)
    transformedInt = integral.transform(coshTerm[0].args[0], sp.Symbol("u", real=True))
    return sp.integrate(transformedInt.args[0], (transformedInt.args[1][0], -sp.oo, sp.oo))


def discreteLaplace(field, index, dx):
    """Returns second order Laplace stencil"""
    dim = field.spatialDimensions
    count = 0
    result = 0
    for d in range(dim):
        for offset in (-1, 1):
            count += 1
            result += field.neighbor(d, offset)(index)

    result -= count * field.center(index)
    result /= dx ** 2
    return result


def symmetricSymbolicSurfaceTension(i, j):
    """Returns symbolic surface tension. The function is symmetric, i.e. interchanging i and j yields the same result.
    If both phase indices i and j are chosen equal, zero is returned"""
    if i == j:
        return 0
    index = (i, j) if i < j else (j, i)
    return sp.Symbol("%s_%d_%d" % ((surfaceTensionSymbolName, ) + index))


def symbolicOrderParameters(numSymbols):
    return sp.symbols("%s_:%i" % (orderParameterSymbolName, numSymbols))


def freeEnergyFunction3Phases(orderParameters=None, interfaceWidth=interfaceWidthSymbol, transformed=True,
                              includeBulk=True, includeInterface=True, expandDerivatives=True,
                              kappa=sp.symbols("kappa_:3")):
    kappaPrime = tuple(interfaceWidth**2 * k for k in kappa)
    C = sp.symbols("C_:3")

    bulkFreeEnergy = sum(k / 2 * C_i ** 2 * (1 - C_i) ** 2 for k, C_i in zip(kappa, C))
    surfaceFreeEnergy = sum(k / 2 * Diff(C_i) ** 2 for k, C_i in zip(kappaPrime, C))

    F = 0
    if includeBulk:
        F += bulkFreeEnergy
    if includeInterface:
        F += surfaceFreeEnergy

    if not transformed:
        return F

    if orderParameters:
        rho, phi, psi = orderParameters
    else:
        rho, phi, psi = sp.symbols("rho phi psi")
    rhoDef = C[0] + C[1] + C[2]
    phiDef = C[0] - C[1]
    psiDef = C[2]

    concentrationToOrderParamRelation = {rho: rhoDef, phi: phiDef, psi: psiDef}
    orderParamToConcentrationRelation = sp.solve([rhoDef - rho, phiDef - phi, psiDef - psi], C)

    F = F.subs(orderParamToConcentrationRelation)
    if expandDerivatives:
        F = expandUsingLinearity(F, functions=orderParameters)

    return F


def freeEnergyFunctionalNPhases(numPhases=None, surfaceTensions=symmetricSymbolicSurfaceTension,
                                interfaceWidth=interfaceWidthSymbol, orderParameters=None,
                                includeBulk=True, includeInterface=True):
    r"""
    Returns a symbolic expression for the free energy of a system with N phases and
    specified surface tensions. The total free energy is the sum of a bulk and an interface component.

    .. math ::

        F_{bulk} = \int \frac{3}{\sqrt{2} \eta}
            \sum_{\substack{\alpha,\beta=0 \\ \alpha \neq \beta}}^{N-1}
            \frac{\tau(\alpha,\beta)}{2} \left[ f(\phi_\alpha) + f(\phi_\beta)
            - f(\phi_\alpha + \phi_\beta)  \right] \; d\Omega

        F_{interface} = \int \sum_{\alpha,\beta=0}^{N-2} \frac{\Lambda_{\alpha\beta}}{2}
                        \left( \nabla \phi_\alpha \cdot \nabla \phi_\beta \right)\; d\Omega

        \Lambda_{\alpha \beta} = \frac{3 \eta}{\sqrt{2}}  \left[ \tau(\alpha,N-1) + \tau(\beta,N-1) -
                                 \tau(\alpha,\beta)  \right]

        f(c) = c^2( 1-c)^2

    :param numPhases: number of phases, called N above
    :param surfaceTensions: surface tension function, called with two phase indices (two integers)
    :param interfaceWidth: called :math:`\eta` above, controls the interface width
    :param orderParameters: explicitly

    Parameter useful for viewing / debugging the function
    :param includeBulk: if false no bulk term is added
    :param includeInterface:if false no interface contribution is added
    """
    assert not (numPhases is None and orderParameters is None)
    if orderParameters is None:
        phi = symbolicOrderParameters(numPhases-1)
    else:
        phi = orderParameters
        numPhases = len(phi) + 1

    phi = tuple(phi) + (1 - sum(phi),)

    # Compared to handwritten notes we scale the interface width parameter here to obtain the correct
    # equations for the interface profile and the surface tensions i.e. to pass tests
    # test_analyticInterfaceSolution and test_surfaceTensionDerivation
    interfaceWidth *= sp.sqrt(2)

    def f(c):
        return c ** 2 * (1 - c) ** 2

    def lambdaCoeff(k, l):
        N = numPhases - 1
        if k == l:
            assert surfaceTensions(l, l) == 0
        return 3 / sp.sqrt(2) * interfaceWidth * (surfaceTensions(k, N) + surfaceTensions(l, N) - surfaceTensions(k, l))

    def bulkTerm(i, j):
        return surfaceTensions(i, j) / 2 * (f(phi[i]) + f(phi[j]) - f(phi[i] + phi[j]))

    F_bulk = 3 / sp.sqrt(2) / interfaceWidth * sum(bulkTerm(i, j) for i, j in multiSum(2, numPhases) if i != j)
    F_interface = sum(lambdaCoeff(i, j) / 2 * Diff(phi[i]) * Diff(phi[j]) for i, j in multiSum(2, numPhases))

    result = 0
    if includeBulk:
        result += F_bulk
    if includeInterface:
        result += F_interface
    return result


def analyticInterfaceProfile(x, interfaceWidth=interfaceWidthSymbol):
    """Analytic expression for a 1D interface normal to x with given interface width

    The following doctest shows that the returned analytical solution is indeed a solution of the ODE that we
    get from the condition :math:`\mu_0 = 0` (thermodynamic equilibrium) for a situation with only a single order
    parameter, i.e. at a transition between two phases.
    >>> numPhases = 4
    >>> x, phi = sp.Symbol("x"), symbolicOrderParameters(numPhases-1)
    >>> F = freeEnergyFunctionalNPhases(orderParameters=phi)
    >>> mu = chemicalPotentialsFromFreeEnergy(F)
    >>> mu0 = mu[0].subs({p: 0 for p in phi[1:]})  # mu[0] as function of one order parameter only
    >>> solution = analyticInterfaceProfile(x)
    >>> solutionSubstitution = {phi[0]: solution, Diff(Diff(phi[0])): sp.diff(solution, x, x) }
    >>> sp.expand(mu0.subs(solutionSubstitution))  # inserting solution should solve the mu_0=0 equation
    0
    """
    return (1 + sp.tanh(x / (2 * interfaceWidth))) / 2


def chemicalPotentialsFromFreeEnergy(freeEnergy, orderParameters=None):
    """Computes chemical potentials as functional derivative of free energy"""
    syms = freeEnergy.atoms(sp.Symbol)
    if orderParameters is None:
        orderParameters = [s for s in syms if s.name.startswith(orderParameterSymbolName)]
        orderParameters.sort(key=lambda e: e.name)
        orderParameters = orderParameters[:-1]
    constants = [s for s in syms if s not in orderParameters]
    return sp.Matrix([functionalDerivative(freeEnergy, op, constants) for op in orderParameters])


def createChemicalPotentialEvolutionEquations(freeEnergy, orderParameters, phiField, muField, dx=1, cse=True):
    """Reads from order parameter (phi) field and updates chemical potentials"""
    chemicalPotential = chemicalPotentialsFromFreeEnergy(freeEnergy, orderParameters)
    laplaceDiscretization = {Diff(Diff(op)): discreteLaplace(phiField, i, dx)
                             for i, op in enumerate(orderParameters)}
    chemicalPotential = chemicalPotential.subs(laplaceDiscretization)
    chemicalPotential = chemicalPotential.subs({op: phiField(i) for i, op in enumerate(orderParameters)})

    muSweepEqs = [sp.Eq(muField(i), cp) for i, cp in enumerate(chemicalPotential)]
    return muSweepEqs if not cse else sympyCseOnEquationList(muSweepEqs)


def createForceUpdateEquations(forceField, phiField, muField, dx=1):
    assert muField.indexDimensions == 1
    muFSize = muField.indexShape[0]
    forceSweepEqs = []
    dim = phiField.spatialDimensions
    for d in range(dim):
        rhs = 0
        for i in range(muFSize):
            rhs -= phiField(i) * (muField.neighbor(d, 1)(i) - muField.neighbor(d, -1)(i)) / (2 * dx)
            # In the C code this form is found: when commenting in make sure phi field is synced before!
            #rhs += muField(i) * (phiField.neighbor(d, 1)(i) - phiField.neighbor(d, -1)(i)) / (2 * dx)
        forceSweepEqs.append(sp.Eq(forceField(d), rhs))
    return forceSweepEqs


def cahnHilliardFdEq(phaseIdx, phi, mu, velocity, mobility, dx, dt):
    from pystencils.finitedifferences import transient, advection, diffusion, Discretization2ndOrder
    cahnHilliard = transient(phi, phaseIdx) + advection(phi, velocity, phaseIdx) - diffusion(mu, mobility, phaseIdx)
    return Discretization2ndOrder(dx, dt)(cahnHilliard)


class CahnHilliardFDStep:
    def __init__(self, dataHandling, phiFieldName, muFieldName, velocityFieldName, name='ch_fd', target='cpu',
                 dx=1, dt=1, mobilities=1):
        from pystencils import createKernel
        self.dataHandling = dataHandling

        muField = self.dataHandling.fields[muFieldName]
        velField = self.dataHandling.fields[velocityFieldName]
        self.phiField = self.dataHandling.fields[phiFieldName]
        self.tmpField = self.dataHandling.addArrayLike(name + '_tmp', phiFieldName, latexName='tmp')

        numPhases = self.dataHandling.fSize(phiFieldName)
        if not hasattr(mobilities, '__len__'):
            mobilities = [mobilities] * numPhases

        updateEqs = []
        for i in range(numPhases):
            rhs = cahnHilliardFdEq(i, self.phiField, muField, velField, mobilities[i], dx, dt)
            updateEqs.append(sp.Eq(self.tmpField(i), rhs))
        self.updateEqs = updateEqs
        self.kernel = createKernel(updateEqs, target=target).compile()
        self.sync = self.dataHandling.synchronizationFunction([phiFieldName, velocityFieldName, muFieldName],
                                                              target=target)

    def timeStep(self):
        self.sync()
        self.dataHandling.runKernel(self.kernel)
        self.dataHandling.swap(self.phiField.name, self.tmpField.name)

    def setPdfFieldsFromMacroscopicValues(self):
        pass

    def preRun(self):
        pass

    def postRun(self):
        pass