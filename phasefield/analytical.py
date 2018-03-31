import sympy as sp
from collections import defaultdict

from pystencils.sympyextensions import multidimensional_sum as multiSum, normalize_product, prod
from pystencils.derivative import functionalDerivative, expandUsingLinearity, Diff, fullDiffExpand

orderParameterSymbolName = "phi"
surfaceTensionSymbolName = "tau"
interfaceWidthSymbol = sp.Symbol("alpha")


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

    bulkFreeEnergy = sum(k * C_i ** 2 * (1 - C_i) ** 2 / 2 for k, C_i in zip(kappa, C))
    surfaceFreeEnergy = sum(k * Diff(C_i) ** 2 / 2 for k, C_i in zip(kappaPrime, C))

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

    transformationMatrix = sp.Matrix([[1,  1, 1],
                                      [1, -1, 0],
                                      [0,  0, 1]])
    rhoDef, phiDef, psiDef = transformationMatrix * sp.Matrix(C)
    orderParamToConcentrationRelation = sp.solve([rhoDef - rho, phiDef - phi, psiDef - psi], C)

    F = F.subs(orderParamToConcentrationRelation)
    if expandDerivatives:
        F = expandUsingLinearity(F, functions=orderParameters)

    return F, transformationMatrix


def freeEnergyFunctionalNPhasesPenaltyTerm(orderParameters, interfaceWidth=interfaceWidthSymbol, kappa=None,
                                           penaltyTermFactor=0.01):
    numPhases = len(orderParameters)
    if kappa is None:
        kappa = sp.symbols("kappa_:%d" % (numPhases,))
    if not hasattr(kappa, "__len__"):
        kappa = [kappa] * numPhases

    def f(x):
        return x ** 2 * (1 - x) ** 2

    bulk = sum(f(c) * k / 2 for c, k in zip(orderParameters, kappa))
    interface = sum(Diff(c) ** 2 / 2 * interfaceWidth ** 2 * k
                    for c, k in zip(orderParameters, kappa))

    bulkPenaltyTerm = (1 - sum(c for c in orderParameters)) ** 2
    return bulk + interface + penaltyTermFactor * bulkPenaltyTerm


def freeEnergyFunctionalNPhases(numPhases=None, surfaceTensions=symmetricSymbolicSurfaceTension,
                                interfaceWidth=interfaceWidthSymbol, orderParameters=None,
                                includeBulk=True, includeInterface=True, symbolicLambda=False,
                                symbolicDependentVariable=False):
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
    :param symbolicLambda: surface energy coefficient is represented by symbol, not in expanded form
    :param symbolicDependentVariable: last phase variable is defined as 1-otherPhaseVars, if this is set to True
                                      it is represented by phi_A for better readability
    """
    assert not (numPhases is None and orderParameters is None)
    if orderParameters is None:
        phi = symbolicOrderParameters(numPhases-1)
    else:
        phi = orderParameters
        numPhases = len(phi) + 1

    if not symbolicDependentVariable:
        phi = tuple(phi) + (1 - sum(phi),)
    else:
        phi = tuple(phi) + (sp.Symbol("phi_D"), )

    # Compared to handwritten notes we scale the interface width parameter here to obtain the correct
    # equations for the interface profile and the surface tensions i.e. to pass tests
    # test_analyticInterfaceSolution and test_surfaceTensionDerivation
    interfaceWidth *= sp.sqrt(2)

    def f(c):
        return c ** 2 * (1 - c) ** 2

    def lambdaCoeff(k, l):
        if symbolicLambda:
            return sp.Symbol("Lambda_%d%d" % ((k, l) if k < l else (l, k)))
        N = numPhases - 1
        if k == l:
            assert surfaceTensions(l, l) == 0
        return 3 / sp.sqrt(2) * interfaceWidth * (surfaceTensions(k, N) + surfaceTensions(l, N) - surfaceTensions(k, l))

    def bulkTerm(i, j):
        return surfaceTensions(i, j) / 2 * (f(phi[i]) + f(phi[j]) - f(phi[i] + phi[j]))

    F_bulk = 3 / sp.sqrt(2) / interfaceWidth * sum(bulkTerm(i, j) for i, j in multiSum(2, numPhases) if i != j)
    F_interface = sum(lambdaCoeff(i, j) / 2 * Diff(phi[i]) * Diff(phi[j]) for i, j in multiSum(2, numPhases-1))

    result = 0
    if includeBulk:
        result += F_bulk
    if includeInterface:
        result += F_interface
    return result


def separateIntoBulkAndInterface(freeEnergy):
    """Separates the bulk and interface parts of a free energy

    >>> F = freeEnergyFunctionalNPhases(3)
    >>> bulk, inter = separateIntoBulkAndInterface(F)
    >>> assert sp.expand(bulk - freeEnergyFunctionalNPhases(3, includeInterface=False)) == 0
    >>> assert sp.expand(inter - freeEnergyFunctionalNPhases(3, includeBulk=False)) == 0
    """
    freeEnergy = freeEnergy.expand()
    bulkPart = freeEnergy.subs({a: 0 for a in freeEnergy.atoms(Diff)})
    interfacePart = freeEnergy - bulkPart
    return bulkPart, interfacePart


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
    return sp.Matrix([expandUsingLinearity(functionalDerivative(freeEnergy, op),constants=constants)
                      for op in orderParameters])


def forceFromPhiAndMu(orderParameters, dim, mu=None):
    if mu is None:
        mu = sp.symbols("mu_:%d" % (len(orderParameters),))

    return sp.Matrix([sum(- c_i * Diff(mu_i, a) for c_i, mu_i in zip(orderParameters, mu))
                      for a in range(dim)])


def substituteLaplacianBySum(eq, dim):
    """Substitutes abstract laplacian represented by ∂∂ by a sum over all dimensions
    i.e. in case of 3D: ∂∂ is replaced by ∂0∂0 + ∂1∂1 + ∂2∂2
    :param eq: the term where the substitutions should be made
    :param dim: spatial dimension, in example above, 3
    """
    functions = [d.args[0] for d in eq.atoms(Diff)]
    substitutions = {Diff(Diff(op)): sum(Diff(Diff(op, i), i) for i in range(dim))
                     for op in functions}
    return fullDiffExpand(eq.subs(substitutions))


def coshIntegral(f, var):
    """Integrates a function f that has exactly one cosh term, from -oo to oo, by
    substituting a new helper variable for the cosh argument"""
    coshTerm = list(f.atoms(sp.cosh))
    assert len(coshTerm) == 1
    integral = sp.Integral(f, var)
    transformedInt = integral.transform(coshTerm[0].args[0], sp.Symbol("u", real=True))
    return sp.integrate(transformedInt.args[0], (transformedInt.args[1][0], -sp.oo, sp.oo))


def symmetricTensorLinearization(dim):
    nextIdx = 0
    resultMap = {}
    for idx in multiSum(2, dim):
        idx = tuple(sorted(idx))
        if idx in resultMap:
            continue
        else:
            resultMap[idx] = nextIdx
            nextIdx += 1
    return resultMap

# ----------------------------------------- Pressure Tensor ------------------------------------------------------------


def extractGamma(freeEnergy, orderParameters):
    """Extracts parameters before the gradient terms"""
    result = defaultdict(lambda: 0)
    freeEnergy = freeEnergy.expand()
    assert freeEnergy.func == sp.Add
    for product in freeEnergy.args:
        product = normalize_product(product)
        diffFactors = [e for e in product if e.func == Diff]
        if len(diffFactors) == 0:
            continue

        if len(diffFactors) != 2:
            raise ValueError("Could not new_filtered Λ because of term " + str(product))

        indices = sorted([orderParameters.index(d.args[0]) for d in diffFactors])
        result[tuple(indices)] += prod(e for e in product if e.func != Diff)
        if diffFactors[0] == diffFactors[1]:
            result[tuple(indices)] *= 2
    return result


def pressureTensorBulkComponent(freeEnergy, orderParameters):
    """Diagonal component of pressure tensor in bulk"""
    bulkFreeEnergy, _ = separateIntoBulkAndInterface(freeEnergy)
    muBulk = chemicalPotentialsFromFreeEnergy(bulkFreeEnergy, orderParameters)
    return sum(c_i * mu_i for c_i, mu_i in zip(orderParameters, muBulk)) - bulkFreeEnergy


def pressureTensorInterfaceComponent(freeEnergy, orderParameters, dim, a, b):
    gamma = extractGamma(freeEnergy, orderParameters)
    d = Diff
    result = 0
    for i, c_i in enumerate(orderParameters):
        for j, c_j in enumerate(orderParameters):
            t = d(c_i, a) * d(c_j, b) + d(c_i, b) * d(c_j, a)
            if a == b:
                t -= sum(d(c_i, g) * d(c_j, g) for g in range(dim))
                t -= sum(c_i * d(d(c_j, g), g) for g in range(dim))
                t -= sum(c_j * d(d(c_i, g), g) for g in range(dim))
            gamma_ij = gamma[(i, j)] if i < j else gamma[(j, i)]
            result += t * gamma_ij / 2
    return result


def pressureTensorFromFreeEnergy(freeEnergy, orderParameters, dim):
    def getEntry(i, j):
        pIf = pressureTensorInterfaceComponent(freeEnergy, orderParameters, dim, i, j)
        pB = pressureTensorBulkComponent(freeEnergy, orderParameters) if i == j else 0
        return sp.expand(pIf + pB)

    return sp.Matrix(dim, dim, getEntry)


def forceFromPressureTensor(pressureTensor, functions=None):
    assert len(pressureTensor.shape) == 2 and pressureTensor.shape[0] == pressureTensor.shape[1]
    dim = pressureTensor.shape[0]

    def forceComponent(b):
        r = -sum(Diff(pressureTensor[a, b], a) for a in range(dim))
        r = fullDiffExpand(r, functions=functions)
        return r

    return sp.Matrix([forceComponent(b) for b in range(dim)])