import sympy as sp
from lbmpy.phasefield.analytical import chemicalPotentialsFromFreeEnergy, substituteLaplacianBySum, \
    finiteDifferences2ndOrder, forceFromPhiAndMu, symmetricTensorLinearization, pressureTensorFromFreeEnergy, \
    forceFromPressureTensor


# ---------------------------------- Kernels to compute force ----------------------------------------------------------


def muKernel(freeEnergy, orderParameters, phiField, muField, dx=1):
    """Reads from order parameter (phi) field and updates chemical potentials"""
    assert phiField.spatialDimensions == muField.spatialDimensions
    dim = phiField.spatialDimensions
    chemicalPotential = chemicalPotentialsFromFreeEnergy(freeEnergy, orderParameters)
    chemicalPotential = substituteLaplacianBySum(chemicalPotential, dim)
    chemicalPotential = chemicalPotential.subs({op: phiField(i) for i, op in enumerate(orderParameters)})
    return [sp.Eq(muField(i), finiteDifferences2ndOrder(mu_i, dx)) for i, mu_i in enumerate(chemicalPotential)]


def forceKernelUsingMu(forceField, phiField, muField, dx=1):
    """Computes forces using precomputed chemical potential - needs muKernel first"""
    assert muField.indexDimensions == 1
    force = forceFromPhiAndMu(phiField.vecCenter, mu=muField.vecCenter, dim=muField.spatialDimensions)
    return [sp.Eq(forceField(i),
                  finiteDifferences2ndOrder(f_i, dx)).expand() for i, f_i in enumerate(force)]


def pressureTensorKernel(freeEnergy, orderParameters, phiField, pressureTensorField, dx=1):
    dim = phiField.spatialDimensions
    p = pressureTensorFromFreeEnergy(freeEnergy, orderParameters, dim)
    p = p.subs({op: phiField(i) for i, op in enumerate(orderParameters)})
    indexMap = symmetricTensorLinearization(dim)

    eqs = []
    for index, linIndex in indexMap.items():
        eq = sp.Eq(pressureTensorField(linIndex),
                   finiteDifferences2ndOrder(p[index], dx).expand())
        eqs.append(eq)
    return eqs


def forceKernelUsingPressureTensor(forceField, pressureTensorField, dx=1):
    dim = forceField.spatialDimensions
    indexMap = symmetricTensorLinearization(dim)

    p = sp.Matrix(dim, dim, lambda i, j: pressureTensorField(indexMap[i,j] if i < j else indexMap[j, i]))
    f = forceFromPressureTensor(p)
    return [sp.Eq(forceField(i), finiteDifferences2ndOrder(f_i, dx).expand())
            for i, f_i in enumerate(f)]


# ---------------------------------- Cahn Hilliard with finite differences ---------------------------------------------


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