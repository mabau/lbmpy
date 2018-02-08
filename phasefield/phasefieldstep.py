from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.phasefield.cahn_hilliard_lbm import cahnHilliardLbmMethod
from pystencils import createKernel
from lbmpy.phasefield import createChemicalPotentialEvolutionEquations, createForceUpdateEquations
from lbmpy.phasefield.analytical import chemicalPotentialsFromFreeEnergy, CahnHilliardFDStep
from pystencils.datahandling import SerialDataHandling
from pystencils.equationcollection.simplifications import sympyCseOnEquationList
from pystencils.slicing import makeSlice, SlicedGetter


class PhaseFieldStep:

    def __init__(self, freeEnergy, orderParameters, domainSize=None, dataHandling=None,
                 name='pf', hydroLbmParameters={},
                 hydroDynamicRelaxationRate=1.0, cahnHilliardRelaxationRates=1.0, densityOrderParameter=None,
                 target='cpu', openMP=False, kernelParams={}, dx=1, dt=1, solveCahnHilliardWithFiniteDifferences=False):

        if dataHandling is None:
            dataHandling = SerialDataHandling(domainSize)

        self.freeEnergy = freeEnergy
        self.chemicalPotentials = chemicalPotentialsFromFreeEnergy(freeEnergy, orderParameters)

        # ------------------ Adding arrays ---------------------
        gpu = target == 'gpu'
        self.gpu = gpu
        self.numOrderParameters = len(orderParameters)
        self.phiFieldName = name + "_phi"
        self.muFieldName = name + "_mu"
        self.velFieldName = name + "_u"
        self.forceFieldName = name + "_F"
        self.dataHandling = dataHandling
        self.phiField = dataHandling.addArray(self.phiFieldName, fSize=len(orderParameters), gpu=gpu, latexName='φ')
        self.muField = dataHandling.addArray(self.muFieldName, fSize=len(orderParameters), gpu=gpu, latexName="μ")
        self.velField = dataHandling.addArray(self.velFieldName, fSize=dataHandling.dim, gpu=gpu, latexName="u")
        self.forceField = dataHandling.addArray(self.forceFieldName, fSize=dataHandling.dim, gpu=gpu, latexName="F")

        # ------------------ Creating kernels ------------------
        phi = tuple(self.phiField(i) for i in range(len(orderParameters)))
        F = self.freeEnergy.subs({old: new for old, new in zip(orderParameters, phi)})

        # μ Kernel
        self.muEqs = createChemicalPotentialEvolutionEquations(F, phi, self.phiField, self.muField, dx, cse=False)
        self.muKernel = createKernel(sympyCseOnEquationList(self.muEqs), target=target, cpuOpenMP=openMP).compile()
        self.phiSync = dataHandling.synchronizationFunction([self.phiFieldName], target=target)

        # F Kernel
        self.forceEqs = createForceUpdateEquations(self.forceField, self.phiField, self.muField, dx)
        self.forceKernel = createKernel(self.forceEqs, target=target, cpuOpenMP=openMP).compile()
        self.muSync = dataHandling.synchronizationFunction([self.muFieldName], target=target)

        # Hydrodynamic LBM
        if densityOrderParameter is not None:
            hydroLbmParameters['output'] = {'density': self.phiField(orderParameters.index(densityOrderParameter))}
        self.hydroLbmStep = LatticeBoltzmannStep(dataHandling=dataHandling, name=name + '_hydroLBM',
                                                 relaxationRate=hydroDynamicRelaxationRate,
                                                 computeVelocityInEveryStep=True, force=self.forceField,
                                                 velocityDataName=self.velFieldName, kernelParams=kernelParams,
                                                 **hydroLbmParameters)

        # Cahn-Hilliard LBMs
        if not hasattr(cahnHilliardRelaxationRates, '__len__'):
            cahnHilliardRelaxationRates = [cahnHilliardRelaxationRates] * len(orderParameters)

        self.cahnHilliardSteps = []

        if solveCahnHilliardWithFiniteDifferences:
            if densityOrderParameter is not None:
                raise NotImplementedError("densityOrderParameter not supported when CH is solved with finite differences")
            chStep = CahnHilliardFDStep(self.dataHandling, self.phiFieldName, self.muFieldName, self.velFieldName,
                                        target=target, dx=dx, dt=dt, mobilities=1)
            self.cahnHilliardSteps.append(chStep)
        else:
            for i, op in enumerate(orderParameters):
                if op == densityOrderParameter:
                    continue

                chMethod = cahnHilliardLbmMethod(self.hydroLbmStep.method.stencil, self.muField(i),
                                                 relaxationRate=cahnHilliardRelaxationRates[i])
                chStep = LatticeBoltzmannStep(dataHandling=dataHandling, relaxationRate=1, lbMethod=chMethod,
                                              velocityInputArrayName=self.velField.name,
                                              densityDataName=self.phiField.name,
                                              computeDensityInEveryStep=True,
                                              densityDataIndex=i,
                                              name=name + "_chLbm_%d" % (i,), )
                self.cahnHilliardSteps.append(chStep)

        # Init φ and μ
        self.dataHandling.fill(self.phiFieldName, 0.0)
        self.dataHandling.fill(self.phiFieldName, 1.0 if densityOrderParameter is not None else 0.0, fValue=0)
        self.dataHandling.fill(self.muFieldName, 0.0)
        self.dataHandling.fill(self.forceFieldName, 0.0)
        self.setPdfFieldsFromMacroscopicValues()

        self.timeStepsRun = 0

        self.runHydroLbm = True

    def setPdfFieldsFromMacroscopicValues(self):
        self.hydroLbmStep.setPdfFieldsFromMacroscopicValues()
        for chStep in self.cahnHilliardSteps:
            chStep.setPdfFieldsFromMacroscopicValues()

    def preRun(self):
        if self.gpu:
            self.dataHandling.toGpu(self.muFieldName)
            self.dataHandling.toGpu(self.forceFieldName)
        self.hydroLbmStep.preRun()
        for chStep in self.cahnHilliardSteps:
            chStep.preRun()

    def postRun(self):
        if self.gpu:
            self.dataHandling.toCpu(self.muFieldName)
            self.dataHandling.toCpu(self.forceFieldName)
        if self.runHydroLbm:
            self.hydroLbmStep.postRun()
        for chStep in self.cahnHilliardSteps:
            chStep.postRun()

    def timeStep(self):
        self.phiSync()
        self.dataHandling.runKernel(self.muKernel)

        self.muSync()
        self.dataHandling.runKernel(self.forceKernel)

        if self.runHydroLbm:
            self.hydroLbmStep.timeStep()

        for chLbm in self.cahnHilliardSteps:
            chLbm.timeStep()

        self.timeStepsRun += 1

    def run(self, timeSteps):
        self.preRun()
        for i in range(timeSteps):
            self.timeStep()
        self.postRun()

    def _getSlice(self, dataName, sliceObj):
        if sliceObj is None:
            sliceObj = makeSlice[:, :] if self.dim == 2 else makeSlice[:, :, 0.5]
        return self.dataHandling.gatherArray(dataName, sliceObj).squeeze()

    def phiSlice(self, sliceObj=None):
        return self._getSlice(self.phiFieldName, sliceObj)

    def muSlice(self, sliceObj=None):
        return self._getSlice(self.muFieldName, sliceObj)

    def velocitySlice(self, sliceObj=None):
        return self._getSlice(self.velFieldName, sliceObj)

    def forceSlice(self, sliceObj=None):
        return self._getSlice(self.forceFieldName, sliceObj)

    @property
    def phi(self):
        return SlicedGetter(self.phiSlice)

    @property
    def mu(self):
        return SlicedGetter(self.muSlice)

    @property
    def velocity(self):
        return SlicedGetter(self.velocitySlice)

    @property
    def force(self):
        return SlicedGetter(self.forceSlice)