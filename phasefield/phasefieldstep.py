import sympy as sp
import numpy as np

from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.phasefield.cahn_hilliard_lbm import cahnHilliardLbmMethod
from lbmpy.phasefield.kerneleqs import muKernel, CahnHilliardFDStep, pressureTensorKernel, \
    forceKernelUsingPressureTensor
from pystencils import createKernel
from lbmpy.phasefield.analytical import chemicalPotentialsFromFreeEnergy, symmetricTensorLinearization
from pystencils.boundaries.boundaryhandling import FlagInterface
from pystencils.boundaries.inkernel import addNeumannBoundary
from pystencils.datahandling import SerialDataHandling
from pystencils.equationcollection.simplifications import sympyCseOnEquationList
from pystencils.slicing import makeSlice, SlicedGetter


class PhaseFieldStep:

    def __init__(self, freeEnergy, orderParameters, domainSize=None, dataHandling=None,
                 name='pf', hydroLbmParameters={},
                 hydroDynamicRelaxationRate=1.0, cahnHilliardRelaxationRates=1.0, densityOrderParameter=None,
                 optimizationParams=None, kernelParams={}, dx=1, dt=1, solveCahnHilliardWithFiniteDifferences=False,
                 orderParameterForce=None, concentrationToOrderParameters=None, orderParametersToConcentrations=None,
                 activateHomogenousNeumannBoundaries=False):

        if optimizationParams is None:
            optimizationParams = {'openMP': False, 'target': 'cpu'}
        openMP, target = optimizationParams['openMP'], optimizationParams['target']

        if dataHandling is None:
            dataHandling = SerialDataHandling(domainSize, periodicity=True)

        self.freeEnergy = freeEnergy
        self.concentrationToOrderParameter = concentrationToOrderParameters
        self.orderParametersToConcentrations = orderParametersToConcentrations

        self.chemicalPotentials = chemicalPotentialsFromFreeEnergy(freeEnergy, orderParameters)

        # ------------------ Adding arrays ---------------------
        gpu = target == 'gpu'
        self.gpu = gpu
        self.numOrderParameters = len(orderParameters)
        pressureTensorSize = len(symmetricTensorLinearization(dataHandling.dim))

        self.phiFieldName = name + "_phi"
        self.muFieldName = name + "_mu"
        self.velFieldName = name + "_u"
        self.forceFieldName = name + "_F"
        self.pressureTensorFieldName = name + "_P"
        self.dataHandling = dataHandling
        self.phiField = dataHandling.addArray(self.phiFieldName, fSize=len(orderParameters), gpu=gpu, latexName='φ')
        self.muField = dataHandling.addArray(self.muFieldName, fSize=len(orderParameters), gpu=gpu, latexName="μ")
        self.velField = dataHandling.addArray(self.velFieldName, fSize=dataHandling.dim, gpu=gpu, latexName="u")
        self.forceField = dataHandling.addArray(self.forceFieldName, fSize=dataHandling.dim, gpu=gpu, latexName="F")
        self.pressureTensorField = dataHandling.addArray(self.pressureTensorFieldName,
                                                         fSize=pressureTensorSize, latexName='P')
        self.flagInterface = FlagInterface(dataHandling, 'flags')

        # ------------------ Creating kernels ------------------
        phi = tuple(self.phiField(i) for i in range(len(orderParameters)))
        F = self.freeEnergy.subs({old: new for old, new in zip(orderParameters, phi)})

        if activateHomogenousNeumannBoundaries:
            def applyNeumannBoundaries(eqs):
                fields = [dataHandling.fields[self.phiFieldName],
                          dataHandling.fields[self.pressureTensorFieldName],
                          ]
                flagField = dataHandling.fields[self.flagInterface.flagFieldName]
                return addNeumannBoundary(eqs, fields, flagField, "neumannFlag", inverseFlag=False)
        else:
            def applyNeumannBoundaries(eqs):
                return eqs

        # μ and pressure tensor update
        self.phiSync = dataHandling.synchronizationFunction([self.phiFieldName], target=target)
        self.muEqs = muKernel(F, phi, self.phiField, self.muField, dx)
        self.pressureTensorEqs = pressureTensorKernel(self.freeEnergy, orderParameters,
                                                      self.phiField, self.pressureTensorField, dx)
        muAndPressureTensorEqs = self.muEqs + self.pressureTensorEqs
        muAndPressureTensorEqs = applyNeumannBoundaries(muAndPressureTensorEqs)
        self.muAndPressureTensorKernel = createKernel(sympyCseOnEquationList(muAndPressureTensorEqs),
                                                      target=target, cpuOpenMP=openMP).compile()

        # F Kernel
        extraForce = sp.Matrix([0] * self.dataHandling.dim)
        if orderParameterForce is not None:
            for orderParameterIdx, force in orderParameterForce.items():
                extraForce += self.phiField(orderParameterIdx) * sp.Matrix(force)
        self.forceEqs = forceKernelUsingPressureTensor(self.forceField, self.pressureTensorField, dx=dx,
                                                       extraForce=extraForce)
        self.forceFromPressureTensorKernel = createKernel(applyNeumannBoundaries(self.forceEqs),
                                                          target=target, cpuOpenMP=openMP).compile()
        self.pressureTensorSync = dataHandling.synchronizationFunction([self.pressureTensorFieldName], target=target)

        # Hydrodynamic LBM
        if densityOrderParameter is not None:
            densityIdx = orderParameters.index(densityOrderParameter)
            hydroLbmParameters['computeDensityInEveryStep'] = True
            hydroLbmParameters['densityDataName'] = self.phiFieldName
            hydroLbmParameters['densityDataIndex'] = densityIdx

        if 'optimizationParams' not in hydroLbmParameters:
            hydroLbmParameters['optimizationParams'] = optimizationParams
        else:
            hydroLbmParameters['optimizationParams'].update(optimizationParams)

        self.hydroLbmStep = LatticeBoltzmannStep(dataHandling=dataHandling, name=name + '_hydroLBM',
                                                 relaxationRate=hydroDynamicRelaxationRate,
                                                 computeVelocityInEveryStep=True, force=self.forceField,
                                                 velocityDataName=self.velFieldName, kernelParams=kernelParams,
                                                 flagInterface=self.flagInterface,
                                                 timeStepOrder='collideStream',
                                                 **hydroLbmParameters)

        # Cahn-Hilliard LBMs
        if not hasattr(cahnHilliardRelaxationRates, '__len__'):
            cahnHilliardRelaxationRates = [cahnHilliardRelaxationRates] * len(orderParameters)

        self.cahnHilliardSteps = []

        if solveCahnHilliardWithFiniteDifferences:
            if densityOrderParameter is not None:
                raise NotImplementedError("densityOrderParameter not supported when "
                                          "CH is solved with finite differences")
            chStep = CahnHilliardFDStep(self.dataHandling, self.phiFieldName, self.muFieldName, self.velFieldName,
                                        target=target, dx=dx, dt=dt, mobilities=1,
                                        equationModifier=applyNeumannBoundaries)
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
                                              stencil='D3Q19' if self.dataHandling.dim == 3 else 'D2Q9',
                                              computeDensityInEveryStep=True,
                                              densityDataIndex=i,
                                              flagInterface=self.hydroLbmStep.boundaryHandling.flagInterface,
                                              name=name + "_chLbm_%d" % (i,),
                                              optimizationParams=optimizationParams)
                self.cahnHilliardSteps.append(chStep)

        self.vtkWriter = self.dataHandling.vtkWriter(name, [self.phiFieldName, self.muFieldName, self.velFieldName,
                                                            self.forceFieldName])

        self.runHydroLbm = True
        self.densityOrderParameter = densityOrderParameter
        self.timeStepsRun = 0
        self.reset()

        self.neumannFlag = 0

    def writeVTK(self):
        self.vtkWriter(self.timeStepsRun)

    def reset(self):
        # Init φ and μ
        self.dataHandling.fill(self.phiFieldName, 0.0)
        self.dataHandling.fill(self.phiFieldName, 1.0 if self.densityOrderParameter is not None else 0.0, fValue=0)
        self.dataHandling.fill(self.muFieldName, 0.0)
        self.dataHandling.fill(self.forceFieldName, 0.0)
        self.dataHandling.fill(self.velFieldName, 0.0)
        self.setPdfFieldsFromMacroscopicValues()

        self.timeStepsRun = 0

    def setPdfFieldsFromMacroscopicValues(self):
        self.hydroLbmStep.setPdfFieldsFromMacroscopicValues()
        for chStep in self.cahnHilliardSteps:
            chStep.setPdfFieldsFromMacroscopicValues()

    def preRun(self):
        if self.gpu:
            self.dataHandling.toGpu(self.phiFieldName)
            self.dataHandling.toGpu(self.muFieldName)
            self.dataHandling.toGpu(self.forceFieldName)
        self.hydroLbmStep.preRun()
        for chStep in self.cahnHilliardSteps:
            chStep.preRun()

    def postRun(self):
        if self.gpu:
            self.dataHandling.toCpu(self.phiFieldName)
            self.dataHandling.toCpu(self.muFieldName)
            self.dataHandling.toCpu(self.forceFieldName)
        if self.runHydroLbm:
            self.hydroLbmStep.postRun()
        for chStep in self.cahnHilliardSteps:
            chStep.postRun()

    def timeStep(self):
        neumannFlag = self.neumannFlag
        #for b in self.dataHandling.iterate(sliceObj=makeSlice[:, 0]):
        #    b[self.phiFieldName][..., 0] = 0.0
        #    b[self.phiFieldName][..., 1] = 1.0
        #for b in self.dataHandling.iterate(sliceObj=makeSlice[0, :]):
        #    b[self.phiFieldName][..., 0] = 1.0
        #    b[self.phiFieldName][..., 1] = 0.0

        self.phiSync()
        self.dataHandling.runKernel(self.muAndPressureTensorKernel, neumannFlag=neumannFlag)
        self.pressureTensorSync()
        self.dataHandling.runKernel(self.forceFromPressureTensorKernel, neumannFlag=neumannFlag)

        if self.runHydroLbm:
            self.hydroLbmStep.timeStep()

        for chLbm in self.cahnHilliardSteps:
            #chLbm.timeStep(neumannFlag=neumannFlag)
            chLbm.timeStep()

        self.timeStepsRun += 1

    @property
    def boundaryHandling(self):
        return self.hydroLbmStep.boundaryHandling

    def setConcentration(self, sliceObj, concentration):
        if self.concentrationToOrderParameter is not None:
            phi = self.concentrationToOrderParameter(concentration)
        else:
            phi = np.array(concentration)

        for b in self.dataHandling.iterate(sliceObj):
            for i in range(phi.shape[-1]):
                b[self.phiFieldName][..., i] = phi[i]

    def setDensity(self, sliceObj, value):
        for b in self.dataHandling.iterate(sliceObj):
            for i in range(self.numOrderParameters):
                b[self.hydroLbmStep.densityDataName].fill(value)

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

    def concentrationSlice(self, sliceObj=None):
        phi = self.phiSlice(sliceObj)
        return phi if self.orderParametersToConcentrations is None else self.orderParametersToConcentrations(phi)

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
    def concentration(self):
        return SlicedGetter(self.concentrationSlice)

    @property
    def mu(self):
        return SlicedGetter(self.muSlice)

    @property
    def velocity(self):
        return SlicedGetter(self.velocitySlice)

    @property
    def force(self):
        return SlicedGetter(self.forceSlice)
