import numpy as np
import numbers
import waLBerla as wlb
from lbmpy.creationfunctions import createLatticeBoltzmannFunction, updateWithDefaultParameters, \
    switchToSymbolicRelaxationRatesForEntropicMethods
from lbmpy.macroscopic_value_kernels import compileMacroscopicValuesGetter, compileMacroscopicValuesSetter
from lbmpy.parallel.boundaryhandling import BoundaryHandling
from lbmpy.parallel.blockiteration import slicedBlockIteration
from lbmpy.boundaries import NoSlip, UBB
from pystencils.slicing import sliceFromDirection


def createLidDrivenCavity(blocks, lidVelocity=0.005, lbmKernel=None, **kwargs):
    """
    Creates a lid driven cavity scenario

    :param lidVelocity: x velocity of lid in lattice coordinates.
    :param lbmKernel: a LBM function, which would otherwise automatically created
    :param kwargs: other parameters are passed on to the method, see :mod:`lbmpy.creationfunctions`
    :return: instance of :class:`Scenario`
    """
    scenario = Scenario(blocks, lbmKernel=lbmKernel, **kwargs)

    myUbb = UBB(velocity=[lidVelocity, 0, 0][:scenario.method.dim])
    dim = scenario.method.dim
    scenario.boundaryHandling.setBoundary(myUbb, sliceFromDirection('N', dim))
    for direction in ('W', 'E', 'S') if scenario.method.dim == 2 else ('W', 'E', 'S', 'T', 'B'):
        scenario.boundaryHandling.setBoundary(NoSlip(), sliceFromDirection(direction, dim))

    return scenario


class Scenario(object):
    vtkCounter = 0

    def __init__(self, blocks, optimizationParams={}, lbmKernel=None,
                 preUpdateFunctions=[], kernelParams={}, blockDataPrefix='', directCommunication=False, **kwargs):

        methodParameters = kwargs
        self.blocks = blocks
        self._blockDataPrefix = blockDataPrefix
        self.kernelParams = kernelParams

        domainShape = [i + 1 for i in blocks.getDomainCellBB().max]
        self.dim = 2 if domainShape[2] == 1 else 3
        if 'stencil' not in methodParameters:
            methodParameters['stencil'] = 'D2Q9' if self.dim == 2 else 'D3Q27'

        methodParameters, optimizationParams = updateWithDefaultParameters(methodParameters, optimizationParams)
        self._target = optimizationParams['target']
        if optimizationParams['fieldLayout'] in ('f', 'reverseNumpy'):
            optimizationParams['fieldLayout'] = 'fzyx'
        if optimizationParams['fieldLayout'] not in ('fzyx', 'zyxf'):
            raise ValueError("In parallel scenarios only layouts 'fxyz'  and 'zyxf' are supported")

        wlbLayoutMap = {
            'fzyx': wlb.field.Layout.fzyx,
            'zyxf': wlb.field.Layout.zyxf,
        }
        wlbLayout = wlbLayoutMap[optimizationParams['fieldLayout']]
        self._fieldLayout = optimizationParams['fieldLayout']

        # ----- Create LBM kernel
        if lbmKernel is None:
            switchToSymbolicRelaxationRatesForEntropicMethods(methodParameters, kernelParams)
            methodParameters['optimizationParams'] = optimizationParams
            self._lbmKernel = createLatticeBoltzmannFunction(**methodParameters)
        else:
            self._lbmKernel = lbmKernel

        # ----- Add fields
        numPdfs = len(self._lbmKernel.method.stencil)
        wlb.field.addToStorage(blocks, self.pdfFieldId, float, fSize=numPdfs, ghostLayers=1, layout=wlbLayout)
        wlb.field.addFlagFieldToStorage(blocks, self.flagFieldId, nrOfBits=16, ghostLayers=1)

        if self._target == 'gpu':
            wlb.cuda.addGpuFieldToStorage(blocks, blockDataPrefix + "gpuPdfs", float, fSize=numPdfs,
                                          usePitchedMem=False)

        # ----- Create communication scheme
        communicationStencil = methodParameters['stencil']

        communicationFunctions = {
            ('cpu', True): (wlb.createUniformDirectScheme, wlb.field.createMPIDatatypeInfo),
            ('cpu', False): (wlb.createUniformBufferedScheme, wlb.field.createPackInfo),
            #('gpu', True): (wlb.createUniformDirectScheme, wlb.cuda.createMPIDatatypeInfo),
            #('gpu', False): (wlb.createUniformBufferedScheme, wlb.cuda.createPackInfo),
        }
        createScheme, createCommunicationData = communicationFunctions[(self._target, directCommunication)]
        self._communicationScheme = createScheme(blocks, communicationStencil)
        self._communicationScheme.addDataToCommunicate(createCommunicationData(blocks, self.pdfFieldId))

        # ----- Create boundary handling

        self.boundaryHandling = BoundaryHandling(blocks, self._lbmKernel.method, self.pdfFieldId, self.flagFieldId,
                                                 target=self._target)

        # ----- Macroscopic value input/output

        wlb.field.addToStorage(blocks, self.densityFieldId, float, fSize=1, ghostLayers=1, layout=wlbLayout)
        wlb.field.addToStorage(blocks, self.velocityFieldId, float, fSize=3, ghostLayers=1, layout=wlbLayout)

        self._getMacroscopicKernel = compileMacroscopicValuesGetter(self._lbmKernel.method, ['density', 'velocity'],
                                                                    fieldLayout=optimizationParams['fieldLayout'],
                                                                    target='cpu')

        self._macroscopicValueSetter = None
        self._macroscopicValueGetter = None

        self._vtkOutput = wlb.vtk.makeOutput(blocks, "vtk_%02d" % (Scenario.vtkCounter,))
        Scenario.vtkCounter += 1
        self._vtkOutput.addCellDataWriter(wlb.field.createVTKWriter(self.blocks, self.flagFieldId))
        self._vtkOutput.addCellDataWriter(wlb.field.createVTKWriter(self.blocks, self.densityFieldId))
        self._vtkOutput.addCellDataWriter(wlb.field.createVTKWriter(self.blocks, self.velocityFieldId))

        self.timeStepsRun = 0
        self._dstFieldCache = dict()

    def run(self, timeSteps):
        if self._target == 'cpu':
            for t in range(timeSteps):
                self._communicationScheme()
                self.boundaryHandling()
                for block in self.blocks:
                    srcField = block[self.pdfFieldId]
                    pdfArr = wlb.field.toArray(srcField, withGhostLayers=True)
                    swapIdx = (srcField.size, srcField.layout)
                    if swapIdx not in self._dstFieldCache:
                        self._dstFieldCache[swapIdx] = srcField.cloneUninitialized()
                    dstField = self._dstFieldCache[swapIdx]
                    dstArr = wlb.field.toArray(dstField, withGhostLayers=True)
                    self._lbmKernel(src=pdfArr, dst=dstArr, **self.kernelParams)
                    srcField.swapDataPointers(dstField)

        self._getMacroscopicValues()
        self.timeStepsRun += timeSteps

    def writeVTK(self):
        self._vtkOutput.write(self.timeStepsRun)

    def setMacroscopicValue(self, density=None, velocity=None, indexExpr=None):

        if density is None and velocity is None:
            raise ValueError("Please specify either density or velocity")

        # Check velocity
        velocityCallback = None
        velocityValue = [0, 0, 0]
        if isinstance(velocity, numbers.Number):
            velocityValue = [velocity] * 3
        elif hasattr(velocity, "__len__"):
            assert len(velocity) == 3
            velocityValue = velocity
        elif hasattr(velocity, "__call__"):
            velocityCallback = velocity
        elif velocity is None:
            pass
        else:
            raise ValueError("velocity has be a number, sequence of length 3 or a callback function")

        # Check density
        densityCallback = None
        densityValue = 1.0
        if isinstance(density, numbers.Number):
            densityValue = density
        elif hasattr(density, "__call__"):
            densityCallback = density
        elif density is None:
            pass
        else:
            raise ValueError("density has to be a number or a callback function")

        for block, (x, y, z), localSlice in slicedBlockIteration(self.blocks, indexExpr, 1, 1):
            if density:
                densityArr = wlb.field.toArray(block[self.densityFieldId], withGhostLayers=True)
                densityArr[localSlice] = densityCallback(x, y, z) if densityCallback else densityValue
            if velocity:
                velArr = wlb.field.toArray(block[self.velocityFieldId], withGhostLayers=True)
                velArr[localSlice, :] = velocityCallback(x, y, z) if velocityCallback else velocityValue

        self._setMacroscpicValues()

    @property
    def pdfFieldId(self):
        return self._blockDataPrefix + "pdfs"

    @property
    def flagFieldId(self):
        return self._blockDataPrefix + "flags"

    @property
    def densityFieldId(self):
        return self._blockDataPrefix + "density"

    @property
    def velocityFieldId(self):
        return self._blockDataPrefix + "velocity"

    @property
    def method(self):
        """Lattice boltzmann method description"""
        return self._lbmKernel.method

    def gatherVelocity(self, sliceDef, targetRank=0):
        velField = wlb.field.gather(self.blocks, self.velocityFieldId, sliceDef, targetRank)
        if not velField:
            return
        velArr = wlb.field.toArray(velField)
        mask = np.logical_not(self.boundaryHandling.getMask(sliceDef, 'fluid', targetRank))
        mask = np.repeat(mask[..., np.newaxis], self.dim, axis=2)
        return np.ma.masked_array(velArr, mask)

    def gatherDensity(self, sliceDef, targetRank=0):
        densityField = wlb.field.gather(self.blocks, self.densityFieldId, sliceDef, targetRank)
        if not densityField:
            return
        densityArr = wlb.field.toArray(densityField)
        mask = np.logical_not(self.boundaryHandling.getMask(sliceDef, 'fluid', targetRank))
        return np.ma.masked_array(densityArr, mask)

    def gatherFlags(self, sliceDef, targetRank=0):
        flagField = wlb.field.gather(self.blocks, self.flagFieldId, sliceDef, targetRank)
        if not flagField:
            return
        return wlb.field.toArray(flagField)

    def _getMacroscopicValues(self):
        """Takes values from pdf field and writes them to density and velocity field"""
        if len(self.blocks) == 0:
            return

        if self._macroscopicValueGetter is None:
            self._macroscopicValueGetter = compileMacroscopicValuesGetter(self.method, ['density', 'velocity'],
                                                                          target='cpu', fieldLayout=self._fieldLayout)
        for block in self.blocks:
            densityArr = wlb.field.toArray(block[self.densityFieldId], withGhostLayers=True)
            velArr = wlb.field.toArray(block[self.velocityFieldId], withGhostLayers=True)
            pdfArr = wlb.field.toArray(block[self.pdfFieldId], withGhostLayers=True)
            self._macroscopicValueGetter(pdfs=pdfArr, velocity=velArr, density=densityArr)

    def _setMacroscpicValues(self):
        """Takes values from density / velocity field and writes them to pdf field"""
        if len(self.blocks) == 0:
            return

        if self._macroscopicValueSetter is None:
            shape = wlb.field.toArray(self.blocks[0][self.flagFieldId], withGhostLayers=True).shape
            vals = {
                'density': np.ones(shape),
                'velocity': np.zeros(shape + (3,)),
            }
            self._macroscopicValueSetter = compileMacroscopicValuesSetter(self.method, vals, target='cpu',
                                                                          fieldLayout=self._fieldLayout,)

        for block in self.blocks:
            densityArr = wlb.field.toArray(block[self.densityFieldId], withGhostLayers=True)
            velArr = wlb.field.toArray(block[self.velocityFieldId], withGhostLayers=True)
            pdfArr = wlb.field.toArray(block[self.pdfFieldId], withGhostLayers=True)
            self._macroscopicValueSetter(pdfArr=pdfArr, velocity=velArr, density=densityArr)

