from functools import partial
import numpy as np
from pystencils import Field
from pystencils.slicing import sliceFromDirection
from lbmpy.creationfunctions import createLatticeBoltzmannFunction
from lbmpy.macroscopic_value_kernels import compileMacroscopicValuesGetter, compileMacroscopicValuesSetter
from lbmpy.boundaries import BoundaryHandling, noSlip, ubb, fixedDensity
from lbmpy.stencils import getStencil


def createScenario(domainSize, boundarySetupFunction, methodParameters, optimizationParams, lbmKernel=None,
                   initialVelocity=None, preUpdateFunctions=[], kernelParams={}):
    if 'target' not in optimizationParams:
        optimizationParams['target'] = 'cpu'

    ghostLayers = 1
    domainSizeWithGhostLayer = tuple([s + 2 * ghostLayers for s in domainSize])
    D = len(domainSize)

    if 'stencil' not in methodParameters:
        methodParameters['stencil'] = 'D2Q9' if D == 2 else 'D3Q27'

    Q = len(getStencil(methodParameters['stencil']))
    pdfArrays = [np.zeros(domainSizeWithGhostLayer + (Q,)),
                 np.zeros(domainSizeWithGhostLayer + (Q,))]

    # Create kernel
    if lbmKernel is None:
        methodParameters['optimizationParams'] = optimizationParams
        lbmKernel = createLatticeBoltzmannFunction(**methodParameters)
    method = lbmKernel.method

    assert D == method.dim, "Domain size and stencil do not match"

    # Boundary setup
    if boundarySetupFunction is not None:
        symPdfField = Field.createFromNumpyArray('pdfs', pdfArrays[0], indexDimensions=1)
        boundaryHandling = BoundaryHandling(symPdfField, domainSize, lbmKernel.method,
                                            target=optimizationParams['target'])
        boundarySetupFunction(boundaryHandling=boundaryHandling, method=method)
        boundaryHandling.prepare()
    else:
        boundaryHandling = None

    # Macroscopic value input/output
    densityArr = [np.zeros(domainSizeWithGhostLayer)]
    velocityArr = [np.zeros(domainSizeWithGhostLayer + (D,))]
    getMacroscopic = compileMacroscopicValuesGetter(method, ['density', 'velocity'], pdfArr=pdfArrays[0], target='cpu')

    if initialVelocity is None:
        initialVelocity = [0] * D
    setMacroscopic = compileMacroscopicValuesSetter(method, {'density': 1.0, 'velocity': initialVelocity},
                                                    pdfArr=pdfArrays[0], target='cpu')
    setMacroscopic(pdfs=pdfArrays[0])

    if optimizationParams['target'] == 'gpu':
        import pycuda.gpuarray as gpuarray
        pdfGpuArrays = [gpuarray.to_gpu(a) for a in pdfArrays]
    else:
        pdfGpuArrays = []

    def cpuTimeLoop(timeSteps):
        for t in range(timeSteps):
            for f in preUpdateFunctions:
                f(pdfArrays[0])
            if boundaryHandling is not None:
                boundaryHandling(pdfs=pdfArrays[0])
            lbmKernel(src=pdfArrays[0], dst=pdfArrays[1], **kernelParams)

            pdfArrays[0], pdfArrays[1] = pdfArrays[1], pdfArrays[0]
        getMacroscopic(pdfs=pdfArrays[0], density=densityArr[0], velocity=velocityArr[0])
        #for vComp in range(velocityArr[0].shape[-1]):
        #    v = velocityArr[0][..., vComp]
        #    v[boundaryHandling.flagField != boundaryHandling._fluidFlag] = 0
        return pdfArrays[0], densityArr[0], velocityArr[0]

    def gpuTimeLoop(timeSteps):
        # Transfer data to gpu
        for cpuArr, gpuArr in zip(pdfArrays, pdfGpuArrays):
            gpuArr.set(cpuArr)

        for t in range(timeSteps):
            for f in preUpdateFunctions:
                f(pdfGpuArrays[0])
            if boundaryHandling is not None:
                boundaryHandling(pdfs=pdfGpuArrays[0])
            lbmKernel(src=pdfGpuArrays[0], dst=pdfGpuArrays[1], **kernelParams)

            pdfGpuArrays[0], pdfGpuArrays[1] = pdfGpuArrays[1], pdfGpuArrays[0]

        # Transfer data from gpu to cpu
        for cpuArr, gpuArr in zip(pdfArrays, pdfGpuArrays):
            gpuArr.get(cpuArr)

        getMacroscopic(pdfs=pdfArrays[0], density=densityArr[0], velocity=velocityArr[0])
        return pdfArrays[0], densityArr[0], velocityArr[0]

    cpuTimeLoop.kernel = lbmKernel
    gpuTimeLoop.kernel = lbmKernel

    return gpuTimeLoop if optimizationParams['target'] == 'gpu' else cpuTimeLoop


def createLidDrivenCavity(domainSize, lidVelocity=0.005, optimizationParams={}, lbmKernel=None,
                          kernelParams={}, **kwargs):
    def boundarySetupFunction(boundaryHandling, method):
        myUbb = partial(ubb, velocity=[lidVelocity, 0, 0][:method.dim])
        myUbb.name = 'ubb'
        boundaryHandling.setBoundary(myUbb, sliceFromDirection('N', method.dim))
        for direction in ('W', 'E', 'S') if method.dim == 2 else ('W', 'E', 'S', 'T', 'B'):
            boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, method.dim))

    return createScenario(domainSize, boundarySetupFunction, kwargs, optimizationParams, lbmKernel=lbmKernel,
                          kernelParams=kernelParams)


def createPressureGradientDrivenChannel(dim, pressureDifference, domainSize=None, radius=None, length=None,
                                        lbmKernel=None, optimizationParams={}, boundarySetupFunctions=[],
                                        kernelParams={}, **kwargs):
    assert dim in (2, 3)

    if radius is not None:
        assert length is not None
        if dim == 3:
            domainSize = (length, 2 * radius + 1, 2 * radius + 1)
            roundChannel = True
        else:
            if domainSize is None:
                domainSize = (length, 2 * radius)
    else:
        roundChannel = False

    def boundarySetupFunction(boundaryHandling, method):
        pressureBoundaryInflow = partial(fixedDensity, density=1.0 + pressureDifference)
        pressureBoundaryInflow.__name__ = "Inflow"

        pressureBoundaryOutflow = partial(fixedDensity, density=1.0)
        pressureBoundaryOutflow.__name__ = "Outflow"
        boundaryHandling.setBoundary(pressureBoundaryInflow, sliceFromDirection('W', method.dim))
        boundaryHandling.setBoundary(pressureBoundaryOutflow, sliceFromDirection('E', method.dim))

        if method.dim == 2:
            for direction in ('N', 'S'):
                boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, method.dim))
        elif method.dim == 3:
            if roundChannel:
                noSlipIdx = boundaryHandling.addBoundary(noSlip)
                ff = boundaryHandling.flagField
                yMid = ff.shape[1] // 2
                zMid = ff.shape[2] // 2
                y, z = np.meshgrid(range(ff.shape[1]), range(ff.shape[2]))
                ff[(y - yMid) ** 2 + (z - zMid) ** 2 > radius ** 2] = noSlipIdx
            else:
                for direction in ('N', 'S', 'T', 'B'):
                    boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, method.dim))

        for userFunction in boundarySetupFunctions:
            userFunction(boundaryHandling, method, domainSize)

    assert domainSize is not None
    return createScenario(domainSize, boundarySetupFunction, kwargs, optimizationParams, lbmKernel=lbmKernel,
                          kernelParams=kernelParams)


def createForceDrivenChannel(dim, force, domainSize=None, radius=None, length=None, lbmKernel=None,
                             optimizationParams={}, initialVelocity=None, boundarySetupFunctions=[],
                             kernelParams={}, **kwargs):
    assert dim in (2, 3)
    kwargs['force'] = tuple([force, 0, 0][:dim])

    if radius is not None:
        assert length is not None
        if dim == 3:
            domainSize = (length, 2 * radius + 1, 2 * radius + 1)
            roundChannel = True
        else:
            if domainSize is None:
                domainSize = (length, 2 * radius)
    else:
        roundChannel = False

    def boundarySetupFunction(boundaryHandling, method):
        if method.dim == 2:
            for direction in ('N', 'S'):
                boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, method.dim))
        elif method.dim == 3:
            if roundChannel:
                noSlipIdx = boundaryHandling.addBoundary(noSlip)
                ff = boundaryHandling.flagField
                yMid = ff.shape[1] // 2
                zMid = ff.shape[2] // 2
                y, z = np.meshgrid(range(ff.shape[1]), range(ff.shape[2]))
                ff[(y - yMid) ** 2 + (z - zMid) ** 2 > radius ** 2] = noSlipIdx
            else:
                for direction in ('N', 'S', 'T', 'B'):
                    boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, method.dim))
        for userFunction in boundarySetupFunctions:
            userFunction(boundaryHandling, method, domainSize)

    def periodicity(pdfArr):
        pdfArr[0, :, :] = pdfArr[-2, :, :]
        pdfArr[-1, :, :] = pdfArr[1, :, :]

    assert domainSize is not None
    if 'forceModel' not in kwargs:
        kwargs['forceModel'] = 'guo'

    return createScenario(domainSize, boundarySetupFunction, kwargs, optimizationParams, lbmKernel=lbmKernel,
                          initialVelocity=initialVelocity, preUpdateFunctions=[periodicity],
                          kernelParams=kernelParams)

