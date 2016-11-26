from waLBerla import lbm
import lbmpy.forcemodels as forcemodels
from pystencils.cpu.kernelcreation import addOpenMP
from lbmpy.cumulantlatticemodel import CumulantRelaxationLatticeModel, CorrectedD3Q27Collision


def convertWalberlaToLbmpyLatticeModel(lm):
    from lbmpy.latticemodel import makeSRT, makeTRT, makeMRT
    stencil = tuple(lm.directions)

    def getForce():
        dim = len(stencil[0])
        forceAsList = list(lm.forceModel.force())
        return tuple(forceAsList[:dim])

    if type(lm.forceModel) == lbm.forceModels.SimpleConstant:
        forceModel = forcemodels.Simple(getForce())
    elif type(lm.forceModel) == lbm.forceModels.LuoConstant:
        forceModel = forcemodels.Luo(getForce())
    elif type(lm.forceModel) == lbm.forceModels.GuoConstant:
        # currently only works with SRT -> more complex Guo version taking omega_bulk has to be implemented
        forceModel = forcemodels.Guo(getForce(), lm.collisionModel.omega)
    elif type(lm.forceModel) == lbm.forceModels.NoForce:
        forceModel = None
    else:
        raise NotImplementedError("No such force model in lbmpy")

    if type(lm.collisionModel) == lbm.collisionModels.SRT:
        return makeSRT(stencil, compressible=lm.compressible, order=lm.equilibriumAccuracyOrder, forceModel=forceModel)
    elif type(lm.collisionModel) == lbm.collisionModels.TRT:
        return makeTRT(stencil, compressible=lm.compressible, order=lm.equilibriumAccuracyOrder, forceModel=forceModel)
    elif type(lm.collisionModel) == lbm.collisionModels.D3Q19MRT:
        return makeMRT(stencil, compressible=lm.compressible, order=lm.equilibriumAccuracyOrder, forceModel=forceModel)
    elif type(lm.collisionModel) == lbm.collisionModels.D3Q27Cumulant:
        coll = CorrectedD3Q27Collision(lm.collisionModel.relaxationRates)
        return CumulantRelaxationLatticeModel(stencil, coll)
    else:
        raise ValueError("Unknown lattice model")


def makeWalberlaSourceDestinationSweep(kernelFunctionNode, sourceFieldName='src', destinationFieldName='dst',
                                       is2D=False):
    from waLBerla import field
    from pystencils.cpu.cpujit import buildCTypeArgumentList, compileAndLoad
    swapFields = {}

    func = compileAndLoad(kernelFunctionNode)[kernelFunctionNode.functionName]
    func.restype = None

    def f(**kwargs):
        src = kwargs[sourceFieldName]
        sizeInfo = (src.size, src.allocSize, src.layout)
        if sizeInfo not in swapFields:
            swapFields[sizeInfo] = src.cloneUninitialized()
        dst = swapFields[sizeInfo]

        kwargs[sourceFieldName] = field.toArray(src, withGhostLayers=True)
        kwargs[destinationFieldName] = field.toArray(dst, withGhostLayers=True)

        # Since waLBerla does not really support 2D domains a small hack is required here
        if is2D:
            assert kwargs[sourceFieldName].shape[2] in [1, 3]
            assert kwargs[destinationFieldName].shape[2] in [1, 3]
            kwargs[sourceFieldName] = kwargs[sourceFieldName][:, :, 1, :]
            kwargs[destinationFieldName] = kwargs[destinationFieldName][:, :, 1, :]

        args = buildCTypeArgumentList(kernelFunctionNode.parameters, kwargs)
        func(*args)
        src.swapDataPointers(dst)

    return f


def makeLbmpySweepFromWalberlaLatticeModel(walberlaLatticeModel, blocks, pdfFieldName,
                                           variableSize=False, replaceRelaxationTimes=False, doCSE=False,
                                           splitInnerLoop=True, openmpThreads=True):
    from lbmpy.lbmgenerator import createStreamCollideUpdateRule, createLbmSplitGroups
    from pystencils.cpu import createKernel
    from waLBerla import field

    if type(walberlaLatticeModel.collisionModel) == lbm.collisionModels.SRT:
        params = {'omega': walberlaLatticeModel.collisionModel.omega}
    elif type(walberlaLatticeModel.collisionModel) == lbm.collisionModels.TRT:
        params = {'lambda_o': walberlaLatticeModel.collisionModel.lambda_d,
                  'lambda_e': walberlaLatticeModel.collisionModel.lambda_e, }
    elif type(walberlaLatticeModel.collisionModel) == lbm.collisionModels.D3Q19MRT:
        s = walberlaLatticeModel.collisionModel.relaxationRates
        params = {}
        for i in [1, 2, 4, 9, 10, 16]:
            params["s_%d" % (i,)] = s[i]
    elif type(walberlaLatticeModel.collisionModel) == lbm.collisionModels.D3Q27Cumulant:
        params = {}
        pass
    else:
        raise ValueError("Unknown lattice model")

    lm = convertWalberlaToLbmpyLatticeModel(walberlaLatticeModel)
    if replaceRelaxationTimes:
        lm.setCollisionDOFs(params)
        params = {}

    numpyField = None
    if not variableSize:
        numpyField = field.toArray(blocks[0][pdfFieldName], withGhostLayers=True)
        dim = len(walberlaLatticeModel.directions[0])
        if dim == 2:
            numpyField = numpyField[:, :, 1, :]

    lbmUpdateRule = createStreamCollideUpdateRule(lm, numpyField=numpyField, doCSE=doCSE)
    splitGroups = createLbmSplitGroups(lm, lbmUpdateRule.equations) if splitInnerLoop else []
    funcNode = createKernel(lbmUpdateRule.equations, splitGroups=splitGroups)

    if openmpThreads:
        numThreads = None
        if type(openmpThreads) is int:
            numThreads = openmpThreads
        addOpenMP(funcNode, numThreads=numThreads)

    sweepFunction = makeWalberlaSourceDestinationSweep(funcNode, 'src', 'dst', is2D=(lm.dim == 2))
    sweepFunction = makeWalberlaSourceDestinationSweep(funcNode, 'src', 'dst', is2D=(lm.dim == 2))
    return lambda block: sweepFunction(src=block[pdfFieldName], **params)


def createBoundaryIndexListFromWalberlaFlagField(flagField, stencil, boundaryFlag, fluidFlag):
    import waLBerla as wlb
    from lbmpy.boundaries import createBoundaryIndexList
    flagFieldArr = wlb.field.toArray(flagField, withGhostLayers=True)
    fluidMask = flagField.flag(fluidFlag)
    boundaryMask = flagField.flag(boundaryFlag)
    gl = flagField.nrOfGhostLayers
    dim = len(stencil[0])
    flagFieldArr = flagFieldArr[:, :, :, 0]
    if dim == 2:
        flagFieldArr = flagFieldArr[:, :, gl]

    return createBoundaryIndexList(flagFieldArr, gl, stencil, boundaryMask, fluidMask)

