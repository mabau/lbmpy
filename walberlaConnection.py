from waLBerla import lbm


def convertWalberlaToLbmpyLatticeModel(lm):
    from lbmpy.collisionoperator import makeSRT, makeTRT, makeMRT
    stencil = lm.directions
    if type(lm.collisionModel) == lbm.collisionModels.SRT:
        return makeSRT(stencil, compressible=lm.compressible, order=lm.equilibriumAccuracyOrder)
    elif type(lm.collisionModel) == lbm.collisionModels.TRT:
        return makeTRT(stencil, compressible=lm.compressible, order=lm.equilibriumAccuracyOrder)
    elif type(lm.collisionModel) == lbm.collisionModels.D3Q19MRT:
        return makeMRT(stencil, compressible=lm.compressible, order=lm.equilibriumAccuracyOrder)
    else:
        raise ValueError("Unknown lattice model")


def makeWalberlaSourceDestinationSweep(kernelFunctionNode, sourceFieldName='src', destinationFieldName='dst'):
    from waLBerla import field
    from lbmpy.jit import buildCTypeArgumentList, compileAndLoad
    from lbmpy.generator import LoopOverCoordinate
    swapFields = {}

    func = compileAndLoad(kernelFunctionNode)[kernelFunctionNode.functionName]
    func.restype = None

    # Counting the number of domain loops to get dimensionality
    dim = len(kernelFunctionNode.atoms(LoopOverCoordinate))

    def f(**kwargs):
        src = kwargs[sourceFieldName]
        sizeInfo = (src.size, src.allocSize, src.layout)
        if sizeInfo not in swapFields:
            swapFields[sizeInfo] = src.cloneUninitialized()
        dst = swapFields[sizeInfo]

        kwargs[sourceFieldName] = field.toArray(src, withGhostLayers=True)
        kwargs[destinationFieldName] = field.toArray(dst, withGhostLayers=True)

        # Since waLBerla does not really support 2D domains a small hack is required here
        if dim == 2:
            assert kwargs[sourceFieldName].shape[2] in [1, 3]
            assert kwargs[destinationFieldName].shape[2] in [1, 3]
            kwargs[sourceFieldName] = kwargs[sourceFieldName][:, :, 1, :]
            kwargs[destinationFieldName] = kwargs[destinationFieldName][:, :, 1, :]

        args = buildCTypeArgumentList(kernelFunctionNode, kwargs)
        func(*args)
        src.swapDataPointers(dst)

    return f


def makeLbmpySweepFromWalberlaLatticeModel(walberlaLatticeModel, blocks, pdfFieldName,
                                           variableSize=False, replaceRelaxationTimes=False):
    from lbmpy.lbmgenerator import createLbmEquations
    from lbmpy.generator import createKernel
    from waLBerla import field

    if type(walberlaLatticeModel.collisionModel) == lbm.collisionModels.SRT:
        params = {'omega': walberlaLatticeModel.collisionModel.omega}
    elif type(walberlaLatticeModel.collisionModel) == lbm.collisionModels.TRT:
        params = {'lambda_o': walberlaLatticeModel.collisionModel.lambda_d,
                  'lambda_e': walberlaLatticeModel.collisionModel.lambda_e, }
    elif type(walberlaLatticeModel.collisionModel == lbm.collisionModels.D3Q19MRT):
        s = walberlaLatticeModel.collisionModel.relaxationRates
        params = {}
        for i in [1, 2, 4, 9, 10, 16]:
            params["s_%d" % (i,)] = s[i]
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

    lbmEquations = createLbmEquations(lm, numpyField=numpyField)
    funcNode = createKernel(lbmEquations)
    print(funcNode.generateC())
    sweepFunction = makeWalberlaSourceDestinationSweep(funcNode, 'src', 'dst')
    sweepFunction = makeWalberlaSourceDestinationSweep(funcNode, 'src', 'dst')
    return lambda block: sweepFunction(src=block[pdfFieldName], **params)
