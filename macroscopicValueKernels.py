from copy import deepcopy
from pystencils.field import Field, getLayoutFromNumpyArray
from lbmpy.simplificationfactory import createSimplificationStrategy


def compileMacroscopicValuesGetter(lbMethod, outputQuantities, pdfArr=None, fieldLayout='numpy', target='cpu'):
    """
    Create kernel to compute macroscopic value(s) from a pdf field (e.g. density or velocity)

    :param lbMethod: instance of :class:`lbmpy.methods.AbstractLbMethod`
    :param outputQuantities: sequence of quantities to compute e.g. ['density', 'velocity']
    :param pdfArr: optional numpy array for pdf field - used to get optimal loop structure for kernel
    :param fieldLayout: layout for output field, also used for pdf field if pdfArr is not given
    :param target: 'cpu' or 'gpu'
    :return: a function to compute macroscopic values:
        - pdfArray
        - keyword arguments from name of conserved quantity (as in outputQuantities) to numpy field
    """
    if not (isinstance(outputQuantities, list) or isinstance(outputQuantities, tuple)):
        outputQuantities = [outputQuantities]

    cqc = lbMethod.conservedQuantityComputation
    unknownQuantities = [oq for oq in outputQuantities if oq not in cqc.conservedQuantities]
    if unknownQuantities:
        raise ValueError("No such conserved quantity: %s, conserved quantities are %s" %
                         (str(unknownQuantities), str(cqc.conservedQuantities.keys())))

    if pdfArr is None:
        pdfField = Field.createGeneric('pdfs', lbMethod.dim, indexDimensions=1, layout=fieldLayout)
    else:
        pdfField = Field.createFromNumpyArray('pdfs', pdfArr, indexDimensions=1)

    outputMapping = {}
    for outputQuantity in outputQuantities:
        numberOfElements = cqc.conservedQuantities[outputQuantity]
        assert numberOfElements >= 1

        indDims = 0 if numberOfElements <= 1 else 1
        if pdfArr is None:
            fieldLayout = getLayoutFromNumpyArray(pdfArr, indexDimensionIds=[len(pdfField.shape) - 1])
            outputField = Field.createGeneric(outputQuantity, lbMethod.dim, layout=fieldLayout, indexDimensions=indDims)
        else:
            outputFieldShape = pdfArr.shape[:-1]
            if indDims > 0:
                outputFieldShape += (numberOfElements,)
                fieldLayout = getLayoutFromNumpyArray(pdfArr)
            else:
                fieldLayout = getLayoutFromNumpyArray(pdfArr, indexDimensionIds=[len(pdfField.shape) - 1])
            outputField = Field.createFixedSize(outputQuantity, outputFieldShape, indDims, pdfArr.dtype, fieldLayout)

        outputMapping[outputQuantity] = [outputField(i) for i in range(numberOfElements)]
        if len(outputMapping[outputQuantity]) == 1:
            outputMapping[outputQuantity] = outputMapping[outputQuantity][0]

    stencil = lbMethod.stencil
    pdfSymbols = [pdfField(i) for i in range(len(stencil))]
    eqs = cqc.outputEquationsFromPdfs(pdfSymbols, outputMapping).allEquations

    if target == 'cpu':
        import pystencils.cpu as cpu
        kernel = cpu.makePythonFunction(cpu.createKernel(eqs))
    elif target == 'gpu':
        import pystencils.gpucuda as gpu
        kernel = gpu.makePythonFunction(gpu.createCUDAKernel(eqs))
    else:
        raise ValueError("Unknown target '%s'. Possible targets are 'cpu' and 'gpu'" % (target,))

    def getter(pdfs, **kwargs):
        if pdfArr is not None:
            assert pdfs.shape == pdfArr.shape and pdfs.strides == pdfArr.strides, \
                "Pdf array not matching blueprint which was used to compile" + str(pdfs.shape) + str(pdfArr.shape)
        if not set(outputQuantities) == set(kwargs.keys()):
            raise ValueError("You have to specify the output field for each of the following quantities: %s"
                             % (str(outputQuantities),))
        kernel(pdfs=pdfs, **kwargs)

    return getter


def compileMacroscopicValuesSetter(lbMethod, quantitiesToSet, pdfArr=None, fieldLayout='numpy', target='cpu'):
    """
    Creates a function that sets a pdf field to specified macroscopic quantities
    The returned function can be called with the pdf field to set as single argument

    :param lbMethod: instance of :class:`lbmpy.methods.AbstractLbMethod`
    :param quantitiesToSet: map from conserved quantity name to fixed value or numpy array
    :param pdfArr: optional numpy array for pdf field - used to get optimal loop structure for kernel
    :param fieldLayout: layout of the pdf field if pdfArr was not given
    :param target: 'cpu' or 'gpu'
    :return: function taking pdf array as single argument and which sets the field to the given values
    """
    if pdfArr is not None:
        pdfField = Field.createFromNumpyArray('pdfs', pdfArr, indexDimensions=1)
    else:
        pdfField = Field.createGeneric('pdfs', lbMethod.dim, indexDimensions=1, layout=fieldLayout)

    fixedKernelParameters = {}
    cqc = lbMethod.conservedQuantityComputation

    valueMap = {}
    atLeastOneFieldInput = False
    for quantityName, value in quantitiesToSet.items():
        if hasattr(value, 'shape'):
            fixedKernelParameters[quantityName] = value
            atLeastOneFieldInput = True
            numComponents = cqc.conservedQuantities[quantityName]
            field = Field.createFromNumpyArray(quantityName, value, indexDimensions=0 if numComponents <= 1 else 1)
            if numComponents == 1:
                value = field(0)
            else:
                value = [field(i) for i in range(numComponents)]

        valueMap[quantityName] = value

    cqEquations = cqc.equilibriumInputEquationsFromInitValues(**valueMap)

    eq = lbMethod.getEquilibrium(conservedQuantityEquations=cqEquations)
    if atLeastOneFieldInput:
        simplification = createSimplificationStrategy(lbMethod)
        eq = simplification(eq)
    else:
        eq = eq.insertSubexpressions()

    substitutions = {sym: pdfField(i) for i, sym in enumerate(lbMethod.postCollisionPdfSymbols)}
    eq = eq.copyWithSubstitutionsApplied(substitutions).allEquations

    if target == 'cpu':
        import pystencils.cpu as cpu
        kernel = cpu.makePythonFunction(cpu.createKernel(eq), argumentDict=fixedKernelParameters)
    elif target == 'gpu':
        import pystencils.gpucuda as gpu
        kernel = gpu.makePythonFunction(gpu.createCUDAKernel(eq), argumentDict=fixedKernelParameters)
    else:
        raise ValueError("Unknown target '%s'. Possible targets are 'cpu' and 'gpu'" % (target,))

    def setter(pdfs):
        if pdfArr is not None:
            assert pdfs.shape == pdfArr.shape and pdfs.strides == pdfArr.strides, \
                "Pdf array not matching blueprint which was used to compile" + str(pdfs.shape) + str(pdfArr.shape)
        kernel(pdfs=pdfs)

    return setter


def compileAdvancedVelocitySetter(lbMethod, velocityArray, velocityRelaxationRate=1.3, pdfArr=None):
    """
    Advanced initialization of velocity field through iteration procedure according to
    Mei, Luo, Lallemand and Humieres: Consistent initial conditions for LBM simulations, 2005

    :param lbMethod:
    :param velocityArray: array with velocity field
    :param velocityRelaxationRate: relaxation rate for the velocity moments - determines convergence behaviour
                                   of the initialization scheme
    :return: collision rule
    """
    velocityField = Field.createFromNumpyArray('velInput', velocityArray, indexDimensions=1)

    cqc = lbMethod.conservedQuantityComputation
    densitySymbol = cqc.definedSymbols(order=0)[1]
    velocitySymbols = cqc.definedSymbols(order=1)[1]

    # density is computed from pdfs
    eqInputFromPdfs = cqc.equilibriumInputEquationsFromPdfs(lbMethod.preCollisionPdfSymbols)
    eqInputFromPdfs = eqInputFromPdfs.extract([densitySymbol])
    # velocity is read from input field
    velSymbols = [velocityField(i) for i in range(lbMethod.dim)]
    eqInputFromField = cqc.equilibriumInputEquationsFromInitValues(velocity=velSymbols)
    eqInputFromField = eqInputFromField.extract(velocitySymbols)
    # then both are merged together
    eqInput = eqInputFromPdfs.merge(eqInputFromField)

    # set first order relaxation rate
    lbMethod = deepcopy(lbMethod)
    lbMethod.setFirstMomentRelaxationRate(velocityRelaxationRate)

    return lbMethod.getCollisionRule(eqInput)


