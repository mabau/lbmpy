import sympy as sp

from lbmpy.simplificationfactory import createSimplificationStrategy
from pystencils.field import Field


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
        outputField = Field.createGeneric(outputQuantity, lbMethod.dim, layout=fieldLayout,
                                          indexDimensions=0 if numberOfElements <= 1 else 1)

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
    """
    if pdfArr is not None:
        pdfField = Field.createFromNumpyArray('pdfs', pdfArr, indexDimensions=1)
    else:
        pdfField = Field.createGeneric('pdfs', lbMethod.dim, indexDimensions=1, layout=fieldLayout)

    fixedKernelParameters = {}

    valueMap = {}
    atLeastOneFieldInput = False
    for quantityName, value in quantitiesToSet.items():
        if hasattr(value, 'shape'):
            fixedKernelParameters[quantityName] = value
            value = Field.createFromNumpyArray(quantityName, value)
            atLeastOneFieldInput = True
        valueMap[quantityName] = value

    cqc = lbMethod.conservedQuantityComputation
    cqEquations = cqc.equilibriumInputEquationsFromInitValues(**valueMap)

    eq = lbMethod.getEquilibrium(conservedQuantityEquations=cqEquations)
    if atLeastOneFieldInput:
        simplification = createSimplificationStrategy(eq)
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


def compileAdvancedVelocitySetter(collisionRule, velocityArray, pdfArr=None):
    """
    Advanced initialization of velocity field through iteration procedure according to
    Mei, Luo, Lallemand and Humieres: Consistent initial conditions for LBM simulations, 2005

    Important: this procedure only works if a non-zero relaxation rate was used for the velocity moments!

    :param collisionRule: unsimplified collision rule
    :param velocityArray: array with velocity field
    :param pdfArr: optional array, to compile kernel with fixed layout and shape
    :return: function, that has to be called multiple times, with a pdf field (src/dst) until convergence
             similar to the normal streamCollide step, also with boundary handling
    """
    velocityField = Field.createFromNumpyArray('vel', velocityArray, indexDimensions=1)

    # create normal LBM kernel and replace velocity by expressions of velocity field
    from lbmpy_old.simplifications import sympyCSE
    latticeModel = collisionRule.latticeModel
    collisionRule = sympyCSE(collisionRule)
    collisionRule = streamPullWithSourceAndDestinationFields(collisionRule, pdfArr)

    replacements = {u_i: sp.Eq(u_i, velocityField(i)) for i, u_i in enumerate(latticeModel.symbolicVelocity)}

    newSubExpressions = [replacements[eq.lhs] if eq.lhs in replacements else eq for eq in collisionRule.subexpressions]
    newCollisionRule = LbmCollisionRule(collisionRule.updateEquations, newSubExpressions,
                                        latticeModel, collisionRule.updateEquationDirections)
    kernelAst = createKernel(newCollisionRule.equations)
    return makePythonFunction(kernelAst, {'vel': velocityArray})

