from copy import deepcopy
from pystencils.field import Field, get_layout_of_array
from lbmpy.simplificationfactory import create_simplification_strategy


def compileMacroscopicValuesGetter(lb_method, outputQuantities, pdfArr=None, field_layout='numpy', target='cpu'):
    """
    Create kernel to compute macroscopic value(s) from a pdf field (e.g. density or velocity)

    :param lb_method: instance of :class:`lbmpy.methods.AbstractLbMethod`
    :param outputQuantities: sequence of quantities to compute e.g. ['density', 'velocity']
    :param pdfArr: optional numpy array for pdf field - used to get optimal loop structure for kernel
    :param field_layout: layout for output field, also used for pdf field if pdfArr is not given
    :param target: 'cpu' or 'gpu'
    :return: a function to compute macroscopic values:
        - pdf_array
        - keyword arguments from name of conserved quantity (as in outputQuantities) to numpy field
    """
    if not (isinstance(outputQuantities, list) or isinstance(outputQuantities, tuple)):
        outputQuantities = [outputQuantities]

    cqc = lb_method.conserved_quantity_computation
    unknownQuantities = [oq for oq in outputQuantities if oq not in cqc.conserved_quantities]
    if unknownQuantities:
        raise ValueError("No such conserved quantity: %s, conserved quantities are %s" %
                         (str(unknownQuantities), str(cqc.conserved_quantities.keys())))

    if pdfArr is None:
        pdfField = Field.create_generic('pdfs', lb_method.dim, index_dimensions=1, layout=field_layout)
    else:
        pdfField = Field.create_from_numpy_array('pdfs', pdfArr, index_dimensions=1)

    outputMapping = {}
    for outputQuantity in outputQuantities:
        numberOfElements = cqc.conserved_quantities[outputQuantity]
        assert numberOfElements >= 1

        indDims = 0 if numberOfElements <= 1 else 1
        if pdfArr is None:
            outputField = Field.create_generic(outputQuantity, lb_method.dim, layout=field_layout, index_dimensions=indDims)
        else:
            outputFieldShape = pdfArr.shape[:-1]
            if indDims > 0:
                outputFieldShape += (numberOfElements,)
                field_layout = get_layout_of_array(pdfArr)
            else:
                field_layout = get_layout_of_array(pdfArr, index_dimension_ids=[len(pdfField.shape) - 1])
            outputField = Field.create_fixed_size(outputQuantity, outputFieldShape, indDims, pdfArr.dtype, field_layout)

        outputMapping[outputQuantity] = [outputField(i) for i in range(numberOfElements)]
        if len(outputMapping[outputQuantity]) == 1:
            outputMapping[outputQuantity] = outputMapping[outputQuantity][0]

    stencil = lb_method.stencil
    pdfSymbols = [pdfField(i) for i in range(len(stencil))]
    eqs = cqc.output_equations_from_pdfs(pdfSymbols, outputMapping).all_assignments

    if target == 'cpu':
        import pystencils.cpu as cpu
        kernel = cpu.make_python_function(cpu.create_kernel(eqs))
    elif target == 'gpu':
        import pystencils.gpucuda as gpu
        kernel = gpu.make_python_function(gpu.create_cuda_kernel(eqs))
    else:
        raise ValueError("Unknown target '%s'. Possible targets are 'cpu' and 'gpu'" % (target,))

    def getter(pdfs, **kwargs):
        if pdfArr is not None:
            assert pdfs.shape == pdfArr.shape and pdfs.strides == pdfArr.strides, \
                "Pdf array not matching blueprint which was used to compile" + str(pdfs.shape) + str(pdfArr.shape)
        if not set(outputQuantities).issubset(kwargs.keys()):
            raise ValueError("You have to specify the output field for each of the following quantities: %s"
                             % (str(outputQuantities),))
        kernel(pdfs=pdfs, **kwargs)

    return getter


def compileMacroscopicValuesSetter(lb_method, quantitiesToSet, pdfArr=None, field_layout='numpy', target='cpu'):
    """
    Creates a function that sets a pdf field to specified macroscopic quantities
    The returned function can be called with the pdf field to set as single argument

    :param lb_method: instance of :class:`lbmpy.methods.AbstractLbMethod`
    :param quantitiesToSet: map from conserved quantity name to fixed value or numpy array
    :param pdfArr: optional numpy array for pdf field - used to get optimal loop structure for kernel
    :param field_layout: layout of the pdf field if pdfArr was not given
    :param target: 'cpu' or 'gpu'
    :return: function taking pdf array as single argument and which sets the field to the given values
    """
    if pdfArr is not None:
        pdfField = Field.create_from_numpy_array('pdfs', pdfArr, index_dimensions=1)
    else:
        pdfField = Field.create_generic('pdfs', lb_method.dim, index_dimensions=1, layout=field_layout)

    fixedKernelParameters = {}
    cqc = lb_method.conserved_quantity_computation

    valueMap = {}
    atLeastOneFieldInput = False
    for quantityName, value in quantitiesToSet.items():
        if hasattr(value, 'shape'):
            fixedKernelParameters[quantityName] = value
            atLeastOneFieldInput = True
            numComponents = cqc.conserved_quantities[quantityName]
            field = Field.create_from_numpy_array(quantityName, value, index_dimensions=0 if numComponents <= 1 else 1)
            if numComponents == 1:
                value = field(0)
            else:
                value = [field(i) for i in range(numComponents)]

        valueMap[quantityName] = value

    cqEquations = cqc.equilibrium_input_equations_from_init_values(**valueMap)

    eq = lb_method.get_equilibrium(conserved_quantity_equations=cqEquations)
    if atLeastOneFieldInput:
        simplification = create_simplification_strategy(lb_method)
        eq = simplification(eq)
    else:
        eq = eq.new_without_subexpressions()

    substitutions = {sym: pdfField(i) for i, sym in enumerate(lb_method.post_collision_pdf_symbols)}
    eq = eq.new_with_substitutions(substitutions).all_assignments

    if target == 'cpu':
        import pystencils.cpu as cpu
        kernel = cpu.make_python_function(cpu.create_kernel(eq), argument_dict=fixedKernelParameters)
    elif target == 'gpu':
        import pystencils.gpucuda as gpu
        kernel = gpu.make_python_function(gpu.create_cuda_kernel(eq), argument_dict=fixedKernelParameters)
    else:
        raise ValueError("Unknown target '%s'. Possible targets are 'cpu' and 'gpu'" % (target,))

    def setter(pdfs, **kwargs):
        if pdfArr is not None:
            assert pdfs.shape == pdfArr.shape and pdfs.strides == pdfArr.strides, \
                "Pdf array not matching blueprint which was used to compile" + str(pdfs.shape) + str(pdfArr.shape)
        kernel(pdfs=pdfs, **kwargs)

    return setter


def createAdvancedVelocitySetterCollisionRule(lb_method, velocityArray, velocityRelaxationRate=0.8):

    velocityField = Field.create_from_numpy_array('velInput', velocityArray, index_dimensions=1)

    cqc = lb_method.conserved_quantity_computation
    densitySymbol = cqc.defined_symbols(order=0)[1]
    velocitySymbols = cqc.defined_symbols(order=1)[1]

    # density is computed from pdfs
    eqInputFromPdfs = cqc.equilibrium_input_equations_from_pdfs(lb_method.pre_collision_pdf_symbols)
    eqInputFromPdfs = eqInputFromPdfs.new_filtered([densitySymbol])
    # velocity is read from input field
    velSymbols = [velocityField(i) for i in range(lb_method.dim)]
    eqInputFromField = cqc.equilibrium_input_equations_from_init_values(velocity=velSymbols)
    eqInputFromField = eqInputFromField.new_filtered(velocitySymbols)
    # then both are merged together
    eqInput = eqInputFromPdfs.new_merged(eqInputFromField)

    # set first order relaxation rate
    lb_method = deepcopy(lb_method)
    lb_method.set_first_moment_relaxation_rate(velocityRelaxationRate)

    simplificationStrategy = create_simplification_strategy(lb_method)
    newCollisionRule = simplificationStrategy(lb_method.get_collision_rule(eqInput))

    return newCollisionRule


def compileAdvancedVelocitySetter(method, velocityArray, velocityRelaxationRate=0.8, pdfArr=None,
                                  field_layout='numpy', optimization={}):
    """
    Advanced initialization of velocity field through iteration procedure according to
    Mei, Luo, Lallemand and Humieres: Consistent initial conditions for LBM simulations, 2005

    :param method:
    :param velocityArray: array with velocity field
    :param velocityRelaxationRate: relaxation rate for the velocity moments - determines convergence behaviour
                                   of the initialization scheme
    :param pdfArr: optional numpy array for pdf field - used to get optimal loop structure for kernel
    :param field_layout: layout of the pdf field if pdfArr was not given
    :param optimization: dictionary with optimization hints
    :return: stream-collide update function
    """
    from lbmpy.updatekernels import create_stream_pull_collide_kernel
    from lbmpy.creationfunctions import create_lb_ast, create_lb_function
    newCollisionRule = createAdvancedVelocitySetterCollisionRule(method, velocityArray, velocityRelaxationRate)
    update_rule = create_stream_pull_collide_kernel(newCollisionRule, pdfArr, generic_layout=field_layout)
    ast = create_lb_ast(update_rule, optimization)
    return create_lb_function(ast, optimization)
