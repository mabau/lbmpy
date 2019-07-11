import functools
from copy import deepcopy

from lbmpy.simplificationfactory import create_simplification_strategy
from pystencils.field import Field, get_layout_of_array


def pdf_initialization_assignments(lb_method, density, velocity, pdfs):
    """Assignments to initialize the pdf field with equilibrium"""
    cqc = lb_method.conserved_quantity_computation
    inp_eqs = cqc.equilibrium_input_equations_from_init_values(density, velocity)
    setter_eqs = lb_method.get_equilibrium(conserved_quantity_equations=inp_eqs)
    setter_eqs = setter_eqs.new_with_substitutions({sym: pdfs[i]
                                                    for i, sym in enumerate(lb_method.post_collision_pdf_symbols)})
    return setter_eqs


def macroscopic_values_getter(lb_method, density, velocity, pdfs):
    cqc = lb_method.conserved_quantity_computation
    assert not (velocity is None and density is None)
    output_spec = {}
    if velocity is not None:
        output_spec['velocity'] = velocity
    if density is not None:
        output_spec['density'] = density
    return cqc.output_equations_from_pdfs(pdfs, output_spec)


macroscopic_values_setter = pdf_initialization_assignments


def compile_macroscopic_values_getter(lb_method, output_quantities, pdf_arr=None, field_layout='numpy', target='cpu'):
    """
    Create kernel to compute macroscopic value(s) from a pdf field (e.g. density or velocity)

    Args:
        lb_method: instance of :class:`lbmpy.methods.AbstractLbMethod`
        output_quantities: sequence of quantities to compute e.g. ['density', 'velocity']
        pdf_arr: optional numpy array for pdf field - used to get optimal loop structure for kernel
        field_layout: layout for output field, also used for pdf field if pdf_arr is not given
        target: 'cpu' or 'gpu'

    Returns:
        a function to compute macroscopic values:
        - pdf_array
        - keyword arguments from name of conserved quantity (as in output_quantities) to numpy field
    """
    if not (isinstance(output_quantities, list) or isinstance(output_quantities, tuple)):
        output_quantities = [output_quantities]

    cqc = lb_method.conserved_quantity_computation
    unknown_quantities = [oq for oq in output_quantities if oq not in cqc.conserved_quantities]
    if unknown_quantities:
        raise ValueError("No such conserved quantity: %s, conserved quantities are %s" %
                         (str(unknown_quantities), str(cqc.conserved_quantities.keys())))

    if pdf_arr is None:
        pdf_field = Field.create_generic('pdfs', lb_method.dim, index_dimensions=1, layout=field_layout)
    else:
        pdf_field = Field.create_from_numpy_array('pdfs', pdf_arr, index_dimensions=1)

    output_mapping = {}
    for output_quantity in output_quantities:
        number_of_elements = cqc.conserved_quantities[output_quantity]
        assert number_of_elements >= 1

        ind_dims = 0 if number_of_elements <= 1 else 1
        if pdf_arr is None:
            output_field = Field.create_generic(output_quantity, lb_method.dim, layout=field_layout,
                                                index_dimensions=ind_dims)
        else:
            output_field_shape = pdf_arr.shape[:-1]
            if ind_dims > 0:
                output_field_shape += (number_of_elements,)
                field_layout = get_layout_of_array(pdf_arr)
            else:
                field_layout = get_layout_of_array(pdf_arr, index_dimension_ids=[len(pdf_field.shape) - 1])
            output_field = Field.create_fixed_size(output_quantity, output_field_shape, ind_dims, pdf_arr.dtype,
                                                   field_layout)

        output_mapping[output_quantity] = [output_field(i) for i in range(number_of_elements)]
        if len(output_mapping[output_quantity]) == 1:
            output_mapping[output_quantity] = output_mapping[output_quantity][0]

    stencil = lb_method.stencil
    pdf_symbols = [pdf_field(i) for i in range(len(stencil))]
    eqs = cqc.output_equations_from_pdfs(pdf_symbols, output_mapping).all_assignments

    if target == 'cpu':
        import pystencils.cpu as cpu
        kernel = cpu.make_python_function(cpu.create_kernel(eqs))
    elif target == 'gpu':
        import pystencils.gpucuda as gpu
        kernel = gpu.make_python_function(gpu.create_cuda_kernel(eqs))
    else:
        raise ValueError("Unknown target '%s'. Possible targets are 'cpu' and 'gpu'" % (target,))

    def getter(pdfs, **kwargs):
        if pdf_arr is not None:
            assert pdfs.shape == pdf_arr.shape and pdfs.strides == pdf_arr.strides, \
                "Pdf array not matching blueprint which was used to compile" + str(pdfs.shape) + str(pdf_arr.shape)
        if not set(output_quantities).issubset(kwargs.keys()):
            raise ValueError("You have to specify the output field for each of the following quantities: %s"
                             % (str(output_quantities),))
        kernel(pdfs=pdfs, **kwargs)

    return getter


def compile_macroscopic_values_setter(lb_method, quantities_to_set, pdf_arr=None, field_layout='numpy', target='cpu'):
    """
    Creates a function that sets a pdf field to specified macroscopic quantities
    The returned function can be called with the pdf field to set as single argument

    Args:
        lb_method: instance of :class:`lbmpy.methods.AbstractLbMethod`
        quantities_to_set: map from conserved quantity name to fixed value or numpy array
        pdf_arr: optional numpy array for pdf field - used to get optimal loop structure for kernel
        field_layout: layout of the pdf field if pdf_arr was not given
        target: 'cpu' or 'gpu'

    Returns:
        function taking pdf array as single argument and which sets the field to the given values
    """
    if pdf_arr is not None:
        pdf_field = Field.create_from_numpy_array('pdfs', pdf_arr, index_dimensions=1)
    else:
        pdf_field = Field.create_generic('pdfs', lb_method.dim, index_dimensions=1, layout=field_layout)

    fixed_kernel_parameters = {}
    cqc = lb_method.conserved_quantity_computation

    value_map = {}
    at_least_one_field_input = False
    for quantity_name, value in quantities_to_set.items():
        if hasattr(value, 'shape'):
            fixed_kernel_parameters[quantity_name] = value
            at_least_one_field_input = True
            num_components = cqc.conserved_quantities[quantity_name]
            field = Field.create_from_numpy_array(quantity_name, value,
                                                  index_dimensions=0 if num_components <= 1 else 1)
            if num_components == 1:
                value = field(0)
            else:
                value = [field(i) for i in range(num_components)]

        value_map[quantity_name] = value

    cq_equations = cqc.equilibrium_input_equations_from_init_values(**value_map)

    eq = lb_method.get_equilibrium(conserved_quantity_equations=cq_equations)
    if at_least_one_field_input:
        simplification = create_simplification_strategy(lb_method)
        eq = simplification(eq)
    else:
        eq = eq.new_without_subexpressions()

    substitutions = {sym: pdf_field(i) for i, sym in enumerate(lb_method.post_collision_pdf_symbols)}
    eq = eq.new_with_substitutions(substitutions).all_assignments

    if target == 'cpu':
        import pystencils.cpu as cpu
        kernel = cpu.make_python_function(cpu.create_kernel(eq))
        kernel = functools.partial(kernel, **fixed_kernel_parameters)
    elif target == 'gpu':
        import pystencils.gpucuda as gpu
        kernel = gpu.make_python_function(gpu.create_cuda_kernel(eq))
        kernel = functools.partial(kernel, **fixed_kernel_parameters)
    else:
        raise ValueError("Unknown target '%s'. Possible targets are 'cpu' and 'gpu'" % (target,))

    def setter(pdfs, **kwargs):
        if pdf_arr is not None:
            assert pdfs.shape == pdf_arr.shape and pdfs.strides == pdf_arr.strides, \
                "Pdf array not matching blueprint which was used to compile" + str(pdfs.shape) + str(pdf_arr.shape)
        kernel(pdfs=pdfs, **kwargs)

    return setter


def create_advanced_velocity_setter_collision_rule(method, velocity_field: Field, velocity_relaxation_rate=0.8):
    """Advanced initialization of velocity field through iteration procedure.

    by Mei, Luo, Lallemand and Humieres: Consistent initial conditions for LBM simulations, 2005

    Args:
        method: lattice boltzmann method object
        velocity_field: pystencils field
        velocity_relaxation_rate: relaxation rate for the velocity moments - determines convergence behaviour
                                  of the initialization scheme

    Returns:
        LB collision rule
    """
    cqc = method.conserved_quantity_computation
    density_symbol = cqc.defined_symbols(order=0)[1]
    velocity_symbols = cqc.defined_symbols(order=1)[1]

    # density is computed from pdfs
    eq_input_from_pdfs = cqc.equilibrium_input_equations_from_pdfs(method.pre_collision_pdf_symbols)
    eq_input_from_pdfs = eq_input_from_pdfs.new_filtered([density_symbol])
    # velocity is read from input field
    vel_symbols = [velocity_field(i) for i in range(method.dim)]
    eq_input_from_field = cqc.equilibrium_input_equations_from_init_values(velocity=vel_symbols)
    eq_input_from_field = eq_input_from_field.new_filtered(velocity_symbols)
    # then both are merged together
    eq_input = eq_input_from_pdfs.new_merged(eq_input_from_field)

    # set first order relaxation rate
    method = deepcopy(method)
    method.set_first_moment_relaxation_rate(velocity_relaxation_rate)

    simplification_strategy = create_simplification_strategy(method)
    new_collision_rule = simplification_strategy(method.get_collision_rule(eq_input))

    return new_collision_rule
