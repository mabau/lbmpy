import functools
from copy import deepcopy
from lbmpy.simplificationfactory import create_simplification_strategy
from pystencils import create_kernel, CreateKernelConfig
from pystencils.field import Field, get_layout_of_array
from pystencils.enums import Target

from lbmpy.advanced_streaming.utility import get_accessor, Timestep


def pdf_initialization_assignments(lb_method, density, velocity, pdfs,
                                   streaming_pattern='pull', previous_timestep=Timestep.BOTH,
                                   set_pre_collision_pdfs=False):
    """Assignments to initialize the pdf field with equilibrium"""
    if isinstance(pdfs, Field):
        accessor = get_accessor(streaming_pattern, previous_timestep)
        if set_pre_collision_pdfs:
            field_accesses = accessor.read(pdfs, lb_method.stencil)
        else:
            field_accesses = accessor.write(pdfs, lb_method.stencil)
    elif streaming_pattern == 'pull' and not set_pre_collision_pdfs:
        field_accesses = pdfs
    else:
        raise ValueError("Invalid value of pdfs: A PDF field reference is required to derive "
                         + f"initialization assignments for streaming pattern {streaming_pattern}.")

    if isinstance(density, Field):
        density = density.center

    if isinstance(velocity, Field):
        velocity = velocity.center_vector

    cqc = lb_method.conserved_quantity_computation
    inp_eqs = cqc.equilibrium_input_equations_from_init_values(density, velocity, force_substitution=False)
    setter_eqs = lb_method.get_equilibrium(conserved_quantity_equations=inp_eqs)
    setter_eqs = setter_eqs.new_with_substitutions({sym: field_accesses[i]
                                                    for i, sym in enumerate(lb_method.post_collision_pdf_symbols)})
    return setter_eqs


def macroscopic_values_getter(lb_method, density, velocity, pdfs,
                              streaming_pattern='pull', previous_timestep=Timestep.BOTH,
                              use_pre_collision_pdfs=False):
    if isinstance(pdfs, Field):
        accessor = get_accessor(streaming_pattern, previous_timestep)
        if use_pre_collision_pdfs:
            field_accesses = accessor.read(pdfs, lb_method.stencil)
        else:
            field_accesses = accessor.write(pdfs, lb_method.stencil)
    elif streaming_pattern == 'pull' and not use_pre_collision_pdfs:
        field_accesses = pdfs
    else:
        raise ValueError("Invalid value of pdfs: A PDF field reference is required to derive "
                         + f"getter assignments for streaming pattern {streaming_pattern}.")
    cqc = lb_method.conserved_quantity_computation
    assert not (velocity is None and density is None)
    output_spec = {}
    if velocity is not None:
        output_spec['velocity'] = velocity
    if density is not None:
        output_spec['density'] = density
    return cqc.output_equations_from_pdfs(field_accesses, output_spec)


macroscopic_values_setter = pdf_initialization_assignments


def compile_macroscopic_values_getter(lb_method, output_quantities, pdf_arr=None,
                                      ghost_layers=1, iteration_slice=None,
                                      field_layout='numpy', target=Target.CPU,
                                      streaming_pattern='pull', previous_timestep=Timestep.BOTH):
    """
    Create kernel to compute macroscopic value(s) from a pdf field (e.g. density or velocity)

    Args:
        lb_method: instance of :class:`lbmpy.methods.AbstractLbMethod`
        output_quantities: sequence of quantities to compute e.g. ['density', 'velocity']
        pdf_arr: optional numpy array for pdf field - used to get optimal loop structure for kernel
        ghost_layers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
                      that should be excluded from the iteration. If None, the number of ghost layers 
                      is determined automatically and assumed to be equal for all dimensions.        
        iteration_slice: if not None, iteration is done only over this slice of the field
        field_layout: layout for output field, also used for pdf field if pdf_arr is not given
        target: `Target.CPU` or `Target.GPU`
        previous_step_accessor: The accessor used by the streaming pattern of the previous timestep

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
    previous_step_accessor = get_accessor(streaming_pattern, previous_timestep)
    pdf_symbols = previous_step_accessor.write(pdf_field, stencil)

    eqs = cqc.output_equations_from_pdfs(pdf_symbols, output_mapping).all_assignments

    config = CreateKernelConfig(target=target, ghost_layers=ghost_layers, iteration_slice=iteration_slice)
    kernel = create_kernel(eqs, config=config).compile()

    def getter(pdfs, **kwargs):
        if pdf_arr is not None:
            assert pdfs.shape == pdf_arr.shape and pdfs.strides == pdf_arr.strides, \
                "Pdf array not matching blueprint which was used to compile" + str(pdfs.shape) + str(pdf_arr.shape)
        if not set(output_quantities).issubset(kwargs.keys()):
            raise ValueError("You have to specify the output field for each of the following quantities: %s"
                             % (str(output_quantities),))
        kernel(pdfs=pdfs, **kwargs)

    return getter


def compile_macroscopic_values_setter(lb_method, quantities_to_set, pdf_arr=None,
                                      ghost_layers=1, iteration_slice=None,
                                      field_layout='numpy', target=Target.CPU,
                                      streaming_pattern='pull', previous_timestep=Timestep.BOTH):
    """
    Creates a function that sets a pdf field to specified macroscopic quantities
    The returned function can be called with the pdf field to set as single argument

    Args:
        lb_method: instance of :class:`lbmpy.methods.AbstractLbMethod`
        quantities_to_set: map from conserved quantity name to fixed value or numpy array
        pdf_arr: optional numpy array for pdf field - used to get optimal loop structure for kernel
        ghost_layers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
                      that should be excluded from the iteration. If None, the number of ghost layers 
                      is determined automatically and assumed to be equal for all dimensions.        
        iteration_slice: if not None, iteration is done only over this slice of the field
        field_layout: layout of the pdf field if pdf_arr was not given
        target: `Target.CPU` or `Target.GPU`
        previous_step_accessor: The accessor used by the streaming pattern of the previous timestep

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

    cq_equations = cqc.equilibrium_input_equations_from_init_values(**value_map, force_substitution=False)

    eq = lb_method.get_equilibrium(conserved_quantity_equations=cq_equations)
    if at_least_one_field_input:
        simplification = create_simplification_strategy(lb_method)
        eq = simplification(eq)
    else:
        eq = eq.new_without_subexpressions()

    previous_step_accessor = get_accessor(streaming_pattern, previous_timestep)
    write_accesses = previous_step_accessor.write(pdf_field, lb_method.stencil)

    substitutions = {sym: write_accesses[i] for i, sym in enumerate(lb_method.post_collision_pdf_symbols)}
    eq = eq.new_with_substitutions(substitutions).all_assignments

    config = CreateKernelConfig(target=target)
    kernel = create_kernel(eq, config=config).compile()
    kernel = functools.partial(kernel, **fixed_kernel_parameters)

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
    density_symbol = cqc.density_symbol
    velocity_symbols = cqc.velocity_symbols

    # density is computed from pdfs
    eq_input_from_pdfs = cqc.equilibrium_input_equations_from_pdfs(
        method.pre_collision_pdf_symbols, force_substitution=False)
    eq_input_from_pdfs = eq_input_from_pdfs.new_filtered([density_symbol])
    # velocity is read from input field
    vel_symbols = [velocity_field(i) for i in range(method.dim)]
    eq_input_from_field = cqc.equilibrium_input_equations_from_init_values(
        velocity=vel_symbols, force_substitution=False)
    eq_input_from_field = eq_input_from_field.new_filtered(velocity_symbols)
    # then both are merged together
    eq_input = eq_input_from_pdfs.new_merged(eq_input_from_field)

    # set first order relaxation rate
    method = deepcopy(method)
    method.set_first_moment_relaxation_rate(velocity_relaxation_rate)

    simplification_strategy = create_simplification_strategy(method)
    new_collision_rule = simplification_strategy(method.get_collision_rule(eq_input))

    return new_collision_rule
