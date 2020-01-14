r"""
Creating LBM kernels
====================

The following list describes common parameters for the creation functions. They have to be passed as keyword parameters.

Method parameters
^^^^^^^^^^^^^^^^^

General:

- ``stencil='D2Q9'``: stencil name e.g. 'D2Q9', 'D3Q19'. See :func:`pystencils.stencils.get_stencil` for details
- ``method='srt'``: name of lattice Boltzmann method. This determines the selection and relaxation pattern of
  moments/cumulants, i.e. which moment/cumulant basis is chosen, and which of the basis vectors are relaxed together

    - ``srt``: single relaxation time (:func:`lbmpy.methods.create_srt`)
    - ``trt``: two relaxation time, first relaxation rate is for even moments and determines the viscosity (as in SRT),
      the second relaxation rate is used for relaxing odd moments, and controls the bulk viscosity.
      (:func:`lbmpy.methods.create_trt`)
    - ``mrt``: orthogonal multi relaxation time model, relaxation rates are used in this order for :
      shear modes, bulk modes, 3rd order modes, 4th order modes, etc.
      Requires also a parameter 'weighted' that should be True if the moments should be orthogonal w.r.t. weighted
      scalar product using the lattice weights. If `False` the normal scalar product is used.
      For custom definition of the method, a 'nested_moments' can be passed.
      For example: [ [1, x, y], [x*y, x**2, y**2], ... ] that groups all moments together that should be relaxed with
      the same rate. Literature values of this list can be obtained through
      :func:`lbmpy.methods.creationfunctions.mrt_orthogonal_modes_literature`.
      See also :func:`lbmpy.methods.create_mrt_orthogonal`
    - ``mrt3``: deprecated
    - ``mrt_raw``: non-orthogonal MRT where all relaxation rates can be specified independently i.e. there are as many
      relaxation rates as stencil entries. Look at the generated method in Jupyter to see which moment<->relaxation rate
      mapping (:func:`lbmpy.methods.create_mrt_raw`)
    - ``trt-kbc-n<N>`` where <N> is 1,2,3 or 4. Special two-relaxation rate method. This is not the entropic method
      yet, only the relaxation pattern. To get the entropic method, see parameters below!
      (:func:`lbmpy.methods.create_trt_kbc`)

- ``relaxation_rates``: sequence of relaxation rates, number depends on selected method. If you specify more rates than
  method needs, the additional rates are ignored. For SRT and TRT models it is possible ot define a single
  ``relaxation_rate`` instead of a list, the second rate for TRT is then determined via magic number.
- ``compressible=False``: affects the selection of equilibrium moments. Both options approximate the *incompressible*
  Navier Stokes Equations. However when chosen as False, the approximation is better, the standard LBM derivation is
  compressible.
- ``equilibrium_order=2``: order in velocity, at which the equilibrium moment/cumulant approximation is
  truncated. Order 2 is sufficient to approximate Navier-Stokes
- ``force_model=None``: possible values: ``None``, ``'simple'``, ``'luo'``, ``'guo'`` ``'buick'``, or an instance of a
  class implementing the same methods as the classes in :mod:`lbmpy.forcemodels`. For details, see
  :mod:`lbmpy.forcemodels`
- ``force=(0,0,0)``: either constant force or a symbolic expression depending on field value
- ``maxwellian_moments=True``: way to compute equilibrium moments/cumulants, if False the standard
  discretized LBM equilibrium is used, otherwise the equilibrium moments are computed from the continuous Maxwellian.
  This makes only a difference if sparse stencils are used e.g. D2Q9 and D3Q27 are not affected, D319 and DQ15 are
- ``cumulant=False``: use cumulants instead of moments
- ``initial_velocity=None``: initial velocity in domain, can either be a tuple (x,y,z) velocity to set a constant
  velocity everywhere, or a numpy array with the same size of the domain, with a last coordinate of shape dim to set
  velocities on cell level
- ``output={}``: a dictionary mapping macroscopic quantites e.g. the strings 'density' and 'velocity' to pystencils
  fields. In each timestep the corresponding quantities are written to the given fields.
- ``velocity_input``: symbolic field where the velocities are read from (for advection diffusion LBM)
- ``density_input``: symbolic field or field access where to read density from. When passing this parameter,
  ``velocity_input`` has to be passed as well
- ``kernel_type``: supported values: 'stream_pull_collide' (default), 'collide_only'


Entropic methods:

- ``entropic=False``: In case there are two distinct relaxation rate in a method, one of them (usually the one, not
  determining the viscosity) can be automatically chosen w.r.t an entropy condition. For details see
  :mod:`lbmpy.methods.entropic`
- ``entropic_newton_iterations=None``: For moment methods the entropy optimum can be calculated in closed form.
  For cumulant methods this is not possible, in that case it is computed using Newton iterations. This parameter can be
  used to force Newton iterations and specify how many should be done
- ``omega_output_field=None``: you can pass a pystencils Field here, where the calculated free relaxation rate of
  an entropic or Smagorinsky method is written to

LES methods:

- ``smagorinsky=False``: set to Smagorinsky constant to activate turbulence model, ``omega_output_field`` can be set to
  write out adapted relaxation rates

Fluctuating LB:

- ``fluctuating``: enables fluctuating lattice Boltzmann by randomizing collision process.
  Pass dictionary with parameters to  ``lbmpy.fluctuatinglb.add_fluctuations_to_collision_rule``


Optimization Parameters
^^^^^^^^^^^^^^^^^^^^^^^

Simplifications / Transformations:

- ``cse_pdfs=False``: run common subexpression elimination for opposing stencil directions
- ``cse_global=False``: run common subexpression elimination after all other simplifications have been executed
- ``split=False``: split innermost loop, to handle only 2 directions per loop. This reduces the number of parallel
  load/store streams and thus speeds up the kernel on most architectures
- ``builtin_periodicity=(False,False,False)``: instead of handling periodicity by copying ghost layers, the periodicity
  is built into the kernel. This parameters specifies if the domain is periodic in (x,y,z) direction. Even if the
  periodicity is built into the kernel, the fields have one ghost layer to be consistent with other functions. 
    

Field size information:

- ``pdf_arr=None``: pass a numpy array here to create kernels with fixed size and create the loop nest according
  to layout of this array
- ``field_size=None``: create kernel for fixed field size
- ``field_layout='c'``:   ``'c'`` or ``'numpy'`` for standard numpy layout, ``'reverse_numpy'`` or ``'f'`` for fortran
  layout, this does not apply when pdf_arr was given, then the same layout as pdf_arr is used
- ``symbolic_field``: pystencils field for source (pdf field that is read)
- ``symbolic_temporary_field``: pystencils field for temporary (pdf field that is written in stream, or stream-collide)


CPU:

- ``openmp=True``: Can be a boolean to turn multi threading on/off, or an integer
  specifying the number of threads. If True is specified OpenMP chooses the number of threads
- ``vectorization=False``: controls manual vectorization using SIMD instrinsics. If True default vectorization settings
  are use. Alternatively a dictionary with parameters for vectorize function can be passed. For example
  ``{'instruction_set': 'avx', 'assume_aligned': True, 'nontemporal': True}``. Nontemporal stores are only used if
  assume_aligned is also activated.


GPU:

- ``target='cpu'``: ``'cpu'`` or ``'gpu'``, last option requires a CUDA enabled graphics card
  and installed *pycuda* package
- ``gpu_indexing='block'``: determines mapping of CUDA threads to cells. Can be either ``'block'`` or ``'line'``
- ``gpu_indexing_params='block'``: parameters passed to init function of gpu indexing.
  For ``'block'`` indexing one can e.g. specify the block size ``{'block_size' : (128, 4, 1)}``


Other:

- ``openmp=True``: only applicable for cpu simulations. Can be a boolean to turn multi threading on/off, or an integer
  specifying the number of threads. If True is specified OpenMP chooses the number of threads
- ``double_precision=True``:  by default simulations run with double precision floating point numbers, by setting this
  parameter to False, single precision is used, which is much faster, especially on GPUs




Terminology and creation pipeline
---------------------------------

Kernel functions are created in three steps:

1. *Method*:
         the method defines the collision process. Currently there are two big categories:
         moment and cumulant based methods. A method defines how each moment or cumulant is relaxed by
         storing the equilibrium value and the relaxation rate for each moment/cumulant.
2. *Collision/Update Rule*:
         Methods can generate a "collision rule" which is an equation collection that define the
         post collision values as a function of the pre-collision values. On these equation collection
         simplifications are applied to reduce the number of floating point operations.
         At this stage an entropic optimization step can also be added to determine one relaxation rate by an
         entropy condition.
         Then a streaming rule is added which transforms the collision rule into an update rule.
         The streaming step depends on the pdf storage (source/destination, AABB pattern, EsoTwist).
         Currently only the simple source/destination  pattern is supported.
3. *AST*:
        The abstract syntax tree describes the structure of the kernel, including loops and conditionals.
        The ast can be modified e.g. to add OpenMP pragmas, reorder loops or apply other optimizations.
4. *Function*:
        This step compiles the AST into an executable function, either for CPU or GPUs. This function
        behaves like a normal Python function and runs one LBM time step.

The function :func:`create_lb_function` runs the whole pipeline, the other functions in this module
execute this pipeline only up to a certain step. Each function optionally also takes the result of the previous step.

For example, to modify the AST one can run::

    ast = create_lb_ast(...)
    # modify ast here
    func = create_lb_function(ast=ast, ...)

"""
from copy import copy

import sympy as sp

import lbmpy.forcemodels as forcemodels
from lbmpy.fieldaccess import (
    AAEvenTimeStepAccessor, AAOddTimeStepAccessor, CollideOnlyInplaceAccessor,
    EsoTwistEvenTimeStepAccessor, EsoTwistOddTimeStepAccessor, PdfFieldAccessor,
    PeriodicTwoFieldsAccessor, StreamPullTwoFieldsAccessor, StreamPushTwoFieldsAccessor)
from lbmpy.fluctuatinglb import add_fluctuations_to_collision_rule
from lbmpy.methods import (
    create_mrt3, create_mrt_orthogonal, create_mrt_raw, create_srt, create_trt, create_trt_kbc)
from lbmpy.methods.creationfunctions import create_generic_mrt
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod
from lbmpy.methods.entropic import add_entropy_condition, add_iterative_entropy_condition
from lbmpy.methods.entropic_eq_srt import create_srt_entropic
from lbmpy.moments import get_order
from lbmpy.relaxationrates import relaxation_rate_from_magic_number
from lbmpy.simplificationfactory import create_simplification_strategy
from lbmpy.stencils import get_stencil
from lbmpy.turbulence_models import add_smagorinsky_model
from lbmpy.updatekernels import create_lbm_kernel, create_stream_pull_with_output_kernel
from pystencils import Assignment, AssignmentCollection, create_kernel
from pystencils.cache import disk_cache_no_fallback
from pystencils.data_types import collate_types
from pystencils.field import Field, get_layout_of_array
from pystencils.simp import sympy_cse
from pystencils.stencil import have_same_entries


def create_lb_function(ast=None, optimization={}, **kwargs):
    """Creates a Python function for the LB method"""
    params, opt_params = update_with_default_parameters(kwargs, optimization)

    if ast is None:
        params['optimization'] = opt_params
        ast = create_lb_ast(**params)

    res = ast.compile()

    res.method = ast.method
    res.update_rule = ast.update_rule
    res.ast = ast
    return res


def create_lb_ast(update_rule=None, optimization={}, **kwargs):
    """Creates a pystencils AST for the LB method"""
    params, opt_params = update_with_default_parameters(kwargs, optimization)

    if update_rule is None:
        params['optimization'] = optimization
        update_rule = create_lb_update_rule(**params)

    field_types = set(fa.field.dtype for fa in update_rule.defined_symbols if isinstance(fa, Field.Access))
    res = create_kernel(update_rule, target=opt_params['target'], data_type=collate_types(field_types),
                        cpu_openmp=opt_params['openmp'], cpu_vectorize_info=opt_params['vectorization'],
                        gpu_indexing=opt_params['gpu_indexing'], gpu_indexing_params=opt_params['gpu_indexing_params'],
                        ghost_layers=1)

    res.method = update_rule.method
    res.update_rule = update_rule
    return res


@disk_cache_no_fallback
def create_lb_update_rule(collision_rule=None, optimization={}, **kwargs):
    """Creates an update rule (list of Assignments) for a LB method that describe a full sweep"""
    params, opt_params = update_with_default_parameters(kwargs, optimization)

    if collision_rule is None:
        collision_rule = create_lb_collision_rule(**params, optimization=opt_params)

    lb_method = collision_rule.method

    field_data_type = 'float64' if opt_params['double_precision'] else 'float32'
    q = len(collision_rule.method.stencil)

    if opt_params['symbolic_field'] is not None:
        src_field = opt_params['symbolic_field']
    elif opt_params['field_size']:
        field_size = [s + 2 for s in opt_params['field_size']] + [q]
        src_field = Field.create_fixed_size(params['field_name'], field_size, index_dimensions=1,
                                            layout=opt_params['field_layout'], dtype=field_data_type)
    else:
        src_field = Field.create_generic(params['field_name'], spatial_dimensions=collision_rule.method.dim,
                                         index_shape=(q,), layout=opt_params['field_layout'], dtype=field_data_type)

    if opt_params['symbolic_temporary_field'] is not None:
        dst_field = opt_params['symbolic_temporary_field']
    else:
        dst_field = src_field.new_field_with_different_name(params['temporary_field_name'])

    kernel_type = params['kernel_type']
    if isinstance(kernel_type, PdfFieldAccessor):
        accessor = kernel_type
        return create_lbm_kernel(collision_rule, src_field, dst_field, accessor)
    elif params['kernel_type'] == 'stream_pull_collide':
        accessor = StreamPullTwoFieldsAccessor
        if any(opt_params['builtin_periodicity']):
            accessor = PeriodicTwoFieldsAccessor(opt_params['builtin_periodicity'], ghost_layers=1)
        return create_lbm_kernel(collision_rule, src_field, dst_field, accessor)
    elif params['kernel_type'] == 'stream_pull_only':
        return create_stream_pull_with_output_kernel(lb_method, src_field, dst_field, params['output'])
    else:
        kernel_type_to_accessor = {
            'collide_only': CollideOnlyInplaceAccessor,
            'collide_stream_push': StreamPushTwoFieldsAccessor,
            'esotwist_even': EsoTwistEvenTimeStepAccessor,
            'esotwist_odd': EsoTwistOddTimeStepAccessor,
            'aa_even': AAEvenTimeStepAccessor,
            'aa_odd': AAOddTimeStepAccessor,
        }
        try:
            accessor = kernel_type_to_accessor[kernel_type]()
        except KeyError:
            raise ValueError("Invalid value of parameter 'kernel_type'", params['kernel_type'])
        return create_lbm_kernel(collision_rule, src_field, dst_field, accessor)


@disk_cache_no_fallback
def create_lb_collision_rule(lb_method=None, optimization={}, **kwargs):
    """Creates a collision rule (list of Assignments) for a LB method describing the collision operator (no stream)"""
    params, opt_params = update_with_default_parameters(kwargs, optimization)

    if lb_method is None:
        lb_method = create_lb_method(**params)

    split_inner_loop = 'split' in opt_params and opt_params['split']
    cqc = lb_method.conserved_quantity_computation

    rho_in = params['density_input']
    u_in = params['velocity_input']

    if u_in is not None and isinstance(u_in, Field):
        u_in = u_in.center_vector
    if rho_in is not None and isinstance(rho_in, Field):
        rho_in = rho_in.center

    keep_rrs_symbolic = opt_params['keep_rrs_symbolic']
    if u_in is not None:
        density_rhs = sum(lb_method.pre_collision_pdf_symbols) if rho_in is None else rho_in
        eqs = [Assignment(cqc.zeroth_order_moment_symbol, density_rhs)]
        eqs += [Assignment(u_sym, u_in[i]) for i, u_sym in enumerate(cqc.first_order_moment_symbols)]
        eqs = AssignmentCollection(eqs, [])
        collision_rule = lb_method.get_collision_rule(conserved_quantity_equations=eqs,
                                                      keep_rrs_symbolic=keep_rrs_symbolic)
    elif u_in is None and rho_in is not None:
        raise ValueError("When setting 'density_input' parameter, 'velocity_input' has to be specified as well.")
    else:
        collision_rule = lb_method.get_collision_rule(keep_rrs_symbolic=keep_rrs_symbolic)

    if opt_params['simplification'] == 'auto':
        simplification = create_simplification_strategy(lb_method, split_inner_loop=split_inner_loop)
    else:
        simplification = opt_params['simplification']
    collision_rule = simplification(collision_rule)

    if params['fluctuating']:
        add_fluctuations_to_collision_rule(collision_rule, **params['fluctuating'])

    if params['entropic']:
        if params['smagorinsky']:
            raise ValueError("Choose either entropic or smagorinsky")
        if params['entropic_newton_iterations']:
            if isinstance(params['entropic_newton_iterations'], bool):
                iterations = 3
            else:
                iterations = params['entropic_newton_iterations']
            collision_rule = add_iterative_entropy_condition(collision_rule, newton_iterations=iterations,
                                                             omega_output_field=params['omega_output_field'])
        else:
            collision_rule = add_entropy_condition(collision_rule, omega_output_field=params['omega_output_field'])
    elif params['smagorinsky']:
        smagorinsky_constant = 0.12 if params['smagorinsky'] is True else params['smagorinsky']
        collision_rule = add_smagorinsky_model(collision_rule, smagorinsky_constant,
                                               omega_output_field=params['omega_output_field'])
        if 'split_groups' in collision_rule.simplification_hints:
            collision_rule.simplification_hints['split_groups'][0].append(sp.Symbol("smagorinsky_omega"))

    cse_pdfs = False if 'cse_pdfs' not in opt_params else opt_params['cse_pdfs']
    cse_global = False if 'cse_global' not in opt_params else opt_params['cse_global']
    if cse_pdfs:
        from lbmpy.methods.momentbasedsimplifications import cse_in_opposing_directions
        collision_rule = cse_in_opposing_directions(collision_rule)
    if cse_global:
        collision_rule = sympy_cse(collision_rule)

    if params['output'] and params['kernel_type'] == 'stream_pull_collide':
        cqc = lb_method.conserved_quantity_computation
        output_eqs = cqc.output_equations_from_pdfs(lb_method.pre_collision_pdf_symbols, params['output'])
        collision_rule = collision_rule.new_merged(output_eqs)

    return collision_rule


def create_lb_method(**params):
    """Creates a LB method, defined by moments/cumulants for collision space, equilibrium and relaxation rates."""
    params, _ = update_with_default_parameters(params, {}, fail_on_unknown_parameter=False)

    if isinstance(params['stencil'], tuple) or isinstance(params['stencil'], list):
        stencil_entries = params['stencil']
    else:
        stencil_entries = get_stencil(params['stencil'])

    dim = len(stencil_entries[0])

    if isinstance(params['force'], Field):
        params['force'] = tuple(params['force'](i) for i in range(dim))

    force_is_zero = True
    for f_i in params['force']:
        if f_i != 0:
            force_is_zero = False

    no_force_model = 'force_model' not in params or params['force_model'] == 'none' or params['force_model'] is None
    if not force_is_zero and no_force_model:
        params['force_model'] = 'guo'

    if 'force_model' in params:
        force_model = force_model_from_string(params['force_model'], params['force'][:dim])
    else:
        force_model = None

    common_params = {
        'compressible': params['compressible'],
        'equilibrium_order': params['equilibrium_order'],
        'force_model': force_model,
        'maxwellian_moments': params['maxwellian_moments'],
        'cumulant': params['cumulant'],
        'c_s_sq': params['c_s_sq'],
    }
    method_name = params['method']
    relaxation_rates = params['relaxation_rates']

    if method_name.lower() == 'srt':
        assert len(relaxation_rates) >= 1, "Not enough relaxation rates"
        method = create_srt(stencil_entries, relaxation_rates[0], **common_params)
    elif method_name.lower() == 'trt':
        assert len(relaxation_rates) >= 2, "Not enough relaxation rates"
        method = create_trt(stencil_entries, relaxation_rates[0], relaxation_rates[1], **common_params)
    elif method_name.lower() == 'mrt':
        next_relaxation_rate = [0]

        def relaxation_rate_getter(moments):
            try:
                if all(get_order(m) < 2 for m in moments):
                    return 0
                res = relaxation_rates[next_relaxation_rate[0]]
                next_relaxation_rate[0] += 1
            except IndexError:
                raise ValueError("Too few relaxation rates specified")
            return res
        weighted = params['weighted'] if 'weighted' in params else True
        nested_moments = params['nested_moments'] if 'nested_moments' in params else None
        method = create_mrt_orthogonal(stencil_entries, relaxation_rate_getter, weighted=weighted,
                                       nested_moments=nested_moments, **common_params)
    elif method_name.lower() == 'mrt_raw':
        method = create_mrt_raw(stencil_entries, relaxation_rates, **common_params)
    elif method_name.lower() == 'mrt3':
        method = create_mrt3(stencil_entries, relaxation_rates, **common_params)
    elif method_name.lower().startswith('trt-kbc-n'):
        if have_same_entries(stencil_entries, get_stencil("D2Q9")):
            dim = 2
        elif have_same_entries(stencil_entries, get_stencil("D3Q27")):
            dim = 3
        else:
            raise NotImplementedError("KBC type TRT methods can only be constructed for D2Q9 and D3Q27 stencils")
        method_nr = method_name[-1]
        method = create_trt_kbc(dim, relaxation_rates[0], relaxation_rates[1], 'KBC-N' + method_nr, **common_params)
    elif method_name.lower() == 'entropic-srt':
        method = create_srt_entropic(stencil_entries, relaxation_rates[0], force_model, params['compressible'])
    else:
        raise ValueError("Unknown method %s" % (method_name,))

    return method


def create_lb_method_from_existing(method, modification_function):
    """Creates a new method based on an existing method by modifying its collision table.

    Args:
        method: old method
        modification_function: function receiving (moment, equilibrium_value, relaxation_rate) as arguments,
                               i.e. one row of the relaxation table, returning a modified version
    """
    relaxation_table = (modification_function(m, eq, rr)
                        for m, eq, rr in zip(method.moments, method.moment_equilibrium_values, method.relaxation_rates))
    compressible = method.conserved_quantity_computation.compressible
    cumulant = isinstance(method, CumulantBasedLbMethod)
    return create_generic_mrt(method.stencil, relaxation_table, compressible, method.force_model, cumulant)

# ----------------------------------------------------------------------------------------------------------------------


def force_model_from_string(force_model_name, force_values):
    if type(force_model_name) is not str:
        return force_model_name
    if force_model_name == 'none':
        return None

    force_model_dict = {
        'simple': forcemodels.Simple,
        'luo': forcemodels.Luo,
        'guo': forcemodels.Guo,
        'buick': forcemodels.Buick,
        'silva': forcemodels.Buick,
        'edm': forcemodels.EDM,
    }
    if force_model_name.lower() not in force_model_dict:
        raise ValueError("Unknown force model %s" % (force_model_name,))

    force_model_class = force_model_dict[force_model_name.lower()]
    return force_model_class(force_values)


def switch_to_symbolic_relaxation_rates_for_omega_adapting_methods(method_parameters, kernel_params, force=False):
    """
    For entropic kernels the relaxation rate has to be a variable. If a constant was passed a
    new dummy variable is inserted and the value of this variable is later on passed to the kernel
    """
    if method_parameters['entropic'] or method_parameters['smagorinsky'] or force:
        value_to_symbol_map = {}
        new_relaxation_rates = []
        for rr in method_parameters['relaxation_rates']:
            if not isinstance(rr, sp.Symbol):
                if rr not in value_to_symbol_map:
                    value_to_symbol_map[rr] = sp.Dummy()
                dummy_var = value_to_symbol_map[rr]
                new_relaxation_rates.append(dummy_var)
                kernel_params[dummy_var.name] = rr
            else:
                new_relaxation_rates.append(rr)
        if len(new_relaxation_rates) < 2:
            new_relaxation_rates.append(sp.Dummy())
        method_parameters['relaxation_rates'] = new_relaxation_rates


def update_with_default_parameters(params, opt_params=None, fail_on_unknown_parameter=True):
    opt_params = opt_params if opt_params is not None else {}

    default_method_description = {
        'stencil': 'D2Q9',
        'method': 'srt',  # can be srt, trt or mrt
        'relaxation_rates': None,
        'compressible': False,
        'equilibrium_order': 2,
        'c_s_sq': sp.Rational(1, 3),
        'weighted': True,
        'nested_moments': None,

        'force_model': 'none',
        'force': (0, 0, 0),
        'maxwellian_moments': True,
        'cumulant': False,
        'initial_velocity': None,

        'entropic': False,
        'entropic_newton_iterations': None,
        'omega_output_field': None,
        'smagorinsky': False,
        'fluctuating': False,
        'temperature': None,

        'output': {},
        'velocity_input': None,
        'density_input': None,

        'kernel_type': 'stream_pull_collide',

        'field_name': 'src',
        'temporary_field_name': 'dst',

        'lb_method': None,
        'collision_rule': None,
        'update_rule': None,
        'ast': None,
    }

    default_optimization_description = {
        'cse_pdfs': False,
        'cse_global': False,
        'simplification': 'auto',
        'keep_rrs_symbolic': True,
        'split': False,

        'field_size': None,
        'field_layout': 'fzyx',  # can be 'numpy' (='c'), 'reverse_numpy' (='f'), 'fzyx', 'zyxf'
        'symbolic_field': None,
        'symbolic_temporary_field': None,

        'target': 'cpu',
        'openmp': False,
        'double_precision': True,

        'gpu_indexing': 'block',
        'gpu_indexing_params': {},

        'vectorization': None,

        'builtin_periodicity': (False, False, False),
    }
    if 'relaxation_rate' in params:
        if 'relaxation_rates' not in params:
            if 'entropic' in params and params['entropic']:
                params['relaxation_rates'] = [params['relaxation_rate']]
            else:
                params['relaxation_rates'] = [params['relaxation_rate'],
                                              relaxation_rate_from_magic_number(params['relaxation_rate'])]

            del params['relaxation_rate']

    if 'pdf_arr' in opt_params:
        arr = opt_params['pdf_arr']
        opt_params['field_size'] = tuple(e - 2 for e in arr.shape[:-1])
        opt_params['field_layout'] = get_layout_of_array(arr)
        del opt_params['pdf_arr']

    if fail_on_unknown_parameter:
        unknown_params = [k for k in params.keys() if k not in default_method_description]
        unknown_opt_params = [k for k in opt_params.keys() if k not in default_optimization_description]
        if unknown_params:
            raise ValueError("Unknown parameter(s): " + ", ".join(unknown_params))
        if unknown_opt_params:
            raise ValueError("Unknown optimization parameter(s): " + ",".join(unknown_opt_params))

    params_result = copy(default_method_description)
    params_result.update(params)
    opt_params_result = copy(default_optimization_description)
    opt_params_result.update(opt_params)

    if params_result['relaxation_rates'] is None:
        stencil_param = params_result['stencil']
        if isinstance(stencil_param, tuple) or isinstance(stencil_param, list):
            stencil_entries = stencil_param
        else:
            stencil_entries = get_stencil(params_result['stencil'])
        params_result['relaxation_rates'] = sp.symbols("omega_:%d" % len(stencil_entries))

    return params_result, opt_params_result
