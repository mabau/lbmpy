r"""
Creating LBM kernels and Parameter Specifications
-------------------------------------------------

Kernel functions are created in four/five steps represented by five
python functions: `create_lb_method`, *create_lb_collision_rule/create_lb_update_rule*, `create_lb_ast` and
`create_lb_function` Each of those functions is configured with three data classes.

One dataclass defines the lattice Boltzmann method itself. This class is called `LBMConfig`. It defines, for example,
which collision space or LB stencil should be used.

The second one determines optimisations that are specific to the LBM. Optimisations like the
common subexpression elimination. Most of these optimisations act on the assignment level.
This means they only manipulate the assignments. The config class is called `LBMOptimisation`.

The third data class determines hardware optimisation. This means that contrary to the `LBMOptimisation` class,
it acts on the level of the abstract syntax tree. Thus, it is independent of the assignments and the LBM
and belongs to pystencils, not lbmpy. This can be found in the pystencils module as
'pystencils.kernelcreation.CreateKernelConfig'. With this class, for example, the target (CPU, GPU etc.)
of the generated code is specified.

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
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Union, List, Tuple, Any, Type, Iterable
from warnings import warn, filterwarnings

import lbmpy.moment_transforms
import pystencils.astnodes
import sympy as sp
import sympy.core.numbers

from lbmpy.enums import Stencil, Method, ForceModel
import lbmpy.forcemodels as forcemodels
import lbmpy.methods.centeredcumulant.force_model as cumulant_force_model
from lbmpy.fieldaccess import CollideOnlyInplaceAccessor, PdfFieldAccessor, PeriodicTwoFieldsAccessor
from lbmpy.fluctuatinglb import add_fluctuations_to_collision_rule
from lbmpy.methods import (create_mrt_orthogonal, create_mrt_raw, create_central_moment,
                           create_srt, create_trt, create_trt_kbc)
from lbmpy.methods.abstractlbmethod import RelaxationInfo
from lbmpy.methods.centeredcumulant import CenteredCumulantBasedLbMethod
from lbmpy.moment_transforms import PdfsToMomentsByChimeraTransform, PdfsToCentralMomentsByShiftMatrix
from lbmpy.methods.centeredcumulant.cumulant_transform import CentralMomentsToCumulantsByGeneratingFunc
from lbmpy.methods.creationfunctions import (
    create_with_monomial_cumulants, create_with_polynomial_cumulants, create_with_default_polynomial_cumulants)
from lbmpy.methods.creationfunctions import create_generic_mrt
from lbmpy.methods.momentbased.entropic import add_entropy_condition, add_iterative_entropy_condition
from lbmpy.methods.momentbased.entropic_eq_srt import create_srt_entropic
from lbmpy.relaxationrates import relaxation_rate_from_magic_number
from lbmpy.simplificationfactory import create_simplification_strategy
from lbmpy.stencils import LBStencil
from lbmpy.turbulence_models import add_smagorinsky_model
from lbmpy.updatekernels import create_lbm_kernel, create_stream_pull_with_output_kernel
from lbmpy.advanced_streaming.utility import Timestep, get_accessor
from pystencils import Assignment, AssignmentCollection, create_kernel, CreateKernelConfig
from pystencils.cache import disk_cache_no_fallback
from pystencils.data_types import collate_types
from pystencils.field import Field
from pystencils.simp import sympy_cse, SimplificationStrategy
# needed for the docstring
from lbmpy.methods.abstractlbmethod import LbmCollisionRule, AbstractLbMethod

# Filter out JobLib warnings. They are not usefull for use:
# https://github.com/joblib/joblib/issues/683
filterwarnings("ignore", message="Persisting input arguments took")


@dataclass
class LBMConfig:
    """
    **Below all parameters for the LBMConfig are explained**
    """
    stencil: lbmpy.stencils.LBStencil = LBStencil(Stencil.D2Q9)
    """
    All stencils are defined in :class:`lbmpy.enums.Stencil`. From that :class:`lbmpy.stencils.LBStenil` 
    class will be created
    """
    method: Method = Method.SRT
    """
    Name of lattice Boltzmann method. Defined by :class:`lbmpy.enums.Method`.
    This determines the selection and relaxation pattern of moments/cumulants, i.e. which moment/cumulant basis is
    chosen, and which of the basis vectors are relaxed together
    """
    relaxation_rates: Iterable = None
    """
    Sequence of relaxation rates, number depends on selected method. If you specify more rates than
    method needs, the additional rates are ignored.
    """
    relaxation_rate: Union[int, float, Type[sp.Symbol]] = None
    """
    For SRT, TRT and polynomial cumulant models it is possible to define
    a single ``relaxation_rate`` instead of a list (Internally this is converted to a list with a single entry).
    The second rate for TRT is then determined via magic number. For the moment, central moment based and the
    cumulant model, it sets only the relaxation rate corresponding to shear viscosity, setting all others to unity.
    """
    compressible: bool = False
    """
    Affects the selection of equilibrium moments. Both options approximate the *incompressible*
    Navier Stokes Equations. However when chosen as False, the approximation is better, the standard LBM derivation is
    compressible.
    """
    equilibrium_order: int = 2
    """
    Order in velocity, at which the equilibrium moment/cumulant approximation is
    truncated. Order 2 is sufficient to approximate Navier-Stokes. Note cumulant methods are by 
    definition at least order 4.
    """
    c_s_sq: sympy.Rational = sp.Rational(1, 3)
    """
    The squared lattice speed of sound used to derive the LB method. It is very uncommon to use a value different 
    to 1 / 3.
    """
    weighted: bool = True
    """
    Affects only orthogonal MRT methods. If set to True a weighted Gram-Schmidt procedure is used to orthogonalise 
    the moments.
    """
    nested_moments: List[List] = None
    """
    A list of lists of modes, grouped by common relaxation times. This is usually used in
    conjunction with `lbmpy.methods.default_moment_sets.mrt_orthogonal_modes_literature`.
    If this argument is not provided, Gram-Schmidt orthogonalization of the default modes is performed.
    """

    force_model: Union[Type[lbmpy.forcemodels.AbstractForceModel], ForceModel] = None
    """
    Force model to determine how forcing terms enter the collision rule.
    Possibilities are defined in :class: `lbmpy.enums.ForceModel`
    """
    force: Union[Tuple, Field] = (0, 0, 0)
    """
    Either constant force or a symbolic expression depending on field value
    """
    maxwellian_moments: bool = True
    """
    Way to compute equilibrium moments/cumulants, if False the standard discretised LBM equilibrium is used,
    otherwise the equilibrium moments are computed from the continuous Maxwellian. This makes only a 
    difference if sparse stencils are used e.g. D2Q9 and D3Q27 are not affected, D319 and DQ15 are affected.
    """
    initial_velocity: Tuple = None,
    """
    Initial velocity in domain, can either be a tuple (x,y,z) velocity to set a constant
    velocity everywhere, or a numpy array with the same size of the domain, with a last coordinate of shape dim to set
    velocities on cell level
    """
    galilean_correction: bool = False
    """
    Special correction for D3Q27 cumulant LBMs. For Details see
    :mod:`lbmpy.methods.centeredcumulant.galilean_correction`
    """
    moment_transform_class: Type[lbmpy.moment_transforms.AbstractMomentTransform] = PdfsToMomentsByChimeraTransform
    """
    Python class that determines how PDFs are transformed to the moment space. Usually, the chimera transform is 
    the best choice (see :cite:`geier2015`). However, for the SRT and TRT methods it defaults to `None`, since 
    no transformation is necessary and the collision can happen in PDF space.
    """
    central_moment_transform_class: Type[lbmpy.moment_transforms.AbstractMomentTransform] = \
        PdfsToCentralMomentsByShiftMatrix
    """
    Python class that determines how PDFs are transformed to the central moment space.
    Usually, a transformation based on the shift matrix is the best choice.
    """
    cumulant_transform_class: Type[lbmpy.moment_transforms.AbstractMomentTransform] = \
        CentralMomentsToCumulantsByGeneratingFunc
    """
    Python class that determines how PDFs are transformed from the central moment space to the cumulant space.
    """

    entropic: bool = False
    """
    In case there are two distinct relaxation rate in a method, one of them (usually the one, not
    determining the viscosity) can be automatically chosen w.r.t an entropy condition. For details see
    :mod:`lbmpy.methods.momentbased.entropic`
    """
    entropic_newton_iterations: int = None
    """
    For moment methods the entropy optimum can be calculated in closed form.
    For cumulant methods this is not possible, in that case it is computed using Newton iterations.
    This parameter can be used to force Newton iterations and specify how many should be done
    """
    omega_output_field: Field = None
    """
    A pystencils Field can be passed here, where the calculated free relaxation rate of
    an entropic or Smagorinsky method is written to
    """
    smagorinsky: Union[float, bool] = False
    """
    set to Smagorinsky constant to activate turbulence model, ``omega_output_field`` can be set to
    write out adapted relaxation rates. If set to `True`, 0.12 is used as default smagorinsky constant.
    """
    fluctuating: dict = False
    """
    Enables fluctuating lattice Boltzmann by randomizing collision process.
    Pass dictionary with parameters to  ``lbmpy.fluctuatinglb.add_fluctuations_to_collision_rule``.
    Can only be used for weighed MRT collision operators.
    """
    temperature: Any = None
    """
    Temperature for fluctuating lattice Boltzmann methods.
    """

    output: dict = field(default_factory=dict)
    """
    A dictionary mapping macroscopic quantites e.g. the strings 'density' and 'velocity' to pystencils
    fields. In each timestep the corresponding quantities are written to the given fields. Possible input would be:
    {'density': density_field, 'velocity': velocity_field}
    """
    velocity_input: Field = None
    """
    Symbolic field where the velocities are read from. If `None` is given the velocity is calculated inplace from
    with first order moments.
    """
    density_input: Field = None
    """
    Symbolic field where the density is read from. If `None` is given the density is calculated inplace from
    with zeroth order moment.
    """

    kernel_type: Union[str, Type[PdfFieldAccessor]] = 'default_stream_collide'
    """
    Supported values: ``'default_stream_collide'`` (default), ``'collide_only'``, ``'stream_pull_only'``. 
    With ``'default_stream_collide'``, streaming pattern and even/odd time-step (for in-place patterns) can be specified
    by the ``streaming_pattern`` and ``timestep`` arguments. For backwards compatibility, ``kernel_type`` also accepts
    ``'stream_pull_collide'``, ``'collide_stream_push'``, ``'esotwist_even'``, ``'esotwist_odd'``, ``'aa_even'``
    and ``'aa_odd'`` for selection of the streaming pattern. 
    """
    streaming_pattern: str = 'pull'
    """
    The streaming pattern to be used with a ``'default_stream_collide'`` kernel. Accepted values are
    ``'pull'``, ``'push'``, ``'aa'`` and ``'esotwist'``.
    """
    timestep: Timestep = Timestep.BOTH
    """
    Timestep modulus for the streaming pattern. For two-fields patterns, this argument is irrelevant and
    by default set to ``Timestep.BOTH``. For in-place patterns, ``Timestep.EVEN`` or ``Timestep.ODD`` must be specified.
    """

    field_name: str = 'src'
    """
    Name of the PDF field.
    """
    temporary_field_name: str = 'dst'
    """
    Name of the temporary PDF field.
    """

    lb_method: Type[AbstractLbMethod] = None
    """
    Instance of `lbmpy.methods.abstractlbmethod.AbstractLbMethod`. If this parameter is `None`, the lb_method is derived
    via `create_lb_method`. 
    """
    collision_rule: LbmCollisionRule = None
    """
    Instance of :class:`lbmpy.methods.LbmCollisionRule`. If this parameter is `None`,
    the collision rule is derived via *create_lb_collision_rule*.
    """
    update_rule: LbmCollisionRule = None
    """
    Instance of :class:`lbmpy.methods.LbmCollisionRule`. If this parameter is `None`,
    the update rule is derived via *create_lb_update_rule*.
    """
    ast: pystencils.astnodes.KernelFunction = None
    """
    Instance of *pystencils.astnodes.KernelFunction*. If this parameter is `None`,
    the ast is derived via `create_lb_ast`.
    """

    def __post_init__(self):
        if isinstance(self.method, str):
            new_method = Method[self.method.upper()]
            warn(f'Method "{self.method}" as str is deprecated. Use {new_method} instead', category=DeprecationWarning)
            self.method = new_method

        if not isinstance(self.stencil, LBStencil):
            self.stencil = LBStencil(self.stencil)

        if self.relaxation_rates is None:
            self.relaxation_rates = [sp.Symbol("omega")] * self.stencil.Q

        # if only a single relaxation rate is defined (which makes sense for SRT or TRT methods)
        # it is internally treated as a list with one element and just sets the relaxation_rates parameter
        if self.relaxation_rate is not None:
            if self.method in [Method.TRT, Method.TRT_KBC_N1, Method.TRT_KBC_N2, Method.TRT_KBC_N3, Method.TRT_KBC_N4]:
                self.relaxation_rates = [self.relaxation_rate,
                                         relaxation_rate_from_magic_number(self.relaxation_rate)]
            else:
                self.relaxation_rates = [self.relaxation_rate]

        #   By default, do not derive moment equations for SRT and TRT methods
        if self.method == Method.SRT or self.method == Method.TRT:
            self.moment_transform_class = None

        #   Workaround until entropic method supports relaxation in subexpressions
        #   and the problem with RNGs in the assignment collection has been solved
        if self.entropic or self.fluctuating:
            self.moment_transform_class = None

        #   for backwards compatibility
        kernel_type_to_streaming_pattern = {
            'stream_pull_collide': ('pull', Timestep.BOTH),
            'collide_stream_push': ('push', Timestep.BOTH),
            'aa_even': ('aa', Timestep.EVEN),
            'aa_odd': ('aa', Timestep.ODD),
            'esotwist_even': ('esotwist', Timestep.EVEN),
            'esotwist_odd': ('esotwist', Timestep.ODD)
        }

        if self.kernel_type in kernel_type_to_streaming_pattern.keys():
            self.streaming_pattern, self.timestep = kernel_type_to_streaming_pattern[self.kernel_type]
            self.kernel_type = 'default_stream_collide'

        if isinstance(self.force, Field):
            self.force = tuple([self.force(i) for i in range(self.stencil.D)])

        force_not_zero = False
        for f_i in self.force:
            if f_i != 0:
                force_not_zero = True

        if self.force_model is None and force_not_zero:
            self.force_model = cumulant_force_model.CenteredCumulantForceModel(self.force[:self.stencil.D]) \
                if self.method == Method.CUMULANT else forcemodels.Guo(self.force[:self.stencil.D])

        force_model_dict = {
            'simple': forcemodels.Simple,
            'luo': forcemodels.Luo,
            'guo': forcemodels.Guo,
            'schiller': forcemodels.Guo,
            'buick': forcemodels.Buick,
            'silva': forcemodels.Buick,
            'edm': forcemodels.EDM,
            'kupershtokh': forcemodels.EDM,
            'cumulant': cumulant_force_model.CenteredCumulantForceModel,
            'he': forcemodels.He,
            'shanchen': forcemodels.ShanChen
        }

        if isinstance(self.force_model, str):
            new_force_model = ForceModel[self.force_model.upper()]
            warn(f'ForceModel "{self.force_model}" as str is deprecated. Use {new_force_model} instead or '
                 f'provide a class of type AbstractForceModel', category=DeprecationWarning)
            force_model_class = force_model_dict[new_force_model.name.lower()]
            self.force_model = force_model_class(force=self.force[:self.stencil.D])
        elif isinstance(self.force_model, ForceModel):
            force_model_class = force_model_dict[self.force_model.name.lower()]
            self.force_model = force_model_class(force=self.force[:self.stencil.D])


@dataclass
class LBMOptimisation:
    """
    **Below all parameters for the LBMOptimisation are explained**
    """
    cse_pdfs: bool = False
    """
    Run common subexpression elimination for opposing stencil directions.
    """
    cse_global: bool = False
    """
    Run common subexpression elimination after all other simplifications have been executed.
    """
    simplification: Union[str, bool] = 'auto'
    """
    Simplifications applied during the derivation of the collision rule. For details
    see :func:`lbmpy.simplificationfactory.create_simplification_strategy`
    """
    pre_simplification: bool = True
    """
    Simplifications applied during the derivation of the collision rule for cumulant LBMs.
    For details see :mod:`lbmpy.moment_transforms`.
    """
    split: bool = False
    """
    Split innermost loop, to handle only two directions per loop. This reduces the number of parallel
    load/store streams and thus speeds up the kernel on most architectures.
    """
    field_size: Any = None
    """
    Create kernel for fixed field size. 
    """
    field_layout: str = 'fzyx'
    """
    ``'c'`` or ``'numpy'`` for standard numpy layout, ``'reverse_numpy'`` or ``'f'`` for fortran
    layout, this does not apply when pdf_arr was given, then the same layout as pdf_arr is used.
    """
    symbolic_field: pystencils.field.Field = None
    """
    Pystencils field for source (pdf field that is read)
    """
    symbolic_temporary_field: pystencils.field.Field = None
    """
    Pystencils field for temporary (pdf field that is written in stream, or stream-collide)
    """
    builtin_periodicity: Tuple[bool] = (False, False, False)
    """
    Instead of handling periodicity by copying ghost layers, the periodicity
    is built into the kernel. This parameters specifies if the domain is periodic in (x,y,z) direction. Even if the
    periodicity is built into the kernel, the fields have one ghost layer to be consistent with other functions.
    """


def create_lb_function(ast=None, lbm_config=None, lbm_optimisation=None, config=None, optimization=None, **kwargs):
    """Creates a Python function for the LB method"""
    lbm_config, lbm_optimisation, config = update_with_default_parameters(kwargs, optimization,
                                                                          lbm_config, lbm_optimisation, config)
    if lbm_config.ast is not None:
        ast = lbm_config.ast

    if ast is None:
        ast = create_lb_ast(lbm_config.update_rule, lbm_config=lbm_config,
                            lbm_optimisation=lbm_optimisation, config=config)

    res = ast.compile()

    res.method = ast.method
    res.update_rule = ast.update_rule
    res.ast = ast
    return res


def create_lb_ast(update_rule=None, lbm_config=None, lbm_optimisation=None, config=None, optimization=None, **kwargs):
    """Creates a pystencils AST for the LB method"""
    lbm_config, lbm_optimisation, config = update_with_default_parameters(kwargs, optimization,
                                                                          lbm_config, lbm_optimisation, config)

    if lbm_config.update_rule is not None:
        update_rule = lbm_config.update_rule

    if update_rule is None:
        update_rule = create_lb_update_rule(lbm_config.collision_rule, lbm_config=lbm_config,
                                            lbm_optimisation=lbm_optimisation, config=config)

    field_types = set(fa.field.dtype for fa in update_rule.defined_symbols if isinstance(fa, Field.Access))

    config = replace(config, data_type=collate_types(field_types), ghost_layers=1)
    res = create_kernel(update_rule, config=config)

    res.method = update_rule.method
    res.update_rule = update_rule
    return res


@disk_cache_no_fallback
def create_lb_update_rule(collision_rule=None, lbm_config=None, lbm_optimisation=None, config=None,
                          optimization=None, **kwargs):
    """Creates an update rule (list of Assignments) for a LB method that describe a full sweep"""
    lbm_config, lbm_optimisation, config = update_with_default_parameters(kwargs, optimization,
                                                                          lbm_config, lbm_optimisation, config)

    if lbm_config.collision_rule is not None:
        collision_rule = lbm_config.collision_rule

    if collision_rule is None:
        collision_rule = create_lb_collision_rule(lbm_config.lb_method, lbm_config=lbm_config,
                                                  lbm_optimisation=lbm_optimisation,
                                                  config=config)

    lb_method = collision_rule.method

    field_data_type = config.data_type
    q = collision_rule.method.stencil.Q

    if lbm_optimisation.symbolic_field is not None:
        src_field = lbm_optimisation.symbolic_field
    elif lbm_optimisation.field_size:
        field_size = tuple([s + 2 for s in lbm_optimisation.field_size] + [q])
        src_field = Field.create_fixed_size(lbm_config.field_name, field_size, index_dimensions=1,
                                            layout=lbm_optimisation.field_layout, dtype=field_data_type)
    else:
        src_field = Field.create_generic(lbm_config.field_name, spatial_dimensions=collision_rule.method.dim,
                                         index_shape=(q,), layout=lbm_optimisation.field_layout, dtype=field_data_type)

    if lbm_optimisation.symbolic_temporary_field is not None:
        dst_field = lbm_optimisation.symbolic_temporary_field
    else:
        dst_field = src_field.new_field_with_different_name(lbm_config.temporary_field_name)

    kernel_type = lbm_config.kernel_type
    if kernel_type == 'stream_pull_only':
        return create_stream_pull_with_output_kernel(lb_method, src_field, dst_field, lbm_config.output)
    else:
        if kernel_type == 'default_stream_collide':
            if lbm_config.streaming_pattern == 'pull' and any(lbm_optimisation.builtin_periodicity):
                accessor = PeriodicTwoFieldsAccessor(lbm_optimisation.builtin_periodicity, ghost_layers=1)
            else:
                accessor = get_accessor(lbm_config.streaming_pattern, lbm_config.timestep)
        elif kernel_type == 'collide_only':
            accessor = CollideOnlyInplaceAccessor
        elif isinstance(kernel_type, PdfFieldAccessor):
            accessor = kernel_type
        else:
            raise ValueError("Invalid value of parameter 'kernel_type'", lbm_config.kernel_type)
        return create_lbm_kernel(collision_rule, src_field, dst_field, accessor)


@disk_cache_no_fallback
def create_lb_collision_rule(lb_method=None, lbm_config=None, lbm_optimisation=None, config=None,
                             optimization=None, **kwargs):
    """Creates a collision rule (list of Assignments) for a LB method describing the collision operator (no stream)"""
    lbm_config, lbm_optimisation, config = update_with_default_parameters(kwargs, optimization,
                                                                          lbm_config, lbm_optimisation, config)

    if lbm_config.lb_method is not None:
        lb_method = lbm_config.lb_method

    if lb_method is None:
        lb_method = create_lb_method(lbm_config)

    cqc = lb_method.conserved_quantity_computation

    rho_in = lbm_config.density_input
    u_in = lbm_config.velocity_input

    if u_in is not None and isinstance(u_in, Field):
        u_in = u_in.center_vector
    if rho_in is not None and isinstance(rho_in, Field):
        rho_in = rho_in.center

    pre_simplification = lbm_optimisation.pre_simplification
    if u_in is not None:
        density_rhs = sum(lb_method.pre_collision_pdf_symbols) if rho_in is None else rho_in
        eqs = [Assignment(cqc.zeroth_order_moment_symbol, density_rhs)]
        eqs += [Assignment(u_sym, u_in[i]) for i, u_sym in enumerate(cqc.first_order_moment_symbols)]
        eqs = AssignmentCollection(eqs, [])
        collision_rule = lb_method.get_collision_rule(conserved_quantity_equations=eqs,
                                                      pre_simplification=pre_simplification)

    elif u_in is None and rho_in is not None:
        raise ValueError("When setting 'density_input' parameter, 'velocity_input' has to be specified as well.")
    else:
        collision_rule = lb_method.get_collision_rule(pre_simplification=pre_simplification)

    if lbm_config.entropic:
        if lbm_config.smagorinsky:
            raise ValueError("Choose either entropic or smagorinsky")
        if lbm_config.entropic_newton_iterations:
            if isinstance(lbm_config.entropic_newton_iterations, bool):
                iterations = 3
            else:
                iterations = lbm_config.entropic_newton_iterations
            collision_rule = add_iterative_entropy_condition(collision_rule, newton_iterations=iterations,
                                                             omega_output_field=lbm_config.omega_output_field)
        else:
            collision_rule = add_entropy_condition(collision_rule, omega_output_field=lbm_config.omega_output_field)
    elif lbm_config.smagorinsky:
        smagorinsky_constant = 0.12 if lbm_config.smagorinsky is True else lbm_config.smagorinsky
        collision_rule = add_smagorinsky_model(collision_rule, smagorinsky_constant,
                                               omega_output_field=lbm_config.omega_output_field)
        if 'split_groups' in collision_rule.simplification_hints:
            collision_rule.simplification_hints['split_groups'][0].append(sp.Symbol("smagorinsky_omega"))

    if lbm_config.output:
        cqc = lb_method.conserved_quantity_computation
        output_eqs = cqc.output_equations_from_pdfs(lb_method.pre_collision_pdf_symbols, lbm_config.output)
        collision_rule = collision_rule.new_merged(output_eqs)

    if lbm_optimisation.simplification == 'auto':
        simplification = create_simplification_strategy(lb_method, split_inner_loop=lbm_optimisation.split)
    elif callable(lbm_optimisation.simplification):
        simplification = lbm_optimisation.simplification
    else:
        simplification = SimplificationStrategy()
    collision_rule = simplification(collision_rule)

    if lbm_config.fluctuating:
        add_fluctuations_to_collision_rule(collision_rule, **lbm_config.fluctuating)

    if lbm_optimisation.cse_pdfs:
        from lbmpy.methods.momentbased.momentbasedsimplifications import cse_in_opposing_directions
        collision_rule = cse_in_opposing_directions(collision_rule)
    if lbm_optimisation.cse_global:
        collision_rule = sympy_cse(collision_rule)

    return collision_rule


def create_lb_method(lbm_config=None, **params):
    """Creates a LB method, defined by moments/cumulants for collision space, equilibrium and relaxation rates."""
    lbm_config, _, _ = update_with_default_parameters(params, lbm_config=lbm_config)

    relaxation_rates = lbm_config.relaxation_rates
    dim = lbm_config.stencil.D

    if isinstance(lbm_config.force, Field):
        lbm_config.force = tuple(lbm_config.force(i) for i in range(dim))

    common_params = {
        'compressible': lbm_config.compressible,
        'equilibrium_order': lbm_config.equilibrium_order,
        'force_model': lbm_config.force_model,
        'maxwellian_moments': lbm_config.maxwellian_moments,
        'c_s_sq': lbm_config.c_s_sq,
        'moment_transform_class': lbm_config.moment_transform_class,
        'central_moment_transform_class': lbm_config.central_moment_transform_class,
    }

    cumulant_params = {
        'equilibrium_order': lbm_config.equilibrium_order,
        'force_model': lbm_config.force_model,
        'c_s_sq': lbm_config.c_s_sq,
        'galilean_correction': lbm_config.galilean_correction,
        'central_moment_transform_class': lbm_config.central_moment_transform_class,
        'cumulant_transform_class': lbm_config.cumulant_transform_class,
    }

    if lbm_config.method == Method.SRT:
        assert len(relaxation_rates) >= 1, "Not enough relaxation rates"
        method = create_srt(lbm_config.stencil, relaxation_rates[0], **common_params)
    elif lbm_config.method == Method.TRT:
        assert len(relaxation_rates) >= 2, "Not enough relaxation rates"
        method = create_trt(lbm_config.stencil, relaxation_rates[0], relaxation_rates[1], **common_params)
    elif lbm_config.method == Method.MRT:
        method = create_mrt_orthogonal(lbm_config.stencil, relaxation_rates, weighted=lbm_config.weighted,
                                       nested_moments=lbm_config.nested_moments, **common_params)
    elif lbm_config.method == Method.CENTRAL_MOMENT:
        method = create_central_moment(lbm_config.stencil, relaxation_rates,
                                       nested_moments=lbm_config.nested_moments, **common_params)
    elif lbm_config.method == Method.MRT_RAW:
        method = create_mrt_raw(lbm_config.stencil, relaxation_rates, **common_params)
    elif lbm_config.method in [Method.TRT_KBC_N1, Method.TRT_KBC_N2, Method.TRT_KBC_N3, Method.TRT_KBC_N4]:
        if lbm_config.stencil.D == 2 and lbm_config.stencil.Q == 9:
            dim = 2
        elif lbm_config.stencil.D == 3 and lbm_config.stencil.Q == 27:
            dim = 3
        else:
            raise NotImplementedError("KBC type TRT methods can only be constructed for D2Q9 and D3Q27 stencils")
        method_nr = lbm_config.method.name[-1]
        method = create_trt_kbc(dim, relaxation_rates[0], relaxation_rates[1], 'KBC-N' + method_nr, **common_params)
    elif lbm_config.method == Method.ENTROPIC_SRT:
        method = create_srt_entropic(lbm_config.stencil, relaxation_rates[0], lbm_config.force_model,
                                     lbm_config.compressible)
    elif lbm_config.method == Method.CUMULANT:
        if lbm_config.nested_moments is not None:
            method = create_with_polynomial_cumulants(
                lbm_config.stencil, relaxation_rates, lbm_config.nested_moments, **cumulant_params)
        else:
            method = create_with_default_polynomial_cumulants(lbm_config.stencil, relaxation_rates, **cumulant_params)
    elif lbm_config.method == Method.MONOMIAL_CUMULANT:
        method = create_with_monomial_cumulants(lbm_config.stencil, relaxation_rates, **cumulant_params)
    else:
        raise ValueError("Failed to create LB method. Please use lbmpy.enums.Method for the creation")

    # >>Entropic methods can only be created for methods with two relaxation rates One free relaxation rate
    # determining the viscosity and one to be determined by the entropy condition<<
    # Thus we fix the conserved quantities to one of the relaxation rates because zero would be recognised as
    # a third relaxation rate here.
    if lbm_config.entropic:
        method.set_conserved_moments_relaxation_rate(relaxation_rates[0])

    return method


def create_lb_method_from_existing(method, modification_function):
    """Creates a new method based on an existing method by modifying its collision table.

    Args:
        method: old method
        modification_function: function receiving (moment, equilibrium_value, relaxation_rate) as arguments,
                               i.e. one row of the relaxation table, returning a modified version
    """
    compressible = method.conserved_quantity_computation.compressible
    if isinstance(method, CenteredCumulantBasedLbMethod):
        rr_dict = OrderedDict()
        relaxation_table = (modification_function(m, eq, rr)
                            for m, eq, rr in
                            zip(method.cumulants, method.cumulant_equilibrium_values, method.relaxation_rates))

        for cumulant, eq_value, rr in relaxation_table:
            cumulant = sp.sympify(cumulant)
            rr_dict[cumulant] = RelaxationInfo(eq_value, rr)

        return CenteredCumulantBasedLbMethod(method.stencil, rr_dict,
                                             method.conserved_quantity_computation,
                                             method.force_model,
                                             galilean_correction=method.galilean_correction,
                                             central_moment_transform_class=method.central_moment_transform_class,
                                             cumulant_transform_class=method.cumulant_transform_class)
    else:
        relaxation_table = (modification_function(m, eq, rr)
                            for m, eq, rr in
                            zip(method.moments, method.moment_equilibrium_values, method.relaxation_rates))
        return create_generic_mrt(method.stencil, relaxation_table, compressible, method.force_model)


# ----------------------------------------------------------------------------------------------------------------------
def update_with_default_parameters(params, opt_params=None, lbm_config=None, lbm_optimisation=None, config=None):
    # Fix CreateKernelConfig params
    pystencils_config_params = ['target', 'backend', 'cpu_openmp', 'double_precision', 'gpu_indexing',
                                'gpu_indexing_params', 'cpu_vectorize_info']
    if opt_params is not None:
        config_params = {k: v for k, v in opt_params.items() if k in pystencils_config_params}
    else:
        config_params = {}
    if 'double_precision' in config_params:
        if config_params['double_precision']:
            config_params['data_type'] = 'float64'
        else:
            config_params['data_type'] = 'float32'
        del config_params['double_precision']

    if not config:
        config = CreateKernelConfig(**config_params)
    else:
        for k, v in config_params.items():
            if not hasattr(config, k):
                raise KeyError(f'{v} is not a valid kwarg. Please look in CreateKernelConfig for valid settings')
        config = replace(config, **config_params)

    lbm_opt_params = ['cse_pdfs', 'cse_global', 'simplification', 'pre_simplification', 'split', 'field_size',
                      'field_layout', 'symbolic_field', 'symbolic_temporary_field', 'builtin_periodicity']

    if opt_params is not None:
        opt_params_dict = {k: v for k, v in opt_params.items() if k in lbm_opt_params}
    else:
        opt_params_dict = {}

    if not lbm_optimisation:
        lbm_optimisation = LBMOptimisation(**opt_params_dict)
    else:
        for k, v in opt_params_dict.items():
            if not hasattr(lbm_optimisation, k):
                raise KeyError(f'{v} is not a valid kwarg. Please look in LBMOptimisation for valid settings')
        lbm_optimisation = replace(lbm_optimisation, **opt_params_dict)

    if params is None:
        params = {}

    if not lbm_config:
        lbm_config = LBMConfig(**params)
    else:
        for k, v in params.items():
            if not hasattr(lbm_config, k):
                raise KeyError(f'{v} is not a valid kwarg. Please look in LBMConfig for valid settings')
        lbm_config = replace(lbm_config, **params)

    return lbm_config, lbm_optimisation, config
