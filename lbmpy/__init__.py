from .creationfunctions import create_lb_ast, create_lb_collision_rule, create_lb_function,\
    create_lb_method, create_lb_update_rule, LBMConfig, LBMOptimisation
from .enums import Stencil, Method, ForceModel, CollisionSpace
from .lbstep import LatticeBoltzmannStep
from .macroscopic_value_kernels import pdf_initialization_assignments, macroscopic_values_getter,\
    compile_macroscopic_values_getter, compile_macroscopic_values_setter, create_advanced_velocity_setter_collision_rule
from .maxwellian_equilibrium import get_weights
from .relaxationrates import relaxation_rate_from_lattice_viscosity, lattice_viscosity_from_relaxation_rate,\
    relaxation_rate_from_magic_number
from .scenarios import create_lid_driven_cavity, create_fully_periodic_flow
from .stencils import LBStencil


__all__ = ['create_lb_ast', 'create_lb_collision_rule', 'create_lb_function', 'create_lb_method',
           'create_lb_method_from_existing', 'create_lb_update_rule', 'LBMConfig', 'LBMOptimisation',
           'Stencil', 'Method', 'ForceModel', 'CollisionSpace',
           'LatticeBoltzmannStep',
           'pdf_initialization_assignments', 'macroscopic_values_getter', 'compile_macroscopic_values_getter',
           'compile_macroscopic_values_setter', 'create_advanced_velocity_setter_collision_rule',
           'get_weights',
           'relaxation_rate_from_lattice_viscosity', 'lattice_viscosity_from_relaxation_rate',
           'relaxation_rate_from_magic_number',
           'create_lid_driven_cavity', 'create_fully_periodic_flow',
           'LBStencil']


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
