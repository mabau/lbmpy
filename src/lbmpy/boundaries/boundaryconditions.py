import abc
from enum import Enum, auto
from warnings import warn

from pystencils import Assignment, Field
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.stencil import offset_to_direction_string, direction_string_to_offset, inverse_direction
from pystencils.sympyextensions import get_symmetric_part, simplify_by_equality, scalar_product
from pystencils.typing import create_type, TypedSymbol

from lbmpy.advanced_streaming.utility import AccessPdfValues, Timestep
from lbmpy.custom_code_nodes import (NeighbourOffsetArrays, MirroredStencilDirections, LbmWeightInfo,
                                     TranslationArraysNode)
from lbmpy.maxwellian_equilibrium import discrete_equilibrium
from lbmpy.simplificationfactory import create_simplification_strategy

import sympy as sp
import numpy as np


class LbBoundary(abc.ABC):
    """Base class that all boundaries should derive from.

    Args:
        name: optional name of the boundary.
    """

    inner_or_boundary = True
    single_link = False

    def __init__(self, name=None):
        self._name = name

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        """
        This function defines the boundary behavior and must therefore be implemented by all boundaries.
        The boundary is defined through a list of sympy equations from which a boundary kernel is generated.

        Args:
        f_out:          a pystencils field acting as a proxy to access the populations streaming out of the current
                        cell, i.e. the post-collision PDFs of the previous LBM step
        f_in:           a pystencils field acting as a proxy to access the populations streaming into the current
                        cell, i.e. the pre-collision PDFs for the next LBM step
        dir_symbol:     a sympy symbol that can be used as an index to f_out and f_in. It describes the direction
                        pointing from the fluid to the boundary cell.
        inv_dir:        an indexed sympy symbol which describes the inversion of a direction index. It can be used in
                        the indices of f_out and f_in for retrieving the PDF of the inverse direction.
        lb_method:      an instance of the LB method used. Use this to adapt the boundary to the method
                        (e.g. compressibility)
        index_field:    the boundary index field that can be used to retrieve and update boundary data

        Returns:
            list of pystencils assignments, or pystencils.AssignmentCollection
        """
        raise NotImplementedError("Boundary class has to overwrite __call__")

    @property
    def additional_data(self):
        """Return a list of (name, type) tuples for additional data items required in this boundary
        These data items can either be initialized in separate kernel see additional_data_kernel_init or by
        Python callbacks - see additional_data_callback """
        return []

    @property
    def additional_data_init_callback(self):
        """Return a callback function called with a boundary data setter object and returning a dict of
        data-name to data for each element that should be initialized"""
        return None

    def get_additional_code_nodes(self, lb_method):
        """Return a list of code nodes that will be added in the generated code before the index field loop.

        Args:
            lb_method: lattice Boltzmann method. See :func:`lbmpy.creationfunctions.create_lb_method`
        """
        return []

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return type(self).__name__

    @name.setter
    def name(self, new_value):
        self._name = new_value

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.__dict__ == other.__dict__


# end class Boundary


class NoSlip(LbBoundary):
    r"""
    No-Slip, (half-way) simple bounce back boundary condition, enforcing zero velocity at obstacle.
    Populations leaving the boundary node :math:`\mathbf{x}_b` at time :math:`t` are reflected
    back with :math:`\mathbf{c}_{\overline{i}} = -\mathbf{c}_{i}`

    .. math ::
        f_{\overline{i}}(\mathbf{x}_b, t + \Delta t) = f^{\star}_{i}(\mathbf{x}_b, t)

    Args:
        name: optional name of the boundary.
    """

    def __init__(self, name=None):
        """Set an optional name here, to mark boundaries, for example for force evaluations"""
        super(NoSlip, self).__init__(name)

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        return Assignment(f_in(inv_dir[dir_symbol]), f_out(dir_symbol))


class NoSlipLinearBouzidi(LbBoundary):
    """
    No-Slip, (half-way) simple bounce back boundary condition with interpolation
    to increase accuracy: :cite:`BouzidiBC`. In order to make the boundary condition work properly a
    Python callback function needs to be provided to calculate the distance from the wall for each cell near to the
    boundary. If this is not done the boundary condition will fall back to a simple NoSlip boundary.
    Furthermore, for this boundary condition a second fluid cell away from the wall is needed. If the second fluid
    cell is not available (e.g. because it is marked as boundary as well), the boundary condition should fall back to
    a NoSlip boundary as well.

    Args:
        name: optional name of the boundary.
        init_wall_distance: Python callback function to calculate the wall distance for each cell near to the  boundary
        data_type: data type of the wall distance q
    """

    def __init__(self, name=None, init_wall_distance=None, data_type='double'):
        self.data_type = data_type
        self.init_wall_distance = init_wall_distance

        super(NoSlipLinearBouzidi, self).__init__(name)

    @property
    def additional_data(self):
        """Used internally only. For the NoSlipLinearBouzidi boundary the distance to the obstacle of every
        direction is needed. This information is stored in the index vector."""
        return [('q', create_type(self.data_type))]

    @property
    def additional_data_init_callback(self):
        def default_callback(boundary_data, **_):
            for cell in boundary_data.index_array:
                cell['q'] = -1

        if self.init_wall_distance:
            return self.init_wall_distance
        else:
            warn("No callback function provided to initialise the wall distance for each cell "
                 "(init_wall_distance=None). The boundary condition will fall back to a simple NoSlip BC")
            return default_callback

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        f_xf = sp.Symbol("f_xf")
        f_xf_inv = sp.Symbol("f_xf_inv")
        d_x2f = sp.Symbol("d_x2f")
        q = sp.Symbol("q")
        one = sp.Float(1.0)
        two = sp.Float(2.0)
        half = sp.Rational(1, 2)

        subexpressions = [Assignment(f_xf, f_out(dir_symbol)),
                          Assignment(f_xf_inv, f_out(inv_dir[dir_symbol])),
                          Assignment(d_x2f, f_in(dir_symbol)),
                          Assignment(q, index_field[0]('q'))]

        case_one = (half * (f_xf + f_xf_inv * (two * q - one))) / q
        case_two = two * q * f_xf + (one - two * q) * d_x2f
        case_three = f_xf

        rhs = sp.Piecewise((case_one, sp.Ge(q, 0.5)),
                           (case_two, sp.And(sp.Gt(q, 0), sp.Lt(q, 0.5))),
                           (case_three, True))

        boundary_assignments = [Assignment(f_in(inv_dir[dir_symbol]), rhs)]

        return AssignmentCollection(boundary_assignments, subexpressions=subexpressions)


# end class NoSlipLinearBouzidi

class QuadraticBounceBack(LbBoundary):
    """
    Second order accurate bounce back boundary condition. Implementation details are provided in a demo notebook here:
    https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/demo_interpolation_boundary_conditions.html

    Args:
        relaxation_rate: relaxation rate to realise a BGK scheme for recovering the pre collision PDF value.
        name: optional name of the boundary.
        init_wall_distance: Python callback function to calculate the wall distance for each cell near to the  boundary
        data_type: data type of the wall distance q
    """

    def __init__(self, relaxation_rate, name=None, init_wall_distance=None, data_type='double'):
        self.relaxation_rate = relaxation_rate
        self.data_type = data_type
        self.init_wall_distance = init_wall_distance
        self.equilibrium_values_name = "f_eq"
        self.inv_dir_symbol = TypedSymbol("inv_dir", create_type("int32"))

        super(QuadraticBounceBack, self).__init__(name)

    @property
    def additional_data(self):
        """Used internally only. For the NoSlipLinearBouzidi boundary the distance to the obstacle of every
        direction is needed. This information is stored in the index vector."""
        return [('q', create_type(self.data_type))]

    @property
    def additional_data_init_callback(self):
        def default_callback(boundary_data, **_):
            for cell in boundary_data.index_array:
                cell['q'] = 0.5

        if self.init_wall_distance:
            return self.init_wall_distance
        else:
            warn("No callback function provided to initialise the wall distance for each cell "
                 "(init_wall_distance=None). The boundary condition will fall back to a simple NoSlip BC")
            return default_callback

    def get_additional_code_nodes(self, lb_method):
        """Return a list of code nodes that will be added in the generated code before the index field loop.

        Args:
            lb_method: Lattice Boltzmann method. See :func:`lbmpy.creationfunctions.create_lb_method`

        Returns:
            list containing LbmWeightInfo
        """
        stencil = lb_method.stencil
        inv_directions = [str(stencil.index(inverse_direction(direction))) for direction in stencil]
        dtype = self.inv_dir_symbol.dtype
        name = self.inv_dir_symbol.name
        inverse_dir_node = TranslationArraysNode([(dtype, name, inv_directions), ], {self.inv_dir_symbol})
        return [LbmWeightInfo(lb_method, self.data_type), inverse_dir_node, NeighbourOffsetArrays(lb_method.stencil)]

    @staticmethod
    def get_equilibrium(v, u, rho, drho, weight, compressible, zero_centered):
        rho_background = sp.Integer(1)

        result = discrete_equilibrium(v, u, rho, weight,
                                      order=2, c_s_sq=sp.Rational(1, 3), compressible=compressible)
        if zero_centered:
            shift = discrete_equilibrium(v, [0] * len(u), rho_background, weight,
                                         order=0, c_s_sq=sp.Rational(1, 3), compressible=False)
            result = simplify_by_equality(result - shift, rho, drho, rho_background)

        return result

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        omega = self.relaxation_rate
        inv = sp.IndexedBase(self.inv_dir_symbol, shape=(1,))[dir_symbol]
        weight_info = LbmWeightInfo(lb_method, data_type=self.data_type)
        weight_of_direction = weight_info.weight_of_direction
        pdf_field_accesses = [f_out(i) for i in range(len(lb_method.stencil))]

        pdf_symbols = [sp.Symbol(f"pdf_{i}") for i in range(len(lb_method.stencil))]
        f_xf = sp.Symbol("f_xf")
        f_xf_inv = sp.Symbol("f_xf_inv")
        q = sp.Symbol("q")
        feq = sp.Symbol("f_eq")
        weight = sp.Symbol("w")
        weight_inv = sp.Symbol("w_inv")
        v = [TypedSymbol(f"c_{i}", self.data_type) for i in range(lb_method.stencil.D)]
        v_inv = [TypedSymbol(f"c_inv_{i}", self.data_type) for i in range(lb_method.stencil.D)]
        one = sp.Float(1.0)
        half = sp.Rational(1, 2)

        subexpressions = [Assignment(pdf_symbols[i], pdf) for i, pdf in enumerate(pdf_field_accesses)]
        subexpressions.append(Assignment(f_xf, f_out(dir_symbol)))
        subexpressions.append(Assignment(f_xf_inv, f_out(inv_dir[dir_symbol])))
        subexpressions.append(Assignment(q, index_field[0]('q')))
        subexpressions.append(Assignment(weight, weight_of_direction(dir_symbol, lb_method)))
        subexpressions.append(Assignment(weight_inv, weight_of_direction(inv, lb_method)))

        for i in range(lb_method.stencil.D):
            offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
            subexpressions.append(Assignment(v[i], offset[i]))

        for i in range(lb_method.stencil.D):
            offset = NeighbourOffsetArrays.neighbour_offset(inv, lb_method.stencil)
            subexpressions.append(Assignment(v_inv[i], offset[i]))

        cqc = lb_method.conserved_quantity_computation
        rho = cqc.density_symbol
        drho = cqc.density_deviation_symbol
        u = sp.Matrix(cqc.velocity_symbols)
        compressible = cqc.compressible
        zero_centered = cqc.zero_centered_pdfs

        cqe = cqc.equilibrium_input_equations_from_pdfs(pdf_symbols, False)
        subexpressions.append(cqe.all_assignments)

        eq_dir = self.get_equilibrium(v, u, rho, drho, weight, compressible, zero_centered)
        eq_inv = self.get_equilibrium(v_inv, u, rho, drho, weight_inv, compressible, zero_centered)

        subexpressions.append(Assignment(feq, eq_dir + eq_inv))

        t1 = (f_xf - f_xf_inv + (f_xf + f_xf_inv - feq * omega) / (one - omega))
        t2 = (q * (f_xf + f_xf_inv)) / (one + q)
        result = (one - q) / (one + q) * t1 * half + t2

        boundary_assignments = [Assignment(f_in(inv_dir[dir_symbol]), result)]
        return AssignmentCollection(boundary_assignments, subexpressions=subexpressions)


# end class QuadraticBounceBack

class FreeSlip(LbBoundary):
    """
    Free-Slip boundary condition, which enforces a zero normal fluid velocity :math:`u_n = 0` but places no restrictions
    on the tangential fluid velocity :math:`u_t`.

    Args:
        stencil: LBM stencil which is used for the simulation
        normal_direction: optional normal direction pointing from wall to fluid.
                          If the Free slip boundary is applied to a certain side in the domain it is not necessary
                          to calculate the normal direction since it can be stated for all boundary cells.
                          This reduces the memory space for the index array significantly.
        name: optional name of the boundary.
    """

    def __init__(self, stencil, normal_direction=None, name=None):
        """Set an optional name here, to mark boundaries, for example for force evaluations"""
        self.stencil = stencil

        if normal_direction and len(normal_direction) - normal_direction.count(0) != 1:
            raise ValueError("It is only possible to pre specify the normal direction for simple situations."
                             "This means if the free slip boundary is applied to a straight wall or side in the "
                             "simulation domain. A possible value for example would be (0, 1, 0) if the "
                             "free slip boundary is applied to the northern wall. For more complex situations "
                             "the normal direction has to be calculated for each cell. This is done when "
                             "the normal direction is not defined for this class")

        if normal_direction:
            normal_direction = tuple([int(n) for n in normal_direction])
            assert all([n in [-1, 0, 1] for n in normal_direction]), \
                "Only -1, 0 and 1 allowed for defining the normal direction"
            self.mirror_axis = normal_direction.index(*[d for d in normal_direction if d != 0])

        self.normal_direction = normal_direction
        self.dim = len(stencil[0])

        if name is None and normal_direction:
            name = f"Free Slip : {offset_to_direction_string([-x for x in normal_direction])}"

        super(FreeSlip, self).__init__(name)

    def init_callback(self, boundary_data, **_):
        if len(boundary_data.index_array) > 1e6:
            warn(f"The calculation of the normal direction for each cell might take a long time, because "
                 f"{len(boundary_data.index_array)} cells are marked as Free Slip boundary cells. Consider specifying "
                 f" the normal direction beforehand, which is possible if it is equal for all cells (e.g. at a wall)")

        dim = boundary_data.dim
        coords = [coord for coord, _ in zip(['x', 'y', 'z'], range(dim))]

        boundary_cells = set()

        # get a set containing all boundary cells
        for cell in boundary_data.index_array:
            fluid_cell = tuple([cell[coord] for coord in coords])
            direction = self.stencil[cell['dir']]
            boundary_cell = tuple([i + d for i, d in zip(fluid_cell, direction)])
            boundary_cells.add(boundary_cell)

        for cell in boundary_data.index_array:
            fluid_cell = tuple([cell[coord] for coord in coords])
            direction = self.stencil[cell['dir']]
            ref_direction = direction
            normal_direction = [0] * dim

            for i in range(dim):
                sub_direction = [0] * dim
                sub_direction[i] = direction[i]
                test_cell = tuple([x + y for x, y in zip(fluid_cell, sub_direction)])

                if test_cell in boundary_cells:
                    normal_direction[i] = direction[i]
                    ref_direction = MirroredStencilDirections.mirror_stencil(ref_direction, i)

            # convex corner special case:
            if all(n == 0 for n in normal_direction):
                normal_direction = direction
            else:
                ref_direction = inverse_direction(ref_direction)

            for i, cell_name in zip(range(dim), self.additional_data):
                cell[cell_name[0]] = -normal_direction[i]
            cell['ref_dir'] = self.stencil.index(ref_direction)

    @property
    def additional_data(self):
        """Used internally only. For the FreeSlip boundary the information of the normal direction for each pdf
        direction is needed. This information is stored in the index vector."""
        if self.normal_direction:
            return []
        else:
            data_type = create_type('int32')
            wnz = [] if self.dim == 2 else [('wnz', data_type)]
            data = [('wnx', data_type), ('wny', data_type)] + wnz
            return data + [('ref_dir', data_type)]

    @property
    def additional_data_init_callback(self):
        if self.normal_direction:
            return None
        else:
            return self.init_callback

    def get_additional_code_nodes(self, lb_method):
        if self.normal_direction:
            return [MirroredStencilDirections(self.stencil, self.mirror_axis), NeighbourOffsetArrays(lb_method.stencil)]
        else:
            return [NeighbourOffsetArrays(lb_method.stencil)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        neighbor_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
        if self.normal_direction:
            tangential_offset = tuple(offset + normal for offset, normal in zip(neighbor_offset, self.normal_direction))
            mirrored_stencil_symbol = MirroredStencilDirections._mirrored_symbol(self.mirror_axis)
            mirrored_direction = inv_dir[sp.IndexedBase(mirrored_stencil_symbol, shape=(1,))[dir_symbol]]
        else:
            normal_direction = list()
            for i, cell_name in zip(range(self.dim), self.additional_data):
                normal_direction.append(index_field[0](cell_name[0]))
            normal_direction = tuple(normal_direction)
            tangential_offset = tuple(offset + normal for offset, normal in zip(neighbor_offset, normal_direction))

            mirrored_direction = index_field[0]('ref_dir')

        return Assignment(f_in.center(inv_dir[dir_symbol]), f_out[tangential_offset](mirrored_direction))


# end class FreeSlip


class WallFunctionBounce(LbBoundary):
    """
    Wall function based on the bounce back idea, cf. :cite:`Han_WFB`. Its implementation is extended to the D3Q27
    stencil, whereas different weights of the drag distribution are proposed.

    Args:
        lb_method: LB method which is used for the simulation
        pdfs: Symbolic representation of the particle distribution functions.
        normal_direction: Normal direction of the wall. Currently, only straight and axis-aligned walls are supported.
        wall_function_model: Wall function that is used to retrieve the wall stress `tau_w` during the simulation. See
                             :class:`lbmpy.boundaries.wall_treatment.WallFunctionModel` for more details
        mean_velocity: Optional field or field access for the mean velocity. As wall functions are typically defined
                       in terms of the mean velocity, it is recommended to provide this variable. Per default, the
                       instantaneous velocity obtained from pdfs is used for the wall function.
        sampling_shift: Optional sampling shift for the velocity sampling. Can be provided as symbolic variable or
                        integer. In both cases, the user must assure that the sampling shift is at least 1, as sampling
                        in boundary cells is not physical. Per default, a sampling shift of 1 is employed which
                        corresponds to a sampling in the first fluid cell normal to the wall. For lower friction
                        Reynolds numbers, choosing a sampling shift >1 has shown to improve the results for higher
                        resolutions.
                        Mutually exclusive with the Maronga sampling shift.
        maronga_sampling_shift: Optionally, apply a correction factor to the wall shear stress proposed by Maronga et
                                al. :cite:`Maronga2008`. Has only been tested and validated for the MOST wall function.
                                No guarantee is given that it also works with other wall functions.
                                Mutually exclusive with the standard sampling shift.
        dt: time discretisation. Usually one in LB units
        dy: space discretisation. Usually one in LB units
        y: distance from the wall
        target_friction_velocity: A target friction velocity can be given if an estimate is known a priori. This target
                                  friction velocity will be used as initial guess for implicit wall functions to ensure
                                  convergence of the Newton algorithm.
        weight_method: The extension of the WFB to a D3Q27 stencil is non-unique. Different weights can be chosen to
                       define the drag distribution onto the pdfs. Per default, weights corresponding to the weights
                       in the D3Q27 stencil are chosen.
        name: Optional name of the boundary.
        data_type: Floating-point precision. Per default, double.
    """

    class WeightMethod(Enum):
        LATTICE_WEIGHT = auto(),
        GEOMETRIC_WEIGHT = auto()

    def __init__(self, lb_method, pdfs, normal_direction, wall_function_model,
                 mean_velocity=None, sampling_shift=1, maronga_sampling_shift=None,
                 dt=1, dy=1, y=0.5,
                 target_friction_velocity=None,
                 weight_method=WeightMethod.LATTICE_WEIGHT,
                 name=None, data_type='double'):
        """Set an optional name here, to mark boundaries, for example for force evaluations"""
        self.stencil = lb_method.stencil
        if not (self.stencil.Q == 19 or self.stencil.Q == 27):
            raise ValueError("WFB boundary is currently only defined for D3Q19 and D3Q27 stencils.")
        self.pdfs = pdfs

        self.wall_function_model = wall_function_model

        if mean_velocity:
            if isinstance(mean_velocity, Field):
                self.mean_velocity = mean_velocity.center_vector
            elif isinstance(mean_velocity, Field.Access):
                self.mean_velocity = mean_velocity
            else:
                raise ValueError("Mean velocity field has to be a pystencils Field or Field.Access")
        else:
            self.mean_velocity = None

        if not isinstance(sampling_shift, int):
            self.sampling_shift = TypedSymbol(sampling_shift.name, np.uint32)
        else:
            assert sampling_shift >= 1, "The sampling shift must be greater than 1."
            self.sampling_shift = sampling_shift

        if maronga_sampling_shift:
            assert self.mean_velocity, "Mean velocity field must be provided when using the Maronga correction"
            if not isinstance(maronga_sampling_shift, int):
                self.maronga_sampling_shift = TypedSymbol(maronga_sampling_shift.name, np.uint32)
            else:
                assert maronga_sampling_shift >= 1, "The Maronga sampling shift must be greater than 1."
                self.maronga_sampling_shift = maronga_sampling_shift
        else:
            self.maronga_sampling_shift = None

        if (self.sampling_shift != 1) and self.maronga_sampling_shift:
            raise ValueError("Both sampling shift and Maronga offset are set. This is currently not supported.")

        self.dt = dt
        self.dy = dy
        self.y = y
        self.data_type = data_type

        self.target_friction_velocity = target_friction_velocity

        self.weight_method = weight_method

        if len(normal_direction) - normal_direction.count(0) != 1:
            raise ValueError("Only normal directions for straight walls are supported for example (0, 1, 0) for "
                             "a WallFunctionBounce applied to the southern boundary of the domain")

        self.mirror_axis = normal_direction.index(*[direction for direction in normal_direction if direction != 0])

        self.normal_direction = normal_direction
        assert all([n in [-1, 0, 1] for n in self.normal_direction]), \
            "Only -1, 0 and 1 allowed for defining the normal direction"
        tangential_component = [int(not n) for n in self.normal_direction]
        self.normal_axis = tangential_component.index(0)
        self.tangential_axis = [0, 1, 2]
        self.tangential_axis.remove(self.normal_axis)

        self.dim = self.stencil.D

        if name is None:
            name = f"WFB : {offset_to_direction_string([-x for x in normal_direction])}"

        super(WallFunctionBounce, self).__init__(name)

    def get_additional_code_nodes(self, lb_method):
        return [MirroredStencilDirections(self.stencil, self.mirror_axis),
                NeighbourOffsetArrays(lb_method.stencil)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        # needed symbols for offsets and indices
        # neighbour offset symbols are basically the stencil directions defined in stencils.py:L130ff.
        neighbor_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
        tangential_offset = tuple(offset + normal for offset, normal in zip(neighbor_offset, self.normal_direction))
        mirrored_stencil_symbol = MirroredStencilDirections._mirrored_symbol(self.mirror_axis)
        mirrored_direction = inv_dir[sp.IndexedBase(mirrored_stencil_symbol, shape=(1,))[dir_symbol]]

        name_base = "f_in_inv_offsets_"
        offset_array_symbols = [TypedSymbol(name_base + d, mirrored_stencil_symbol.dtype) for d in ['x', 'y', 'z']]
        mirrored_offset = sp.IndexedBase(mirrored_stencil_symbol, shape=(1,))[dir_symbol]
        offsets = tuple(sp.IndexedBase(s, shape=(1,))[mirrored_offset] for s in offset_array_symbols)

        # needed symbols in the Assignments
        u_m = sp.Symbol("u_m")
        tau_w = sp.Symbol("tau_w")
        wall_stress = sp.symbols("tau_w_x tau_w_y tau_w_z")

        # if the mean velocity field is not given, or the Maronga correction is applied, density and velocity values
        # will be calculated from pdfs
        cqc = lb_method.conserved_quantity_computation

        result = []
        if (not self.mean_velocity) or self.maronga_sampling_shift:
            pdf_center_vector = sp.Matrix([0] * self.stencil.Q)

            for i in range(self.stencil.Q):
                pdf_center_vector[i] = self.pdfs[offsets[0] + self.normal_direction[0],
                                                 offsets[1] + self.normal_direction[1],
                                                 offsets[2] + self.normal_direction[2]](i)

            eq_equations = cqc.equilibrium_input_equations_from_pdfs(pdf_center_vector)
            result.append(eq_equations.all_assignments)

        # sample velocity which will be used in the wall stress calculation
        if self.mean_velocity:
            if self.maronga_sampling_shift:
                u_for_tau_wall = tuple(u_mean_i.get_shifted(
                    self.maronga_sampling_shift * self.normal_direction[0],
                    self.maronga_sampling_shift * self.normal_direction[1],
                    self.maronga_sampling_shift * self.normal_direction[2]
                ) for u_mean_i in self.mean_velocity)
            else:
                u_for_tau_wall = tuple(u_mean_i.get_shifted(
                    self.sampling_shift * self.normal_direction[0],
                    self.sampling_shift * self.normal_direction[1],
                    self.sampling_shift * self.normal_direction[2]
                ) for u_mean_i in self.mean_velocity)

            rho_for_tau_wall = sp.Float(1)
        else:
            rho_for_tau_wall = cqc.density_symbol
            u_for_tau_wall = cqc.velocity_symbols

        # calculate Maronga factor in case of correction
        maronga_fix = sp.Symbol("maronga_fix")
        if self.maronga_sampling_shift:
            inst_first_cell_vel = cqc.velocity_symbols
            mean_first_cell_vel = tuple(u_mean_i.get_shifted(*self.normal_direction) for u_mean_i in self.mean_velocity)

            mag_inst_vel_first_cell = sp.sqrt(sum([inst_first_cell_vel[i] ** 2 for i in self.tangential_axis]))
            mag_mean_vel_first_cell = sp.sqrt(sum([mean_first_cell_vel[i] ** 2 for i in self.tangential_axis]))

            result.append(Assignment(maronga_fix, mag_inst_vel_first_cell / mag_mean_vel_first_cell))
        else:
            maronga_fix = 1

        # store which direction is tangential component (only those are used for the wall shear stress)
        red_u_mag = sp.sqrt(sum([u_for_tau_wall[i]**2 for i in self.tangential_axis]))

        u_mag = Assignment(u_m, red_u_mag)
        result.append(u_mag)

        wall_distance = self.maronga_sampling_shift if self.maronga_sampling_shift else self.sampling_shift

        # using wall function model
        wall_law_assignments = self.wall_function_model.shear_stress_assignments(
            density_symbol=rho_for_tau_wall, velocity_symbol=u_m, shear_stress_symbol=tau_w,
            wall_distance=(wall_distance - sp.Rational(1, 2) * self.dy),
            u_tau_target=self.target_friction_velocity)
        result.append(wall_law_assignments)

        # calculate wall stress components and use them to calculate the drag
        for i in self.tangential_axis:
            result.append(Assignment(wall_stress[i], - u_for_tau_wall[i] / u_m * tau_w * maronga_fix))

        weight, inv_weight_sq = sp.symbols("wfb_weight inverse_weight_squared")

        if self.stencil.Q == 19:
            result.append(Assignment(weight, sp.Rational(1, 2)))
        elif self.stencil.Q == 27:
            result.append(Assignment(inv_weight_sq, sum([neighbor_offset[i]**2 for i in self.tangential_axis])))
            a, b = sp.symbols("wfb_a wfb_b")

            if self.weight_method == self.WeightMethod.LATTICE_WEIGHT:
                res_ab = sp.solve([2 * a + 4 * b - 1, a - 4 * b], [a, b])  # lattice weight scaling
            elif self.weight_method == self.WeightMethod.GEOMETRIC_WEIGHT:
                res_ab = sp.solve([2 * a + 4 * b - 1, a - sp.sqrt(2) * b], [a, b])  # geometric scaling
            else:
                raise ValueError("Unknown weighting method for the WFB D3Q27 extension. Currently, only lattice "
                                 "weights and geometric weights are supported.")

            result.append(Assignment(weight, sp.Piecewise((sp.Float(0), sp.Equality(inv_weight_sq, 0)),
                                                          (res_ab[a], sp.Equality(inv_weight_sq, 1)),
                                                          (res_ab[b], True))))

        factor = self.dt / self.dy * weight
        drag = sum([neighbor_offset[i] * factor * wall_stress[i] for i in self.tangential_axis])

        result.append(Assignment(f_in.center(inv_dir[dir_symbol]), f_out[tangential_offset](mirrored_direction) - drag))

        return result

# end class WallFunctionBounce


class UBB(LbBoundary):
    r"""Velocity bounce back boundary condition, enforcing specified velocity at obstacle. Furthermore, a density
        at the wall can be implied. The boundary condition is implemented with the following formula:

    .. math ::
        f_{\overline{i}}(\mathbf{x}_b, t + \Delta t) = f^{\star}_{i}(\mathbf{x}_b, t) -
        2 w_{i} \rho_{w} \frac{\mathbf{c}_i \cdot \mathbf{u}_w}{c_s^2}


    Args:
        velocity: Prescribe the fluid velocity :math:`\mathbf{u}_w` at the wall.
                  Can either be a constant, an access into a field, or a callback function.
                  The callback functions gets a numpy record array with members, ``x``, ``y``, ``z``, ``dir``
                  (direction) and ``velocity`` which has to be set to the desired velocity of the corresponding link
        density: Prescribe the fluid density :math:`\rho_{w}` at the wall. If not prescribed the density is
                 calculated from the PDFs at the wall. The density can only be set constant.
        adapt_velocity_to_force: adapts the velocity to the correct equilibrium when the lattice Boltzmann method holds
                                 a forcing term. If no forcing term is set and adapt_velocity_to_force is set to True
                                 it has no effect.
        dim: number of spatial dimensions
        name: optional name of the boundary.
    """

    def __init__(self, velocity, density=None, adapt_velocity_to_force=False, dim=None, name=None, data_type='double'):
        self._velocity = velocity
        self._density = density
        self._adaptVelocityToForce = adapt_velocity_to_force
        if callable(self._velocity) and not dim:
            raise ValueError("When using a velocity callback the dimension has to be specified with the dim parameter")
        elif not callable(self._velocity):
            dim = len(velocity)
        self.dim = dim
        self.data_type = data_type

        super(UBB, self).__init__(name)

    @property
    def additional_data(self):
        """ In case of the UBB boundary additional data is a velocity vector. This vector is added to each cell to
            realize velocity profiles for the inlet."""
        if self.velocity_is_callable:
            return [(f'vel_{i}', create_type(self.data_type)) for i in range(self.dim)]
        else:
            return []

    @property
    def additional_data_init_callback(self):
        """Initialise additional data of the boundary. For an example see
            `tutorial 02 <https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/02_tutorial_boundary_setup.html>`_
            or lbmpy.geometry.add_pipe_inflow_boundary"""
        if callable(self._velocity):
            return self._velocity

    def get_additional_code_nodes(self, lb_method):
        """Return a list of code nodes that will be added in the generated code before the index field loop.

        Args:
            lb_method: Lattice Boltzmann method. See :func:`lbmpy.creationfunctions.create_lb_method`

        Returns:
            list containing LbmWeightInfo and NeighbourOffsetArrays
        """
        return [LbmWeightInfo(lb_method, self.data_type), NeighbourOffsetArrays(lb_method.stencil)]

    @property
    def velocity_is_callable(self):
        """Returns True is velocity is callable. This means the velocity should be initialised via a callback function.
        This is useful if the inflow velocity should have a certain profile for instance"""
        return callable(self._velocity)

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        vel_from_idx_field = callable(self._velocity)
        vel = [index_field(f'vel_{i}') for i in range(self.dim)] if vel_from_idx_field else self._velocity

        assert self.dim == lb_method.dim, \
            f"Dimension of UBB ({self.dim}) does not match dimension of method ({lb_method.dim})"

        neighbor_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)

        velocity = tuple(v_i.get_shifted(*neighbor_offset)
                         if isinstance(v_i, Field.Access) and not vel_from_idx_field
                         else v_i
                         for v_i in vel)

        if self._adaptVelocityToForce:
            cqc = lb_method.conserved_quantity_computation
            shifted_vel_eqs = cqc.equilibrium_input_equations_from_init_values(velocity=velocity)
            shifted_vel_eqs = shifted_vel_eqs.new_without_subexpressions()
            velocity = [eq.rhs for eq in shifted_vel_eqs.new_filtered(cqc.velocity_symbols).main_assignments]

        c_s_sq = sp.Rational(1, 3)
        weight_info = LbmWeightInfo(lb_method, data_type=self.data_type)
        weight_of_direction = weight_info.weight_of_direction
        vel_term = 2 / c_s_sq * sum([d_i * v_i for d_i, v_i in zip(neighbor_offset, velocity)]) * weight_of_direction(
            dir_symbol, lb_method)

        # Better alternative: in conserved value computation
        # rename what is currently called density to "virtual_density"
        # provide a new quantity density, which is constant in case of incompressible models
        if lb_method.conserved_quantity_computation.compressible:
            cqc = lb_method.conserved_quantity_computation
            density_symbol = sp.Symbol("rho")
            pdf_field_accesses = [f_out(i) for i in range(len(lb_method.stencil))]
            density_equations = cqc.output_equations_from_pdfs(pdf_field_accesses, {'density': density_symbol})
            density_symbol = lb_method.conserved_quantity_computation.density_symbol
            if self._density:
                result = [Assignment(density_symbol, self._density)]
            else:
                result = density_equations.all_assignments
            result += [Assignment(f_in(inv_dir[dir_symbol]),
                                  f_out(dir_symbol) - vel_term * density_symbol)]
            return result
        else:
            return [Assignment(f_in(inv_dir[dir_symbol]),
                               f_out(dir_symbol) - vel_term)]


# end class UBB


class SimpleExtrapolationOutflow(LbBoundary):
    r"""
    Simple Outflow boundary condition :cite:`geier2015`, equation F.1 (listed below).
    This boundary condition extrapolates missing populations from the last layer of
    fluid cells onto the boundary by copying them in the normal direction.

    .. math ::
        f_{\overline{1}jkxyzt} = f_{\overline{1}jk(x - \Delta x)yzt}


    Args:
        normal_direction: direction vector normal to the outflow
        stencil: stencil used by the lattice boltzmann method
        name: optional name of the boundary.
    """

    def __init__(self, normal_direction, stencil, name=None):
        if isinstance(normal_direction, str):
            normal_direction = direction_string_to_offset(normal_direction, dim=len(stencil[0]))

        if name is None:
            name = f"Simple Outflow: {offset_to_direction_string(normal_direction)}"

        self.normal_direction = tuple([int(n) for n in normal_direction])
        assert all([n in [-1, 0, 1] for n in self.normal_direction]), \
            "Only -1, 0 and 1 allowed for defining the normal direction"
        super(SimpleExtrapolationOutflow, self).__init__(name)

    def get_additional_code_nodes(self, lb_method):
        """Return a list of code nodes that will be added in the generated code before the index field loop.

        Args:
            lb_method: Lattice Boltzmann method. See :func:`lbmpy.creationfunctions.create_lb_method`

        Returns:
            list containing NeighbourOffsetArrays

        """
        return [NeighbourOffsetArrays(lb_method.stencil)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        neighbor_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
        tangential_offset = tuple(offset - normal for offset, normal in zip(neighbor_offset, self.normal_direction))

        return Assignment(f_in.center(inv_dir[dir_symbol]), f_out[tangential_offset](inv_dir[dir_symbol]))


# end class SimpleExtrapolationOutflow


class ExtrapolationOutflow(LbBoundary):
    r"""
    Outflow boundary condition :cite:`geier2015`, equation F.2, with u neglected (listed below).
    This boundary condition interpolates populations missing on the boundary in normal direction.
    For this interpolation, the PDF values of the last time step are used. They are interpolated
    between fluid cell and boundary cell. To get the PDF values from the last time step an index
    array is used which stores them.

    .. math ::
        f_{\overline{1}jkxyzt} = f_{\overline{1}jk(x - \Delta x)yz(t - \Delta t)} c \theta^{\frac{1}{2}}
        \frac{\Delta t}{\Delta x} + \left(1 - c \theta^{\frac{1}{2}} \frac{\Delta t}{\Delta x} \right)
         f_{\overline{1}jk(x - \Delta x)yzt}


    Args:
        normal_direction: direction vector normal to the outflow
        lb_method: the lattice boltzman method to be used in the simulation
        dt: lattice time step size
        dx: lattice spacing distance
        name: optional name of the boundary.
        streaming_pattern: Streaming pattern to be used in the simulation
        zeroth_timestep: for in-place patterns, whether the initial setup corresponds to an even or odd time step
        initial_density: floating point constant or callback taking spatial coordinates (x, y [,z]) as
                         positional arguments, specifying the initial density on boundary nodes
        initial_velocity: tuple of floating point constants or callback taking spatial coordinates (x, y [,z]) as
                          positional arguments, specifying the initial velocity on boundary nodes
    """

    def __init__(self, normal_direction, lb_method, dt=1, dx=1, name=None,
                 streaming_pattern='pull', zeroth_timestep=Timestep.BOTH,
                 initial_density=None, initial_velocity=None, data_type='double'):

        self.lb_method = lb_method
        self.stencil = lb_method.stencil
        self.dim = len(self.stencil[0])

        if isinstance(normal_direction, str):
            normal_direction = direction_string_to_offset(normal_direction, dim=self.dim)

        if name is None:
            name = f"Outflow: {offset_to_direction_string(normal_direction)}"

        self.normal_direction = tuple([int(n) for n in normal_direction])
        assert all([n in [-1, 0, 1] for n in self.normal_direction]), \
            "Only -1, 0 and 1 allowed for defining the normal direction"
        self.streaming_pattern = streaming_pattern
        self.zeroth_timestep = zeroth_timestep
        self.dx = sp.Number(dx)
        self.dt = sp.Number(dt)
        self.c = sp.sqrt(sp.Rational(1, 3)) * (self.dx / self.dt)

        self.initial_density = initial_density
        self.initial_velocity = initial_velocity
        self.equilibrium_calculation = None

        self.data_type = data_type

        if initial_density and initial_velocity:
            equilibrium = lb_method.get_equilibrium(conserved_quantity_equations=AssignmentCollection([]))
            rho = lb_method.zeroth_order_equilibrium_moment_symbol
            u_vec = lb_method.first_order_equilibrium_moment_symbols
            eq_lambda = equilibrium.lambdify((rho,) + u_vec)
            post_pdf_symbols = lb_method.post_collision_pdf_symbols

            def calc_eq_pdfs(density, velocity, j):
                return eq_lambda(density, *velocity)[post_pdf_symbols[j]]

            self.equilibrium_calculation = calc_eq_pdfs

        super(ExtrapolationOutflow, self).__init__(name)

    def init_callback(self, boundary_data, **_):
        dim = boundary_data.dim
        coord_names = ['x', 'y', 'z'][:dim]
        pdf_acc = AccessPdfValues(self.stencil, streaming_pattern=self.streaming_pattern,
                                  timestep=self.zeroth_timestep, streaming_dir='out')

        def get_boundary_cell_pdfs(f_cell, b_cell, direction):
            if self.equilibrium_calculation is not None:
                density = self.initial_density(
                    *b_cell) if callable(self.initial_density) else self.initial_density
                velocity = self.initial_velocity(
                    *b_cell) if callable(self.initial_velocity) else self.initial_velocity
                return self.equilibrium_calculation(density, velocity, direction)
            else:
                return pdf_acc.read_pdf(boundary_data.pdf_array, f_cell, direction)

        for entry in boundary_data.index_array:
            center = tuple(entry[c] for c in coord_names)
            direction = self.stencil[entry["dir"]]
            inv_dir = self.stencil.index(inverse_direction(direction))
            tangential_offset = tuple(offset - normal for offset, normal in zip(direction, self.normal_direction))
            domain_cell = tuple(f + o for f, o in zip(center, tangential_offset))
            outflow_cell = tuple(f + o for f, o in zip(center, direction))

            #   Initial fluid cell PDF values
            entry['pdf'] = pdf_acc.read_pdf(boundary_data.pdf_array, domain_cell, inv_dir)
            entry['pdf_nd'] = get_boundary_cell_pdfs(domain_cell, outflow_cell, inv_dir)

    @property
    def additional_data(self):
        """Used internally only. For the ExtrapolationOutflow information of the previous PDF values is needed. This
        information is stored in the index vector."""
        data = [('pdf', create_type(self.data_type)), ('pdf_nd', create_type(self.data_type))]
        return data

    @property
    def additional_data_init_callback(self):
        return self.init_callback

    def get_additional_code_nodes(self, lb_method):
        """Return a list of code nodes that will be added in the generated code before the index field loop.

        Args:
            lb_method: Lattice Boltzmann method. See :func:`lbmpy.creationfunctions.create_lb_method`

        Returns:
            list containing NeighbourOffsetArrays

        """
        return [NeighbourOffsetArrays(lb_method.stencil)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        subexpressions = []
        boundary_assignments = []
        dtdx = sp.Rational(self.dt, self.dx)

        neighbor_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
        tangential_offset = tuple(offset - normal for offset, normal in zip(neighbor_offset, self.normal_direction))

        interpolated_pdf_sym = sp.Symbol('pdf_inter')
        interpolated_pdf_asm = Assignment(interpolated_pdf_sym, (index_field[0]('pdf') * (self.c * dtdx))
                                          + ((sp.Number(1) - self.c * dtdx) * index_field[0]('pdf_nd')))
        subexpressions.append(interpolated_pdf_asm)

        asm = Assignment(f_in.center(inv_dir[dir_symbol]), interpolated_pdf_sym)
        boundary_assignments.append(asm)

        asm = Assignment(index_field[0]('pdf'), f_out[tangential_offset](inv_dir[dir_symbol]))
        boundary_assignments.append(asm)

        asm = Assignment(index_field[0]('pdf_nd'), interpolated_pdf_sym)
        boundary_assignments.append(asm)

        return AssignmentCollection(boundary_assignments, subexpressions=subexpressions)


# end class ExtrapolationOutflow


class FixedDensity(LbBoundary):
    r"""Boundary condition for prescribing a density at the wall. Through :math:`p = c_s^2 \rho` this boundary condition
        can also function as a pressure boundary condition.

    .. math ::
        f_{\overline{i}}(\mathbf{x}_b, t + \Delta t) = - f^{\star}_{i}(\mathbf{x}_b, t) +
        2 w_{i} \rho_{w} (1 + \frac{(\mathbf{c}_i \cdot \mathbf{u}_w)^2}{2c_s^4} + \frac{\mathbf{u}_w^2}{2c_s^2})

    Args:
        density: value of the density which should be set.
        name: optional name of the boundary.
    """

    def __init__(self, density, name=None):
        if name is None:
            name = "Fixed Density " + str(density)
        self.density = density

        super(FixedDensity, self).__init__(name)

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        def remove_asymmetric_part_of_main_assignments(assignment_collection, degrees_of_freedom):
            new_main_assignments = [Assignment(a.lhs, get_symmetric_part(a.rhs, degrees_of_freedom))
                                    for a in assignment_collection.main_assignments]
            return assignment_collection.copy(new_main_assignments)

        cqc = lb_method.conserved_quantity_computation
        velocity = cqc.velocity_symbols
        symmetric_eq = remove_asymmetric_part_of_main_assignments(lb_method.get_equilibrium(),
                                                                  degrees_of_freedom=velocity)
        substitutions = {sym: f_out(i) for i, sym in enumerate(lb_method.pre_collision_pdf_symbols)}
        symmetric_eq = symmetric_eq.new_with_substitutions(substitutions)

        simplification = create_simplification_strategy(lb_method)
        symmetric_eq = simplification(symmetric_eq)

        density = self.density
        equilibrium_input = cqc.equilibrium_input_equations_from_init_values(density=density)
        equilibrium_input = equilibrium_input.new_without_subexpressions()
        equilibrium_input = equilibrium_input.main_assignments_dict

        subexpressions_dict = symmetric_eq.subexpressions_dict
        subexpressions_dict[cqc.density_symbol] = equilibrium_input[cqc.density_symbol]
        subexpressions_dict[cqc.density_deviation_symbol] = equilibrium_input[cqc.density_deviation_symbol]

        conditions = [(eq_i.rhs, sp.Equality(dir_symbol, i))
                      for i, eq_i in enumerate(symmetric_eq.main_assignments)] + [(0, True)]
        eq_component = sp.Piecewise(*conditions)

        main_assignments = [Assignment(f_in(inv_dir[dir_symbol]), 2 * eq_component - f_out(dir_symbol))]

        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions_dict)
        ac = ac.new_without_unused_subexpressions()
        ac.topological_sort()

        return ac


# end class FixedDensity

class DiffusionDirichlet(LbBoundary):
    """Concentration boundary which is used for concentration or thermal boundary conditions of convection-diffusion
    equation Base on https://doi.org/10.1103/PhysRevE.85.016701.

    Args:
        concentration: can either be a constant, an access into a field, or a callback function.
                       The callback functions gets a numpy record array with members, ``x``, ``y``, ``z``, ``dir``
                       (direction) and ``concentration`` which has to be set to the desired
                       velocity of the corresponding link
        velocity_field: if velocity field is given the boundary value is approximated by using the discrete equilibrium.
        name: optional name of the boundary.
        data_type: data type of the concentration value. default is double
    """

    def __init__(self, concentration, velocity_field=None, name=None, data_type='double'):
        if name is None:
            name = "DiffusionDirichlet"
        self.concentration = concentration
        self._data_type = data_type
        self.concentration_is_callable = callable(self.concentration)
        self.velocity_field = velocity_field

        super(DiffusionDirichlet, self).__init__(name)

    @property
    def additional_data(self):
        """ In case of the UBB boundary additional data is a velocity vector. This vector is added to each cell to
            realize velocity profiles for the inlet."""
        if self.concentration_is_callable:
            return [('concentration', create_type(self._data_type))]
        else:
            return []

    @property
    def additional_data_init_callback(self):
        """Initialise additional data of the boundary. For an example see
            `tutorial 02 <https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/02_tutorial_boundary_setup.html>`_
            or lbmpy.geometry.add_pipe_inflow_boundary"""
        if self.concentration_is_callable:
            return self.concentration

    def get_additional_code_nodes(self, lb_method):
        """Return a list of code nodes that will be added in the generated code before the index field loop.

        Args:
            lb_method: Lattice Boltzmann method. See :func:`lbmpy.creationfunctions.create_lb_method`

        Returns:
            list containing LbmWeightInfo
        """
        if self.velocity_field:
            return [LbmWeightInfo(lb_method, self._data_type), NeighbourOffsetArrays(lb_method.stencil)]
        else:
            return [LbmWeightInfo(lb_method, self._data_type)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        assert lb_method.conserved_quantity_computation.zero_centered_pdfs is False, \
            "DiffusionDirichlet only works for methods with normal pdfs storage -> set zero_centered=False"
        weight_info = LbmWeightInfo(lb_method, self._data_type)
        w_dir = weight_info.weight_of_direction(dir_symbol, lb_method)

        if self.concentration_is_callable:
            concentration = index_field[0]('concentration')
        else:
            concentration = self.concentration

        if self.velocity_field:
            neighbour_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
            u = self.velocity_field
            cs = sp.Rational(1, 3)

            equilibrium = (1 + scalar_product(neighbour_offset, u.center_vector)**2 / (2 * cs**4)
                           - scalar_product(u.center_vector, u.center_vector) / (2 * cs**2))
        else:
            equilibrium = sp.Rational(1, 1)

        result = [Assignment(f_in(inv_dir[dir_symbol]), 2.0 * w_dir * concentration * equilibrium - f_out(dir_symbol))]
        return result


# end class DiffusionDirichlet


class NeumannByCopy(LbBoundary):
    """Neumann boundary condition which is implemented by coping the PDF values to achieve similar values at the fluid
       and the boundary node"""

    def get_additional_code_nodes(self, lb_method):
        """Return a list of code nodes that will be added in the generated code before the index field loop.

        Args:
            lb_method: Lattice Boltzmann method. See :func:`lbmpy.creationfunctions.create_lb_method`

        Returns:
            list containing NeighbourOffsetArrays
        """
        return [NeighbourOffsetArrays(lb_method.stencil)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        neighbour_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
        return [Assignment(f_in(inv_dir[dir_symbol]), f_out(inv_dir[dir_symbol])),
                Assignment(f_out[neighbour_offset](dir_symbol), f_out(dir_symbol))]


# end class NeumannByCopy


class StreamInConstant(LbBoundary):
    """Boundary condition that takes a constant and overrides the boundary PDFs with this value. This is used for
    debugging mainly.

    Args:
        constant: value which should be set for the PDFs at the boundary cell.
        name: optional name of the boundary.
    """

    def __init__(self, constant, name=None):
        super(StreamInConstant, self).__init__(name)
        self.constant = constant

    def get_additional_code_nodes(self, lb_method):
        """Return a list of code nodes that will be added in the generated code before the index field loop.

        Args:
            lb_method: Lattice Boltzmann method. See :func:`lbmpy.creationfunctions.create_lb_method`

        Returns:
            list containing NeighbourOffsetArrays
        """
        return [NeighbourOffsetArrays(lb_method.stencil)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        neighbour_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
        return [Assignment(f_in(inv_dir[dir_symbol]), self.constant),
                Assignment(f_out[neighbour_offset](dir_symbol), self.constant)]

# end class StreamInConstant
