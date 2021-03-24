from lbmpy.advanced_streaming.utility import AccessPdfValues, Timestep
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils import Assignment, Field
from lbmpy.boundaries.boundaryhandling import LbmWeightInfo
from pystencils.data_types import create_type
from pystencils.sympyextensions import get_symmetric_part
from lbmpy.simplificationfactory import create_simplification_strategy
from lbmpy.advanced_streaming.indexing import NeighbourOffsetArrays
from pystencils.stencil import offset_to_direction_string, direction_string_to_offset, inverse_direction

import sympy as sp


class LbBoundary:
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


# end class Boundary


class NoSlip(LbBoundary):
    """
        No-Slip, (half-way) simple bounce back boundary condition, enforcing zero velocity at obstacle.
        Extended for use with any streaming pattern.

    Args:
        name: optional name of the boundary.
    """

    def __init__(self, name=None):
        """Set an optional name here, to mark boundaries, for example for force evaluations"""
        super(NoSlip, self).__init__(name)

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        return Assignment(f_in(inv_dir[dir_symbol]), f_out(dir_symbol))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, NoSlip):
            return False
        return self.name == other.name


# end class NoSlip


class UBB(LbBoundary):
    """Velocity bounce back boundary condition, enforcing specified velocity at obstacle

    Args:
        velocity: can either be a constant, an access into a field, or a callback function.
                  The callback functions gets a numpy record array with members, 'x','y','z', 'dir' (direction)
                  and 'velocity' which has to be set to the desired velocity of the corresponding link
        adapt_velocity_to_force: adapts the velocity to the correct equilibrium when the lattice Boltzmann method holds
                                 a forcing term. If no forcing term is set and adapt_velocity_to_force is set to True
                                 it has no effect.
        dim: number of spatial dimensions
        name: optional name of the boundary.
    """

    def __init__(self, velocity, adapt_velocity_to_force=False, dim=None, name=None, data_type='double'):
        super(UBB, self).__init__(name)
        self._velocity = velocity
        self._adaptVelocityToForce = adapt_velocity_to_force
        if callable(self._velocity) and not dim:
            raise ValueError("When using a velocity callback the dimension has to be specified with the dim parameter")
        elif not callable(self._velocity):
            dim = len(velocity)
        self.dim = dim
        self.data_type = data_type

    @property
    def additional_data(self):
        """ In case of the UBB boundary additional data is a velocity vector. This vector is added to each cell to
            realize velocity profiles for the inlet."""
        if self.velocity_is_callable:
            return [('vel_%d' % (i,), create_type(self.data_type)) for i in range(self.dim)]
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
        return [LbmWeightInfo(lb_method), NeighbourOffsetArrays(lb_method.stencil)]

    @property
    def velocity_is_callable(self):
        """Returns True is velocity is callable. This means the velocity should be initialised via a callback function.
        This is useful if the inflow velocity should have a certain profile for instance"""
        return callable(self._velocity)

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        vel_from_idx_field = callable(self._velocity)
        vel = [index_field(f'vel_{i}') for i in range(self.dim)] if vel_from_idx_field else self._velocity
        direction = dir_symbol

        assert self.dim == lb_method.dim, \
            f"Dimension of UBB ({self.dim}) does not match dimension of method ({lb_method.dim})"

        neighbor_offset = NeighbourOffsetArrays.neighbour_offset(direction, lb_method.stencil)

        velocity = tuple(v_i.get_shifted(*neighbor_offset)
                         if isinstance(v_i, Field.Access) and not vel_from_idx_field
                         else v_i
                         for v_i in vel)

        if self._adaptVelocityToForce:
            cqc = lb_method.conserved_quantity_computation
            shifted_vel_eqs = cqc.equilibrium_input_equations_from_init_values(velocity=velocity)
            velocity = [eq.rhs for eq in shifted_vel_eqs.new_filtered(cqc.first_order_moment_symbols).main_assignments]

        c_s_sq = sp.Rational(1, 3)
        weight_of_direction = LbmWeightInfo.weight_of_direction
        vel_term = 2 / c_s_sq * sum([d_i * v_i for d_i, v_i in zip(neighbor_offset, velocity)]) * weight_of_direction(
            direction, lb_method)

        # Better alternative: in conserved value computation
        # rename what is currently called density to "virtual_density"
        # provide a new quantity density, which is constant in case of incompressible models
        if not lb_method.conserved_quantity_computation.zero_centered_pdfs:
            cqc = lb_method.conserved_quantity_computation
            density_symbol = sp.Symbol("rho")
            pdf_field_accesses = [f_out(i) for i in range(len(lb_method.stencil))]
            density_equations = cqc.output_equations_from_pdfs(pdf_field_accesses, {'density': density_symbol})
            density_symbol = lb_method.conserved_quantity_computation.defined_symbols()['density']
            result = density_equations.all_assignments
            result += [Assignment(f_in(inv_dir[direction]),
                                  f_out(direction) - vel_term * density_symbol)]
            return result
        else:
            return [Assignment(f_in(inv_dir[direction]),
                               f_out(direction) - vel_term)]


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

        self.normal_direction = normal_direction
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

        self.data_type = data_type

        self.lb_method = lb_method
        self.stencil = lb_method.stencil
        self.dim = len(self.stencil[0])
        if isinstance(normal_direction, str):
            normal_direction = direction_string_to_offset(normal_direction, dim=self.dim)

        if name is None:
            name = f"Outflow: {offset_to_direction_string(normal_direction)}"

        self.normal_direction = normal_direction
        self.streaming_pattern = streaming_pattern
        self.zeroth_timestep = zeroth_timestep
        self.dx = sp.Number(dx)
        self.dt = sp.Number(dt)
        self.c = sp.sqrt(sp.Rational(1, 3)) * (self.dx / self.dt)

        self.initial_density = initial_density
        self.initial_velocity = initial_velocity
        self.equilibrium_calculation = None

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
    """Boundary condition that fixes the density/pressure at the obstacle.

    Args:
        density: value of the density which should be set.
        name: optional name of the boundary.
    """

    def __init__(self, density, name=None):
        if name is None:
            name = "Fixed Density " + str(density)
        super(FixedDensity, self).__init__(name)
        self._density = density

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        def remove_asymmetric_part_of_main_assignments(assignment_collection, degrees_of_freedom):
            new_main_assignments = [Assignment(a.lhs, get_symmetric_part(a.rhs, degrees_of_freedom))
                                    for a in assignment_collection.main_assignments]
            return assignment_collection.copy(new_main_assignments)

        cqc = lb_method.conserved_quantity_computation
        velocity = cqc.defined_symbols()['velocity']
        symmetric_eq = remove_asymmetric_part_of_main_assignments(lb_method.get_equilibrium(),
                                                                  degrees_of_freedom=velocity)
        substitutions = {sym: f_out(i) for i, sym in enumerate(lb_method.pre_collision_pdf_symbols)}
        symmetric_eq = symmetric_eq.new_with_substitutions(substitutions)

        simplification = create_simplification_strategy(lb_method)
        symmetric_eq = simplification(symmetric_eq)

        density_symbol = cqc.defined_symbols()['density']

        density = self._density
        equilibrium_input = cqc.equilibrium_input_equations_from_init_values(density=density)
        equilibrium_input = equilibrium_input.new_without_subexpressions()
        density_eq = equilibrium_input.main_assignments[0]
        assert density_eq.lhs == density_symbol
        transformed_density = density_eq.rhs

        conditions = [(eq_i.rhs, sp.Equality(dir_symbol, i))
                      for i, eq_i in enumerate(symmetric_eq.main_assignments)] + [(0, True)]
        eq_component = sp.Piecewise(*conditions)

        subexpressions = [Assignment(eq.lhs, transformed_density if eq.lhs == density_symbol else eq.rhs)
                          for eq in symmetric_eq.subexpressions]

        return subexpressions + [Assignment(f_in(inv_dir[dir_symbol]),
                                            2 * eq_component - f_out(dir_symbol))]


# end class FixedDensity


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

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal
        return hash("NeumannByCopy")

    def __eq__(self, other):
        return type(other) == NeumannByCopy


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
        self._constant = constant

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
        return [Assignment(f_in(inv_dir[dir_symbol]), self._constant),
                Assignment(f_out[neighbour_offset](dir_symbol), self._constant)]

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal
        return hash("StreamInConstant")

    def __eq__(self, other):
        return type(other) == StreamInConstant
# end class StreamInConstant
