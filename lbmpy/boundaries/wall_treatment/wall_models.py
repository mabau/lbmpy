import sympy as sp
import numpy as np

from pystencils import Assignment, TypedSymbol
from pystencils.stencil import offset_to_direction_string

from lbmpy.advanced_streaming.indexing import MirroredStencilDirections, NeighbourOffsetArrays
from lbmpy.boundaries.boundaryconditions import LbBoundary
from lbmpy.boundaries.wall_treatment.spaldings_law import spaldings_law
from lbmpy.relaxationrates import lattice_viscosity_from_relaxation_rate


class WallFunctionBounce(LbBoundary):
    """
    Wall function based on the bounce back idea.

    Args:
        stencil: LBM stencil which is used for the simulation
        pdfs: symbolic representation of the particle distribution functions.
        normal_direction: optional normal direction. If the Free slip boundary is applied to a certain side in the
                          domain it is not necessary to calculate the normal direction since it can be stated for all
                          boundary cells. This reduces the memory space for the index array significantly.
        omega: relaxation rate used in the simulation
        kappa: free parameter used for spaldings law
        B: free parameter used for spaldings law
        dt: time discretisation. Usually one in LB units
        dy: space discretisation. Usually one in LB units
        y: distance from the wall
        newton_steps: number of newton steps to evaluate spaldings law.
        name: optional name of the boundary.
    """

    def __init__(self, stencil, pdfs, normal_direction, omega, kappa=0.41, B=5.5,
                 dt=1, dy=1, y=0.5, newton_steps=10, name=None):
        """Set an optional name here, to mark boundaries, for example for force evaluations"""
        self.stencil = stencil
        self.pdfs = pdfs
        self.omega = omega

        self.kappa = kappa
        self.B = B
        self.dt = dt
        self.dy = dy
        self.y = y
        self.newton_steps = newton_steps

        if len(normal_direction) - normal_direction.count(0) != 1:
            raise ValueError("Only normal directions for straight walls are supported for example (0, 1, 0) for "
                             "a WallFunctionBounce applied to the southern boundary of the domain")

        self.mirror_axis = normal_direction.index(*[dir for dir in normal_direction if dir != 0])

        self.normal_direction = normal_direction
        self.dim = stencil.D

        if name is None:
            name = f"WFB : {offset_to_direction_string([-x for x in normal_direction])}"

        super(WallFunctionBounce, self).__init__(name)

    def get_additional_code_nodes(self, lb_method):
        return [MirroredStencilDirections(self.stencil, self.mirror_axis), NeighbourOffsetArrays(lb_method.stencil)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        # needed symbols for offsets and indices
        mirrored_stencil_symbol = MirroredStencilDirections._mirrored_symbol(self.mirror_axis)
        mirrored_direction = inv_dir[sp.IndexedBase(mirrored_stencil_symbol, shape=(1,))[dir_symbol]]

        name_base = "f_in_inv_offsets_"
        offset_array_symbols = [TypedSymbol(name_base + d, np.int64) for d in ['x', 'y', 'z']]
        mirrored_offset = sp.IndexedBase(mirrored_stencil_symbol, shape=(1,))[dir_symbol]
        offsets = tuple(sp.IndexedBase(s, shape=(1,))[mirrored_offset] for s in offset_array_symbols)

        # needed symbols in the Assignments
        u_t = sp.symbols(f"u_tau_:{self.newton_steps + 1}")
        u_m = sp.Symbol("u_m")
        delta = sp.symbols(f"delta_:{self.newton_steps}")
        tau_w = sp.Symbol("tau_w")
        tau_w_x, tau_w_z = sp.symbols("tau_w_x tau_w_z")

        normal_direction = self.normal_direction

        # get velocity and density
        cqc = lb_method.conserved_quantity_computation
        rho = cqc.zeroth_order_moment_symbol
        u = cqc.first_order_moment_symbols

        pdf_center_vector = sp.Matrix([0] * self.stencil.Q)

        for i in range(self.stencil.Q):
            pdf_center_vector[i] = self.pdfs[offsets[0] + normal_direction[0],
                                             offsets[1] + normal_direction[1],
                                             offsets[2] + normal_direction[2]](i)

        eq_equations = cqc.equilibrium_input_equations_from_pdfs(pdf_center_vector)
        result = eq_equations.all_assignments

        u_mag = Assignment(u_m, sp.sqrt(sum([x**2 for x in u])))
        result.append(u_mag)

        # using spaldings law

        nu = lattice_viscosity_from_relaxation_rate(self.omega)
        up = u_m / u_t[0]

        y_plus = (self.y * u_t[0]) / nu

        wall_law = spaldings_law(up, y_plus, kappa=self.kappa, B=self.B)
        m = -wall_law / wall_law.diff(u_t[0])

        init_guess = Assignment(u_t[0], u_m)

        newton_assignments = []
        for i in range(self.newton_steps):
            newton_assignments.append(Assignment(delta[i], m.subs({u_t[0]: u_t[i]})))
            newton_assignments.append(Assignment(u_t[i + 1], u_t[i] + delta[i]))

        shear_stress = Assignment(tau_w, u_t[self.newton_steps]**2 * rho)

        result.append(init_guess)
        result.extend(newton_assignments)
        result.append(shear_stress)

        # calculate tau_wx and tau_wz and use it to calculate the drag

        result.append(Assignment(tau_w_x, - u[0] / (sp.sqrt(u[0]**2 + u[2]**2)) * tau_w))
        result.append(Assignment(tau_w_z, - u[2] / (sp.sqrt(u[0] ** 2 + u[2] ** 2)) * tau_w))

        factor = self.dt / (2 * self.dy)
        neighbor_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
        drag = neighbor_offset[0] * factor * tau_w_x + neighbor_offset[2] * factor * tau_w_z

        result.append(Assignment(f_in(inv_dir[dir_symbol]), f_in[normal_direction](mirrored_direction) - drag))

        return result
