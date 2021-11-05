import sympy as sp
from pystencils import Assignment
from pystencils.stencil import offset_to_direction_string

from lbmpy.advanced_streaming.indexing import MirroredStencilDirections, NeighbourOffsetArrays
from lbmpy.boundaries.boundaryconditions import LbBoundary
from lbmpy.relaxationrates import lattice_viscosity_from_relaxation_rate


class WallFunctionBounce(LbBoundary):
    """
    Wall function based on the bounce back idea.

    Args:
        stencil: LBM stencil which is used for the simulation
        normal_direction: optional normal direction. If the Free slip boundary is applied to a certain side in the
                          domain it is not necessary to calculate the normal direction since it can be stated for all
                          boundary cells. This reduces the memory space for the index array significantly.
        name: optional name of the boundary.
    """

    def __init__(self, stencil, normal_direction, omega, name=None):
        """Set an optional name here, to mark boundaries, for example for force evaluations"""
        self.stencil = stencil
        self.omega = omega

        if len(normal_direction) - normal_direction.count(0) != 1:
            raise ValueError("Only normal directions for straight walls are supported for example (0, 1, 0) for "
                             "a WallFunctionBounce applied to the southern boundary of the domain")

        self.mirror_axis = normal_direction.index(*[dir for dir in normal_direction if dir != 0])

        self.normal_direction = normal_direction
        self.dim = len(stencil[0])

        if name is None:
            name = f"WFB : {offset_to_direction_string([-x for x in normal_direction])}"

        super(WallFunctionBounce, self).__init__(name)

    def get_additional_code_nodes(self, lb_method):
        return [MirroredStencilDirections(self.stencil, self.mirror_axis), NeighbourOffsetArrays(lb_method.stencil)]

    def __call__(self, f_out, f_in, dir_symbol, inv_dir, lb_method, index_field):
        normal_direction = self.normal_direction
        mirrored_stencil_symbol = MirroredStencilDirections._mirrored_symbol(self.mirror_axis)
        mirrored_direction = inv_dir[sp.IndexedBase(mirrored_stencil_symbol, shape=(1,))[dir_symbol]]

        dt = 1
        dy = 1
        factor = dt / (2 * dy)

        stencil = self.stencil
        cqc = lb_method.conserved_quantity_computation
        rho = cqc.zeroth_order_moment_symbol
        u = cqc.first_order_moment_symbols

        newton_steps = 10

        u_t = sp.symbols(f"u_tau_:{newton_steps + 1}")
        u_m = sp.Symbol("u_m")
        B = 5.5
        k = 0.41
        delta = sp.symbols(f"delta_:{newton_steps}")
        tau_w = sp.Symbol("tau_w")
        tau_w_x, tau_w_z = sp.symbols("tau_w_x tau_w_z")

        pdf_center_vector = sp.Matrix([0] * stencil.Q)

        for i in range(stencil.Q):
            pdf_center_vector[i] = f_in[normal_direction](i)

        eq_equations = cqc.equilibrium_input_equations_from_pdfs(pdf_center_vector)

        u_mag = Assignment(u_m, sp.sqrt(sum([x**2 for x in u])))
        nu = lattice_viscosity_from_relaxation_rate(self.omega)
        up = u_m / u_t[0]

        y = 0.5
        y_plus = (y * u_t[0]) / nu

        spaldings_law = up + sp.exp(-k * B) * (
                sp.exp(k * up) - 1 - (k * up) - ((k * up) ** 2) / 2 - ((k * up) ** 3) / 6) - y_plus

        m = -spaldings_law / spaldings_law.diff(u_t[0])

        init_guess = Assignment(u_t[0], u_m)

        newton_assignmets = []
        for i in range(newton_steps):
            newton_assignmets.append(Assignment(delta[i], m.subs({u_t[0]: u_t[i]})))
            newton_assignmets.append(Assignment(u_t[i + 1], u_t[i] + delta[i]))

        shear_stress = Assignment(tau_w, u_t[newton_steps]**2 * rho)

        result = eq_equations.all_assignments
        result.append(u_mag)
        result.append(init_guess)
        result.extend(newton_assignmets)
        result.append(shear_stress)

        result.append(Assignment(tau_w_x, - u[0] / (sp.sqrt(u[0]**2 + u[2]**2)) * tau_w))
        result.append(Assignment(tau_w_z, - u[2] / (sp.sqrt(u[0] ** 2 + u[2] ** 2)) * tau_w))

        neighbor_offset = NeighbourOffsetArrays.neighbour_offset(dir_symbol, lb_method.stencil)
        drag = neighbor_offset[0] * factor * tau_w_x + neighbor_offset[2] * factor * tau_w_z

        result.append(Assignment(f_in(inv_dir[dir_symbol]), f_in[normal_direction](mirrored_direction) * drag))

        return result
