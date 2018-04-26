import sympy as sp
from pystencils import Assignment
from pystencils.fd import Discretization2ndOrder
from lbmpy.phasefield.analytical import chemical_potentials_from_free_energy, substitute_laplacian_by_sum, \
    force_from_phi_and_mu, symmetric_tensor_linearization, pressure_tensor_from_free_energy, force_from_pressure_tensor


# ---------------------------------- Kernels to compute force ----------------------------------------------------------


def mu_kernel(free_energy, order_parameters, phi_field, mu_field, dx=1):
    """Reads from order parameter (phi) field and updates chemical potentials"""
    assert phi_field.spatial_dimensions == mu_field.spatial_dimensions
    dim = phi_field.spatial_dimensions
    chemical_potential = chemical_potentials_from_free_energy(free_energy, order_parameters)
    chemical_potential = substitute_laplacian_by_sum(chemical_potential, dim)
    chemical_potential = chemical_potential.subs({op: phi_field(i) for i, op in enumerate(order_parameters)})
    discretize = Discretization2ndOrder(dx=dx)
    return [Assignment(mu_field(i), discretize(mu_i)) for i, mu_i in enumerate(chemical_potential)]


def force_kernel_using_mu(force_field, phi_field, mu_field, dx=1):
    """Computes forces using precomputed chemical potential - needs mu_kernel first"""
    assert mu_field.index_dimensions == 1
    force = force_from_phi_and_mu(phi_field.center_vector, mu=mu_field.center_vector, dim=mu_field.spatial_dimensions)
    discretize = Discretization2ndOrder(dx=dx)
    return [Assignment(force_field(i),
                       discretize(f_i)).expand() for i, f_i in enumerate(force)]


def pressure_tensor_kernel(free_energy, order_parameters, phi_field, pressure_tensor_field, dx=1):
    dim = phi_field.spatial_dimensions
    p = pressure_tensor_from_free_energy(free_energy, order_parameters, dim)
    p = p.subs({op: phi_field(i) for i, op in enumerate(order_parameters)})
    index_map = symmetric_tensor_linearization(dim)
    discretize = Discretization2ndOrder(dx=dx)
    eqs = []
    for index, lin_index in index_map.items():
        eq = Assignment(pressure_tensor_field(lin_index), discretize(p[index]).expand())
        eqs.append(eq)
    return eqs


def force_kernel_using_pressure_tensor(force_field, pressure_tensor_field, extra_force=None, dx=1):
    dim = force_field.spatial_dimensions
    index_map = symmetric_tensor_linearization(dim)

    p = sp.Matrix(dim, dim, lambda i, j: pressure_tensor_field(index_map[i, j] if i < j else index_map[j, i]))
    f = force_from_pressure_tensor(p)
    if extra_force:
        f += extra_force
    discretize = Discretization2ndOrder(dx=dx)
    return [Assignment(force_field(i), discretize(f_i).expand())
            for i, f_i in enumerate(f)]


# ---------------------------------- Cahn Hilliard with finite differences ---------------------------------------------


def cahn_hilliard_fd_eq(phase_idx, phi, mu, velocity, mobility, dx, dt):
    from pystencils.fd import transient, advection, diffusion
    cahn_hilliard = transient(phi, phase_idx) + advection(phi, velocity, phase_idx) - diffusion(mu, mobility, phase_idx)
    return Discretization2ndOrder(dx, dt)(cahn_hilliard)


class CahnHilliardFDStep:
    def __init__(self, data_handling, phi_field_name, mu_field_name, velocity_field_name, name='ch_fd', target='cpu',
                 dx=1, dt=1, mobilities=1, equation_modifier=lambda eqs: eqs):
        from pystencils import create_kernel
        self.data_handling = data_handling

        mu_field = self.data_handling.fields[mu_field_name]
        vel_field = self.data_handling.fields[velocity_field_name]
        self.phi_field = self.data_handling.fields[phi_field_name]
        self.tmp_field = self.data_handling.add_array_like(name + '_tmp', phi_field_name, latex_name='tmp')

        num_phases = self.data_handling.values_per_cell(phi_field_name)
        if not hasattr(mobilities, '__len__'):
            mobilities = [mobilities] * num_phases

        update_eqs = []
        for i in range(num_phases):
            rhs = cahn_hilliard_fd_eq(i, self.phi_field, mu_field, vel_field, mobilities[i], dx, dt)
            update_eqs.append(Assignment(self.tmp_field(i), rhs))
        self.update_eqs = update_eqs
        self.update_eqs = equation_modifier(update_eqs)
        self.kernel = create_kernel(self.update_eqs, target=target).compile()
        self.sync = self.data_handling.synchronization_function([phi_field_name, velocity_field_name, mu_field_name],
                                                                target=target)

    def time_step(self, **kwargs):
        self.sync()
        self.data_handling.run_kernel(self.kernel, **kwargs)
        self.data_handling.swap(self.phi_field.name, self.tmp_field.name)

    def set_pdf_fields_from_macroscopic_values(self):
        pass

    def pre_run(self):
        pass

    def post_run(self):
        pass
