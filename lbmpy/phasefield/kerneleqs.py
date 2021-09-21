import sympy as sp

from lbmpy.phasefield.analytical import (
    chemical_potentials_from_free_energy, force_from_phi_and_mu, force_from_pressure_tensor,
    pressure_tensor_bulk_sqrt_term, pressure_tensor_from_free_energy, substitute_laplacian_by_sum,
    symmetric_tensor_linearization)
from pystencils import Assignment, Target
from pystencils.fd import Discretization2ndOrder, discretize_spatial

# ---------------------------------- Kernels to compute force ----------------------------------------------------------


def mu_kernel(free_energy, order_parameters, phi_field, mu_field, dx=1, discretization='standard'):
    """Reads from order parameter (phi) field and updates chemical potentials"""
    assert phi_field.spatial_dimensions == mu_field.spatial_dimensions
    dim = phi_field.spatial_dimensions
    chemical_potential = chemical_potentials_from_free_energy(free_energy, order_parameters)
    chemical_potential = substitute_laplacian_by_sum(chemical_potential, dim)
    chemical_potential = chemical_potential.subs({op: phi_field(i) for i, op in enumerate(order_parameters)})
    return [Assignment(mu_field(i), discretize_spatial(mu_i, dx, discretization))
            for i, mu_i in enumerate(chemical_potential)]


def force_kernel_using_mu(force_field, phi_field, mu_field, dx=1, discretization='standard'):
    """Computes forces using precomputed chemical potential - needs mu_kernel first"""
    assert mu_field.index_dimensions == 1
    force = force_from_phi_and_mu(phi_field.center_vector, mu=mu_field.center_vector, dim=mu_field.spatial_dimensions)
    return [Assignment(force_field(i),
                       discretize_spatial(f_i, dx, discretization)).expand() for i, f_i in enumerate(force)]


def pressure_tensor_kernel(free_energy, order_parameters, phi_field, pressure_tensor_field,
                           dx=1, discretization='standard', bulk_chemical_potential=None):
    dim = phi_field.spatial_dimensions

    p = pressure_tensor_from_free_energy(free_energy, order_parameters, dim,
                                         bulk_chemical_potential=bulk_chemical_potential)

    p = p.subs({op: phi_field(i) for i, op in enumerate(order_parameters)})
    index_map = symmetric_tensor_linearization(dim)
    eqs = []
    for index, lin_index in index_map.items():
        eq = Assignment(pressure_tensor_field(lin_index), discretize_spatial(p[index], dx, discretization).expand())
        eqs.append(eq)
    return eqs


def pressure_tensor_kernel_pbs(free_energy, order_parameters,
                               phi_field, pressure_tensor_field, pbs_field, density_field,
                               transformation_matrix=None, dx=1, discretization='standard'):
    dim = phi_field.spatial_dimensions

    p = pressure_tensor_from_free_energy(free_energy, order_parameters, dim, transformation_matrix, include_bulk=False)
    assert transformation_matrix is None

    p = p.subs({op: phi_field(i) for i, op in enumerate(order_parameters)})
    index_map = symmetric_tensor_linearization(dim)
    eqs = []
    for index, lin_index in index_map.items():
        eq = Assignment(pressure_tensor_field(lin_index), discretize_spatial(p[index], dx, discretization).expand())
        eqs.append(eq)

    rhs = pressure_tensor_bulk_sqrt_term(free_energy, order_parameters, density_field.center)
    pbs = Assignment(pbs_field.center, discretize_spatial(rhs, dx, discretization))

    eqs.append(pbs)
    return eqs


def force_kernel_using_pressure_tensor(force_field, pressure_tensor_field, extra_force=None,
                                       pbs=None, dx=1, discretization='standard'):
    dim = force_field.spatial_dimensions
    index_map = symmetric_tensor_linearization(dim)

    p = sp.Matrix(dim, dim, lambda i, j: pressure_tensor_field(index_map[i, j] if i < j else index_map[j, i]))
    f = force_from_pressure_tensor(p, pbs=pbs)
    if extra_force:
        f += extra_force
    return [Assignment(force_field(i), discretize_spatial(f_i, dx, discretization).expand())
            for i, f_i in enumerate(f)]


# ---------------------------------- Cahn Hilliard with finite differences ---------------------------------------------


def cahn_hilliard_fd_eq(phase_idx, phi, mu, velocity, mobility, dx, dt):
    from pystencils.fd import transient, advection, diffusion
    cahn_hilliard = transient(phi, phase_idx) + advection(phi, velocity, phase_idx) - diffusion(mu, mobility, phase_idx)
    return Discretization2ndOrder(dx, dt)(cahn_hilliard)


class CahnHilliardFDStep:
    def __init__(self, data_handling, phi_field_name, mu_field_name, velocity_field_name, name='ch_fd',
                 target=Target.CPU, dx=1, dt=1, mobilities=1, equation_modifier=lambda eqs: eqs):
        from pystencils import create_kernel
        self.data_handling = data_handling

        mu_field = self.data_handling.fields[mu_field_name]
        vel_field = self.data_handling.fields[velocity_field_name]
        self.phi_field = self.data_handling.fields[phi_field_name]
        self.tmp_field = self.data_handling.add_array_like(name + '_tmp', phi_field_name, latex_name='tmp')

        num_phases = self.phi_field.index_shape[0]
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
