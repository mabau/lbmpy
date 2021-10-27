from types import MappingProxyType

import numpy as np
import sympy as sp

from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.phasefield.analytical import (
    chemical_potentials_from_free_energy, symmetric_tensor_linearization)
from lbmpy.phasefield.cahn_hilliard_lbm import cahn_hilliard_lb_method
from lbmpy.phasefield.kerneleqs import (
    CahnHilliardFDStep, force_kernel_using_pressure_tensor, mu_kernel, pressure_tensor_kernel)
from pystencils import create_data_handling, create_kernel
from pystencils.boundaries.boundaryhandling import FlagInterface
from pystencils.boundaries.inkernel import add_neumann_boundary
from pystencils.enums import Target
from pystencils.simp import sympy_cse_on_assignment_list
from pystencils.slicing import SlicedGetter, make_slice


class PhaseFieldStep:

    def __init__(self, free_energy, order_parameters, domain_size=None, data_handling=None,
                 name='pf', hydro_lbm_parameters=MappingProxyType({}),
                 hydro_dynamic_relaxation_rate=1.0,
                 cahn_hilliard_relaxation_rates=1.0,
                 cahn_hilliard_gammas=1,
                 density_order_parameter=None,
                 optimization=None, kernel_params=MappingProxyType({}),
                 dx=1, dt=1, solve_cahn_hilliard_with_finite_differences=False,
                 order_parameter_force=None,
                 transformation_matrix=None,
                 concentration_to_order_parameters=None,
                 order_parameters_to_concentrations=None,
                 homogeneous_neumann_boundaries=False,
                 discretization='standard'):

        if optimization is None:
            optimization = {'openmp': False, 'target': Target.CPU}
        openmp = optimization.get('openmp', False)
        target = optimization.get('target', Target.CPU)

        if data_handling is None:
            data_handling = create_data_handling(domain_size, periodicity=True, parallel=False)

        self.free_energy = free_energy

        self.concentration_to_order_parameter = concentration_to_order_parameters
        self.order_parameters_to_concentrations = order_parameters_to_concentrations
        if transformation_matrix:
            op_transformation = np.array(transformation_matrix).astype(float)
            op_transformation_inv = np.array(transformation_matrix.inv()).astype(float)
            axes = ([-1], [1])
            if self.concentration_to_order_parameter is None:
                self.concentration_to_order_parameter = lambda c: np.tensordot(c, op_transformation, axes=axes)
            if self.order_parameters_to_concentrations is None:
                self.order_parameters_to_concentrations = lambda p: np.tensordot(p, op_transformation_inv, axes=axes)

        self.chemical_potentials = chemical_potentials_from_free_energy(free_energy, order_parameters)

        # ------------------ Adding arrays ---------------------
        gpu = target == Target.GPU
        self.gpu = gpu
        self.num_order_parameters = len(order_parameters)
        self.order_parameters = order_parameters
        pressure_tensor_size = len(symmetric_tensor_linearization(data_handling.dim))

        self.name = name
        self.phi_field_name = name + "_phi"
        self.mu_field_name = name + "_mu"
        self.vel_field_name = name + "_u"
        self.force_field_name = name + "_F"
        self.pressure_tensor_field_name = name + "_P"
        self.data_handling = data_handling

        dh = self.data_handling
        phi_size = len(order_parameters)
        self.phi_field = dh.add_array(self.phi_field_name, values_per_cell=phi_size, gpu=gpu, latex_name='φ')
        self.mu_field = dh.add_array(self.mu_field_name, values_per_cell=phi_size, gpu=gpu, latex_name="μ")
        self.vel_field = dh.add_array(self.vel_field_name, values_per_cell=data_handling.dim, gpu=gpu, latex_name="u")
        self.force_field = dh.add_array(self.force_field_name, values_per_cell=dh.dim, gpu=gpu, latex_name="F")
        self.pressure_tensor_field = data_handling.add_array(self.pressure_tensor_field_name, gpu=gpu,
                                                             values_per_cell=pressure_tensor_size, latex_name='P')
        self.flag_interface = FlagInterface(data_handling, 'flags')

        # ------------------ Creating kernels ------------------
        phi = tuple(self.phi_field(i) for i in range(len(order_parameters)))
        F = self.free_energy.subs({old: new for old, new in zip(order_parameters, phi)})

        if homogeneous_neumann_boundaries:
            def apply_neumann_boundaries(eqs):
                fields = [data_handling.fields[self.phi_field_name],
                          data_handling.fields[self.pressure_tensor_field_name]]
                flag_field = data_handling.fields[self.flag_interface.flag_field_name]
                return add_neumann_boundary(eqs, fields, flag_field, "neumann_flag", inverse_flag=False)
        else:
            def apply_neumann_boundaries(eqs):
                return eqs

        # μ and pressure tensor update
        self.phi_sync = data_handling.synchronization_function([self.phi_field_name], target=target)
        self.mu_eqs = mu_kernel(F, phi, self.phi_field, self.mu_field, dx)

        self.pressure_tensor_eqs = pressure_tensor_kernel(self.free_energy, order_parameters,
                                                          self.phi_field, self.pressure_tensor_field, dx=dx,
                                                          discretization=discretization)
        mu_and_pressure_tensor_eqs = self.mu_eqs + self.pressure_tensor_eqs
        mu_and_pressure_tensor_eqs = apply_neumann_boundaries(mu_and_pressure_tensor_eqs)
        self.mu_and_pressure_tensor_kernel = create_kernel(sympy_cse_on_assignment_list(mu_and_pressure_tensor_eqs),
                                                           target=target, cpu_openmp=openmp).compile()

        # F Kernel
        extra_force = sp.Matrix([0] * self.data_handling.dim)
        if order_parameter_force is not None:
            for order_parameter_idx, force in order_parameter_force.items():
                extra_force += sp.Matrix([f_i * self.phi_field(order_parameter_idx) for f_i in force])
        self.force_eqs = force_kernel_using_pressure_tensor(self.force_field, self.pressure_tensor_field, dx=dx,
                                                            extra_force=extra_force, discretization=discretization)
        self.force_from_pressure_tensor_kernel = create_kernel(apply_neumann_boundaries(self.force_eqs),
                                                               target=target, cpu_openmp=openmp).compile()
        self.pressure_tensor_sync = data_handling.synchronization_function([self.pressure_tensor_field_name],
                                                                           target=target)

        hydro_lbm_parameters = hydro_lbm_parameters.copy()
        # Hydrodynamic LBM
        if density_order_parameter is not None:
            density_idx = order_parameters.index(density_order_parameter)
            hydro_lbm_parameters['compute_density_in_every_step'] = True
            hydro_lbm_parameters['density_data_name'] = self.phi_field_name
            hydro_lbm_parameters['density_data_index'] = density_idx

        if 'optimization' not in hydro_lbm_parameters:
            hydro_lbm_parameters['optimization'] = optimization
        else:
            hydro_lbm_parameters['optimization'].update(optimization)

        self.hydro_lbm_step = LatticeBoltzmannStep(data_handling=data_handling, name=name + '_hydroLBM',
                                                   relaxation_rate=hydro_dynamic_relaxation_rate,
                                                   compute_velocity_in_every_step=True,
                                                   force=tuple([self.force_field(i) for i in range(dh.dim)]),
                                                   velocity_data_name=self.vel_field_name, kernel_params=kernel_params,
                                                   flag_interface=self.flag_interface,
                                                   time_step_order='collide_stream',
                                                   **hydro_lbm_parameters)

        # Cahn-Hilliard LBMs
        if not hasattr(cahn_hilliard_relaxation_rates, '__len__'):
            cahn_hilliard_relaxation_rates = [cahn_hilliard_relaxation_rates] * len(order_parameters)

        if not hasattr(cahn_hilliard_gammas, '__len__'):
            cahn_hilliard_gammas = [cahn_hilliard_gammas] * len(order_parameters)

        self.cahn_hilliard_steps = []

        if solve_cahn_hilliard_with_finite_differences:
            if density_order_parameter is not None:
                raise NotImplementedError("density_order_parameter not supported when "
                                          "CH is solved with finite differences")
            ch_step = CahnHilliardFDStep(self.data_handling, self.phi_field_name, self.mu_field_name,
                                         self.vel_field_name, target=target, dx=dx, dt=dt, mobilities=1,
                                         equation_modifier=apply_neumann_boundaries)
            self.cahn_hilliard_steps.append(ch_step)
        else:
            for i, op in enumerate(order_parameters):
                if op == density_order_parameter:
                    continue

                ch_method = cahn_hilliard_lb_method(self.hydro_lbm_step.method.stencil, self.mu_field(i),
                                                    relaxation_rate=cahn_hilliard_relaxation_rates[i],
                                                    gamma=cahn_hilliard_gammas[i])
                ch_step = LatticeBoltzmannStep(data_handling=data_handling, relaxation_rate=1, lb_method=ch_method,
                                               velocity_input_array_name=self.vel_field.name,
                                               density_data_name=self.phi_field.name,
                                               stencil='D3Q19' if self.data_handling.dim == 3 else 'D2Q9',
                                               compute_density_in_every_step=True,
                                               density_data_index=i,
                                               flag_interface=self.hydro_lbm_step.boundary_handling.flag_interface,
                                               name=name + f"_chLbm_{i}",
                                               optimization=optimization)
                self.cahn_hilliard_steps.append(ch_step)

        self._vtk_writer = None
        self.run_hydro_lbm = True
        self.density_order_parameter = density_order_parameter
        self.time_steps_run = 0
        self.reset()

        self.neumann_flag = 0

    @property
    def vtk_writer(self):
        if self._vtk_writer is None:
            self._vtk_writer = self.data_handling.create_vtk_writer(self.name, [self.phi_field_name,
                                                                                self.mu_field_name,
                                                                                self.vel_field_name,
                                                                                self.force_field_name])

        return self._vtk_writer

    @property
    def shape(self):
        return self.data_handling.shape

    def write_vtk(self):
        self.vtk_writer(self.time_steps_run)

    def reset(self):
        # Init φ and μ
        self.data_handling.fill(self.phi_field_name, 0.0)
        self.data_handling.fill(self.phi_field_name, 1.0 if self.density_order_parameter is not None else 0.0,
                                value_idx=0)
        self.data_handling.fill(self.mu_field_name, 0.0)
        self.data_handling.fill(self.force_field_name, 0.0)
        self.data_handling.fill(self.vel_field_name, 0.0)
        self.set_pdf_fields_from_macroscopic_values()

        self.time_steps_run = 0

    def set_pdf_fields_from_macroscopic_values(self):
        self.hydro_lbm_step.set_pdf_fields_from_macroscopic_values()
        for ch_step in self.cahn_hilliard_steps:
            ch_step.set_pdf_fields_from_macroscopic_values()

    def pre_run(self):
        if self.gpu:
            self.data_handling.to_gpu(self.phi_field_name)
            self.data_handling.to_gpu(self.mu_field_name)
            self.data_handling.to_gpu(self.force_field_name)
        self.hydro_lbm_step.pre_run()
        for ch_step in self.cahn_hilliard_steps:
            ch_step.pre_run()

    def post_run(self):
        if self.gpu:
            self.data_handling.to_cpu(self.phi_field_name)
            self.data_handling.to_cpu(self.mu_field_name)
            self.data_handling.to_cpu(self.force_field_name)
        if self.run_hydro_lbm:
            self.hydro_lbm_step.post_run()
        for ch_step in self.cahn_hilliard_steps:
            ch_step.post_run()

    def time_step(self):
        neumann_flag = self.neumann_flag

        self.phi_sync()
        self.data_handling.run_kernel(self.mu_and_pressure_tensor_kernel, neumann_flag=neumann_flag)
        self.pressure_tensor_sync()
        self.data_handling.run_kernel(self.force_from_pressure_tensor_kernel, neumann_flag=neumann_flag)

        if self.run_hydro_lbm:
            self.hydro_lbm_step.time_step()

        for ch_lbm in self.cahn_hilliard_steps:
            ch_lbm.time_step()

        self.time_steps_run += 1

    @property
    def boundary_handling(self):
        return self.hydro_lbm_step.boundary_handling

    def set_concentration(self, slice_obj, concentration):
        if self.concentration_to_order_parameter is not None:
            phi = self.concentration_to_order_parameter(concentration)
        else:
            phi = np.array(concentration)

        for b in self.data_handling.iterate(slice_obj):
            for i in range(phi.shape[-1]):
                b[self.phi_field_name][..., i] = phi[i]

    def set_single_concentration(self, slice_obj, phase_idx, value=1):
        num_phases = self.concentration[:, :, :].shape[-1]
        zeros = [0] * num_phases
        zeros[phase_idx] = value
        self.set_concentration(slice_obj, zeros)

    def set_density(self, slice_obj, value):
        for b in self.data_handling.iterate(slice_obj):
            for i in range(self.num_order_parameters):
                b[self.hydro_lbm_step.density_data_name].fill(value)

    def run(self, time_steps):
        self.pre_run()
        for i in range(time_steps):
            self.time_step()
        self.post_run()

    def _get_slice(self, data_name, slice_obj):
        if slice_obj is None:
            slice_obj = make_slice[:, :] if self.data_handling.dim == 2 else make_slice[:, :, 0.5]
        return self.data_handling.gather_array(data_name, slice_obj).squeeze()

    def phi_slice(self, slice_obj=None):
        return self._get_slice(self.phi_field_name, slice_obj)

    def concentration_slice(self, slice_obj=None):
        if slice_obj is not None and len(slice_obj) > self.data_handling.dim:
            assert len(slice_obj) - 1 == self.data_handling.dim
            last_index = slice_obj[-1]
            slice_obj = slice_obj[:-1]
        else:
            last_index = None

        phi = self.phi_slice(slice_obj)
        result = self.order_parameters_to_concentrations(phi) if self.order_parameters_to_concentrations else phi
        if last_index is not None:
            result = result[..., last_index]
        return result

    def mu_slice(self, slice_obj=None):
        return self._get_slice(self.mu_field_name, slice_obj)

    def velocity_slice(self, slice_obj=None):
        return self._get_slice(self.vel_field_name, slice_obj)

    def force_slice(self, slice_obj=None):
        return self._get_slice(self.force_field_name, slice_obj)

    @property
    def phi(self):
        return SlicedGetter(self.phi_slice)

    @property
    def concentration(self):
        return SlicedGetter(self.concentration_slice)

    @property
    def mu(self):
        return SlicedGetter(self.mu_slice)

    @property
    def velocity(self):
        return SlicedGetter(self.velocity_slice)

    @property
    def force(self):
        return SlicedGetter(self.force_slice)
