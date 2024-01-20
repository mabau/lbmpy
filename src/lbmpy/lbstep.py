from types import MappingProxyType
from dataclasses import replace

import numpy as np

from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.creationfunctions import (create_lb_function, update_with_default_parameters)
from lbmpy.enums import Stencil
from lbmpy.macroscopic_value_kernels import (
    create_advanced_velocity_setter_collision_rule, pdf_initialization_assignments)
from lbmpy.simplificationfactory import create_simplification_strategy
from lbmpy.stencils import LBStencil
from pystencils import create_data_handling, create_kernel, make_slice, Target, Backend
from pystencils.slicing import SlicedGetter
from pystencils.timeloop import TimeLoop


class LatticeBoltzmannStep:

    def __init__(self, domain_size=None, lbm_kernel=None, periodicity=False,
                 kernel_params=MappingProxyType({}), data_handling=None, name="lbm", optimization=None,
                 velocity_data_name=None, density_data_name=None, density_data_index=None,
                 compute_velocity_in_every_step=False, compute_density_in_every_step=False,
                 velocity_input_array_name=None, time_step_order='stream_collide', flag_interface=None,
                 alignment_if_vectorized=64, fixed_loop_sizes=True,
                 timeloop_creation_function=TimeLoop,
                 lbm_config=None, lbm_optimisation=None, config=None, **method_parameters):

        if optimization is None:
            optimization = {}
        self._timeloop_creation_function = timeloop_creation_function

        # --- Parameter normalization  ---
        if data_handling is not None:
            if domain_size is not None:
                raise ValueError("When passing a data_handling, the domain_size parameter can not be specified")

        if config is not None:
            target = config.target
        else:
            target = optimization.get('target', Target.CPU)

        if data_handling is None:
            if domain_size is None:
                raise ValueError("Specify either domain_size or data_handling")
            data_handling = create_data_handling(domain_size,
                                                 default_ghost_layers=1,
                                                 periodicity=periodicity,
                                                 default_target=target,
                                                 parallel=False)

        if 'stencil' not in method_parameters:
            method_parameters['stencil'] = LBStencil(Stencil.D2Q9) \
                if data_handling.dim == 2 else LBStencil(Stencil.D3Q27)

        lbm_config, lbm_optimisation, config = update_with_default_parameters(method_parameters, optimization,
                                                                              lbm_config, lbm_optimisation, config)

        # the parallel datahandling understands only numpy datatypes. Strings lead to an errors
        field_dtype = config.data_type.default_factory().numpy_dtype

        if lbm_kernel:
            q = lbm_kernel.method.stencil.Q
        else:
            q = lbm_config.stencil.Q

        self.name = name
        self._data_handling = data_handling
        self._pdf_arr_name = name + "_pdfSrc"
        self._tmp_arr_name = name + "_pdfTmp"
        self.velocity_data_name = name + "_velocity" if velocity_data_name is None else velocity_data_name
        self.density_data_name = name + "_density" if density_data_name is None else density_data_name
        self.density_data_index = density_data_index

        self._gpu = target == Target.GPU
        layout = lbm_optimisation.field_layout

        alignment = False
        if config.backend == Backend.C and config.cpu_vectorize_info:
            alignment = alignment_if_vectorized

        self._data_handling.add_array(self._pdf_arr_name, values_per_cell=q, gpu=self._gpu, layout=layout,
                                      latex_name='src', dtype=field_dtype, alignment=alignment)
        self._data_handling.add_array(self._tmp_arr_name, values_per_cell=q, gpu=self._gpu, cpu=not self._gpu,
                                      layout=layout, latex_name='dst', dtype=field_dtype, alignment=alignment)

        if velocity_data_name is None:
            self._data_handling.add_array(self.velocity_data_name, values_per_cell=self._data_handling.dim,
                                          gpu=self._gpu and compute_velocity_in_every_step,
                                          layout=layout, latex_name='u', dtype=field_dtype, alignment=alignment)
        if density_data_name is None:
            self._data_handling.add_array(self.density_data_name, values_per_cell=1,
                                          gpu=self._gpu and compute_density_in_every_step,
                                          layout=layout, latex_name='ρ', dtype=field_dtype, alignment=alignment)

        if compute_velocity_in_every_step:
            lbm_config.output['velocity'] = self._data_handling.fields[self.velocity_data_name]
        if compute_density_in_every_step:
            density_field = self._data_handling.fields[self.density_data_name]
            if self.density_data_index is not None:
                density_field = density_field(density_data_index)
            lbm_config.output['density'] = density_field
        if velocity_input_array_name is not None:
            lbm_config = replace(lbm_config, velocity_input=self._data_handling.fields[velocity_input_array_name])
        if isinstance(lbm_config.omega_output_field, str):
            lbm_config = replace(lbm_config, omega_output_field=data_handling.add_array(lbm_config.omega_output_field,
                                                                                        dtype=field_dtype,
                                                                                        alignment=alignment,
                                                                                        values_per_cell=1))

        self.kernel_params = kernel_params.copy()

        # --- Kernel creation ---
        if lbm_kernel is None:

            if fixed_loop_sizes:
                lbm_optimisation = replace(lbm_optimisation, symbolic_field=data_handling.fields[self._pdf_arr_name])
            lbm_config = replace(lbm_config, field_name=self._pdf_arr_name)
            lbm_config = replace(lbm_config, temporary_field_name=self._tmp_arr_name)

            if time_step_order == 'stream_collide':
                self._lbmKernels = [create_lb_function(lbm_config=lbm_config,
                                                       lbm_optimisation=lbm_optimisation,
                                                       config=config)]
            elif time_step_order == 'collide_stream':
                self._lbmKernels = [create_lb_function(lbm_config=lbm_config,
                                                       lbm_optimisation=lbm_optimisation,
                                                       config=config,
                                                       kernel_type='collide_only'),
                                    create_lb_function(lbm_config=lbm_config,
                                                       lbm_optimisation=lbm_optimisation,
                                                       config=config,
                                                       kernel_type='stream_pull_only')]

        else:
            assert self._data_handling.dim == lbm_kernel.method.dim, \
                f"Error: {lbm_kernel.method.dim}D Kernel for {self._data_handling.dim} dimensional domain"
            self._lbmKernels = [lbm_kernel]

        self.method = self._lbmKernels[0].method
        self.ast = self._lbmKernels[0].ast

        # -- Boundary Handling  & Synchronization ---
        stencil_name = lbm_config.stencil.name
        self._sync_src = data_handling.synchronization_function([self._pdf_arr_name], stencil_name, target,
                                                                stencil_restricted=True)
        self._sync_tmp = data_handling.synchronization_function([self._tmp_arr_name], stencil_name, target,
                                                                stencil_restricted=True)

        self._boundary_handling = LatticeBoltzmannBoundaryHandling(self.method, self._data_handling, self._pdf_arr_name,
                                                                   name=name + "_boundary_handling",
                                                                   flag_interface=flag_interface,
                                                                   target=target, openmp=config.cpu_openmp)

        self._lbm_config = lbm_config
        self._lbm_optimisation = lbm_optimisation
        self._config = config

        # -- Macroscopic Value Kernels
        self._getterKernel, self._setterKernel = self._compile_macroscopic_setter_and_getter()

        self._data_handling.fill(self.density_data_name, 1.0, value_idx=self.density_data_index,
                                 ghost_layers=True, inner_ghost_layers=True)
        self._data_handling.fill(self.velocity_data_name, 0.0, ghost_layers=True, inner_ghost_layers=True)
        self.set_pdf_fields_from_macroscopic_values()

        # -- VTK output
        self._vtk_writer = None
        self.time_steps_run = 0

        self._velocity_init_kernel = None
        self._velocity_init_vel_backup = None

    @property
    def boundary_handling(self):
        """Boundary handling instance of the scenario. Use this to change the boundary setup"""
        return self._boundary_handling

    @property
    def data_handling(self):
        return self._data_handling

    @property
    def vtk_writer(self):
        if self._vtk_writer is None:
            # Create vtk writer on demand - otherwise output folders are created even if no vtk output is done
            self._vtk_writer = self.data_handling.create_vtk_writer(self.name,
                                                                    [self.velocity_data_name, self.density_data_name])
        return self._vtk_writer

    @property
    def dim(self):
        return self._data_handling.dim

    @property
    def domain_size(self):
        return self._data_handling.shape

    @property
    def number_of_cells(self):
        result = 1
        for d in self.domain_size:
            result *= d
        return result

    @property
    def pdf_array_name(self):
        return self._pdf_arr_name

    @property
    def lbm_config(self):
        """LBM configuration of the scenario"""
        return self._lbm_config

    @property
    def lbm_optimisation(self):
        """LBM optimisation parameters"""
        return self._lbm_optimisation

    @property
    def config(self):
        """Configutation of pystencils parameters"""
        return self.config

    def _get_slice(self, data_name, slice_obj, masked):
        if slice_obj is None:
            slice_obj = make_slice[:, :] if self.dim == 2 else make_slice[:, :, 0.5]

        result = self._data_handling.gather_array(data_name, slice_obj)

        if masked:
            mask = self.boundary_handling.get_mask(slice_obj[:self.dim], 'domain', True)
            if result is not None:
                if len(mask.shape) < len(result.shape):
                    assert len(mask.shape) + 1 == len(result.shape)
                    mask = np.repeat(mask[..., np.newaxis], result.shape[-1], axis=2)

                result = np.ma.masked_array(result, mask).squeeze()
        return result

    def velocity_slice(self, slice_obj=None, masked=True):
        return self._get_slice(self.velocity_data_name, slice_obj, masked)

    def density_slice(self, slice_obj=None, masked=True):
        if self.density_data_index is not None:
            slice_obj += (self.density_data_index,)
        return self._get_slice(self.density_data_name, slice_obj, masked)

    @property
    def velocity(self):
        return SlicedGetter(self.velocity_slice)

    @property
    def density(self):
        return SlicedGetter(self.density_slice)

    def pre_run(self):
        if self._gpu:
            self._data_handling.to_gpu(self._pdf_arr_name)
            if self._data_handling.is_on_gpu(self.velocity_data_name):
                self._data_handling.to_gpu(self.velocity_data_name)
            if self._data_handling.is_on_gpu(self.density_data_name):
                self._data_handling.to_gpu(self.density_data_name)

    def set_pdf_fields_from_macroscopic_values(self):
        self._data_handling.run_kernel(self._setterKernel, **self.kernel_params)

    def time_step(self):
        if len(self._lbmKernels) == 2:  # collide stream
            self._data_handling.run_kernel(self._lbmKernels[0], **self.kernel_params)
            self._sync_src()
            self._boundary_handling(**self.kernel_params)
            self._data_handling.run_kernel(self._lbmKernels[1], **self.kernel_params)
        else:  # stream collide
            self._sync_src()
            self._boundary_handling(**self.kernel_params)
            self._data_handling.run_kernel(self._lbmKernels[0], **self.kernel_params)

        self._data_handling.swap(self._pdf_arr_name, self._tmp_arr_name, self._gpu)

    def get_time_loop(self):
        self.pre_run()  # make sure GPU arrays are allocated

        fixed_loop = self._timeloop_creation_function(steps=2)
        fixed_loop.add_pre_run_function(self.pre_run)
        fixed_loop.add_post_run_function(self.post_run)
        fixed_loop.add_single_step_function(self.time_step)

        for t in range(2):
            if len(self._lbmKernels) == 2:  # collide stream
                collide_args = self._data_handling.get_kernel_kwargs(self._lbmKernels[0], **self.kernel_params)
                fixed_loop.add_call(self._lbmKernels[0], collide_args)

                fixed_loop.add_call(self._sync_src if t == 0 else self._sync_tmp, {})
                self._boundary_handling.add_fixed_steps(fixed_loop, **self.kernel_params)

                stream_args = self._data_handling.get_kernel_kwargs(self._lbmKernels[1], **self.kernel_params)
                fixed_loop.add_call(self._lbmKernels[1], stream_args)
            else:  # stream collide
                fixed_loop.add_call(self._sync_src if t == 0 else self._sync_tmp, {})
                self._boundary_handling.add_fixed_steps(fixed_loop, **self.kernel_params)
                stream_collide_args = self._data_handling.get_kernel_kwargs(self._lbmKernels[0], **self.kernel_params)
                fixed_loop.add_call(self._lbmKernels[0], stream_collide_args)

            self._data_handling.swap(self._pdf_arr_name, self._tmp_arr_name, self._gpu)
        return fixed_loop

    def post_run(self):
        if self._gpu:
            self._data_handling.to_cpu(self._pdf_arr_name)
        self._data_handling.run_kernel(self._getterKernel, **self.kernel_params)

    def run(self, time_steps):
        time_loop = self.get_time_loop()
        time_loop.run(time_steps)
        self.time_steps_run += time_loop.time_steps_run

    def run_old(self, time_steps):
        self.pre_run()
        for i in range(time_steps):
            self.time_step()
        self.post_run()

        self.time_steps_run += time_steps

    def benchmark_run(self, time_steps, number_of_cells=None):
        if number_of_cells is None:
            number_of_cells = self.number_of_cells
        time_loop = self.get_time_loop()
        duration_of_time_step = time_loop.benchmark_run(time_steps)
        mlups = number_of_cells / duration_of_time_step * 1e-6
        self.time_steps_run += time_loop.time_steps_run
        return mlups

    def benchmark(self, time_for_benchmark=5, init_time_steps=2, number_of_time_steps_for_estimation='auto'):
        time_loop = self.get_time_loop()
        duration_of_time_step = time_loop.benchmark(time_for_benchmark, init_time_steps,
                                                    number_of_time_steps_for_estimation)
        mlups = self.number_of_cells / duration_of_time_step * 1e-6
        self.time_steps_run += time_loop.time_steps_run
        return mlups

    def write_vtk(self):
        self.vtk_writer(self.time_steps_run)

    def run_iterative_initialization(self, velocity_relaxation_rate=1.0, convergence_threshold=1e-5, max_steps=5000,
                                     check_residuum_after=100):
        """Runs Advanced initialization of velocity field through iteration procedure.

        Usually the pdfs are initialized in equilibrium with given density and velocity. Higher order moments are
        set to their equilibrium values. This routine also initializes the higher order moments and the density field
        using an iterative routine. For scenarios with high relaxation rates this might take long to converge.
        For details, see Mei, Luo, Lallemand and Humieres: Consistent initial conditions for LBM simulations, 2005.

        Args:
            velocity_relaxation_rate: relaxation rate for the velocity moments - determines convergence behaviour
                                      of the initialization scheme, should be in the range of the other relaxation
                                      rate(s) otherwise the scheme could get unstable
            convergence_threshold: The residuum is computed as average difference between prescribed and calculated
                                   velocity field. If (residuum < convergence_threshold) the function returns
                                   successfully.
            max_steps: stop if not converged after this number of steps
            check_residuum_after: the residuum criterion is tested after this number of steps

        Returns:
            tuple (residuum, steps_run) if successful or raises ValueError if not converged
        """
        dh = self.data_handling
        gpu = self._gpu

        def on_first_call():
            self._velocity_init_vel_backup = 'velocity_init_vel_backup'
            vel_backup_field = dh.add_array_like(self._velocity_init_vel_backup, self.velocity_data_name, cpu=True,
                                                 gpu=gpu)

            collision_rule = create_advanced_velocity_setter_collision_rule(self.method, vel_backup_field,
                                                                            velocity_relaxation_rate)
            self._lbm_optimisation.symbolic_field = dh.fields[self._pdf_arr_name]

            kernel = create_lb_function(collision_rule=collision_rule, field_name=self._pdf_arr_name,
                                        temporary_field_name=self._tmp_arr_name,
                                        lbm_optimisation=self._lbm_optimisation)
            self._velocity_init_kernel = kernel

        def make_velocity_backup():
            for b in dh.iterate():
                np.copyto(b[self._velocity_init_vel_backup], b[self.velocity_data_name])

        def restore_velocity_backup():
            for b in dh.iterate():
                np.copyto(b[self.velocity_data_name], b[self._velocity_init_vel_backup])

        def compute_residuum():
            residuum = 0
            for b in dh.iterate(ghost_layers=False, inner_ghost_layers=False):
                residuum = np.average(np.abs(b[self._velocity_init_vel_backup] - b[self.velocity_data_name]))
            reduce_result = dh.reduce_float_sequence([residuum, 1.0], 'sum', all_reduce=True)
            return reduce_result[0] / reduce_result[1]

        if self._velocity_init_kernel is None:
            on_first_call()

        make_velocity_backup()
        outer_iterations = max_steps // check_residuum_after
        global_residuum = None
        steps_run = 0
        for outer_iteration in range(outer_iterations):
            self._data_handling.all_to_gpu()
            for i in range(check_residuum_after):
                steps_run += 1
                self._sync_src()
                self._boundary_handling(**self.kernel_params)
                self._data_handling.run_kernel(self._velocity_init_kernel, **self.kernel_params)
                self._data_handling.swap(self._pdf_arr_name, self._tmp_arr_name, gpu=gpu)
            self._data_handling.all_to_cpu()
            self._data_handling.run_kernel(self._getterKernel, **self.kernel_params)
            global_residuum = compute_residuum()
            print(f"Initialization iteration {steps_run}, residuum {global_residuum}")
            if np.isnan(global_residuum) or global_residuum < convergence_threshold:
                break

        assert global_residuum is not None
        converged = global_residuum < convergence_threshold
        if not converged:
            restore_velocity_backup()
            raise ValueError(f"Iterative initialization did not converge after {steps_run} steps.\n"
                             f"Current residuum is {global_residuum}")

        return global_residuum, steps_run

    def _compile_macroscopic_setter_and_getter(self):
        lb_method = self.method
        cqc = lb_method.conserved_quantity_computation
        pdf_field = self._data_handling.fields[self._pdf_arr_name]
        rho_field = self._data_handling.fields[self.density_data_name]
        rho_field = rho_field.center if self.density_data_index is None else rho_field(self.density_data_index)
        vel_field = self._data_handling.fields[self.velocity_data_name]

        getter_eqs = cqc.output_equations_from_pdfs(pdf_field.center_vector,
                                                    {'density': rho_field, 'velocity': vel_field})
        getter_kernel = create_kernel(getter_eqs, target=Target.CPU, cpu_openmp=self._config.cpu_openmp).compile()

        setter_eqs = pdf_initialization_assignments(lb_method, rho_field,
                                                    vel_field.center_vector, pdf_field.center_vector)
        setter_eqs = create_simplification_strategy(lb_method)(setter_eqs)
        setter_kernel = create_kernel(setter_eqs, target=Target.CPU, cpu_openmp=self._config.cpu_openmp).compile()
        return getter_kernel, setter_kernel
