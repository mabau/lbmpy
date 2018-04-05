import numpy as np
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.creationfunctions import switch_to_symbolic_relaxation_rates_for_omega_adapting_methods, \
    create_lb_function, \
    update_with_default_parameters
from lbmpy.simplificationfactory import create_simplification_strategy
from lbmpy.stencils import get_stencil
from pystencils.datahandling.serial_datahandling import SerialDataHandling
from pystencils import create_kernel, make_slice
from pystencils.slicing import SlicedGetter
from pystencils.timeloop import TimeLoop


class LatticeBoltzmannStep:

    def __init__(self, domain_size=None, lbm_kernel=None, periodicity=False,
                 kernel_params={}, data_handling=None, name="lbm", optimization=None,
                 velocity_data_name=None, density_data_name=None, density_data_index=None,
                 compute_velocity_in_every_step=False, compute_density_in_every_step=False,
                 velocity_input_array_name=None, time_step_order='streamCollide', flag_interface=None,
                 **method_parameters):

        # --- Parameter normalization  ---
        if data_handling is not None:
            if domain_size is not None:
                raise ValueError("When passing a data_handling, the domain_size parameter can not be specified")

        if data_handling is None:
            if domain_size is None:
                raise ValueError("Specify either domain_size or data_handling")
            data_handling = SerialDataHandling(domain_size, default_ghost_layers=1, periodicity=periodicity)

        if 'stencil' not in method_parameters:
            method_parameters['stencil'] = 'D2Q9' if data_handling.dim == 2 else 'D3Q27'

        method_parameters, optimization = update_with_default_parameters(method_parameters, optimization)
        del method_parameters['kernel_type']

        if lbm_kernel:
            q = len(lbm_kernel.method.stencil)
        else:
            q = len(get_stencil(method_parameters['stencil']))
        target = optimization['target']

        self.name = name
        self._data_handling = data_handling
        self._pdf_arr_name = name + "_pdfSrc"
        self._tmp_arr_name = name + "_pdfTmp"
        self.velocity_data_name = name + "_velocity" if velocity_data_name is None else velocity_data_name
        self.density_data_name = name + "_density" if density_data_name is None else density_data_name
        self.density_data_index = density_data_index

        self._gpu = target == 'gpu'
        layout = optimization['field_layout']
        self._data_handling.add_array(self._pdf_arr_name, values_per_cell=q, gpu=self._gpu, layout=layout,
                                      latex_name='src')
        self._data_handling.add_array(self._tmp_arr_name, values_per_cell=q, gpu=self._gpu, cpu=not self._gpu,
                                      layout=layout, latex_name='dst')

        if velocity_data_name is None:
            self._data_handling.add_array(self.velocity_data_name, values_per_cell=self._data_handling.dim,
                                          gpu=self._gpu and compute_velocity_in_every_step,
                                          layout=layout, latex_name='u')
        if density_data_name is None:
            self._data_handling.add_array(self.density_data_name, values_per_cell=1,
                                          gpu=self._gpu and compute_density_in_every_step,
                                          layout=layout, latex_name='ρ')

        if compute_velocity_in_every_step:
            method_parameters['output']['velocity'] = self._data_handling.fields[self.velocity_data_name]
        if compute_density_in_every_step:
            density_field = self._data_handling.fields[self.density_data_name]
            if self.density_data_index is not None:
                density_field = density_field(density_data_index)
            method_parameters['output']['density'] = density_field
        if velocity_input_array_name is not None:
            method_parameters['velocity_input'] = self._data_handling.fields[velocity_input_array_name]
        if method_parameters['omega_output_field'] and isinstance(method_parameters['omega_output_field'], str):
            method_parameters['omega_output_field'] = data_handling.add_array(method_parameters['omega_output_field'])

        self.kernel_params = kernel_params

        # --- Kernel creation ---
        if lbm_kernel is None:
            switch_to_symbolic_relaxation_rates_for_omega_adapting_methods(method_parameters, self.kernel_params)
            optimization['symbolic_field'] = data_handling.fields[self._pdf_arr_name]
            method_parameters['field_name'] = self._pdf_arr_name
            method_parameters['temporary_field_name'] = self._tmp_arr_name
            if time_step_order == 'streamCollide':
                self._lbmKernels = [create_lb_function(optimization=optimization,
                                                       **method_parameters)]
            elif time_step_order == 'collideStream':
                self._lbmKernels = [create_lb_function(optimization=optimization,
                                                       kernel_type='collide_only',
                                                       **method_parameters),
                                    create_lb_function(optimization=optimization,
                                                       kernel_type='stream_pull_only',
                                                       ** method_parameters)]

        else:
            assert self._data_handling.dim == lbm_kernel.method.dim, \
                "Error: %dD Kernel for %d dimensional domain" % (lbm_kernel.method.dim, self._data_handling.dim)
            self._lbmKernels = [lbm_kernel]

        self.method = self._lbmKernels[0].method
        self.ast = self._lbmKernels[0].ast

        # -- Boundary Handling  & Synchronization ---
        self._sync = data_handling.synchronization_function([self._pdf_arr_name], method_parameters['stencil'], target)
        self._boundary_handling = LatticeBoltzmannBoundaryHandling(self.method, self._data_handling, self._pdf_arr_name,
                                                                   name=name + "_boundary_handling",
                                                                   flag_interface=flag_interface,
                                                                   target=target, openmp=optimization['openmp'])

        # -- Macroscopic Value Kernels
        self._getterKernel, self._setterKernel = self._compile_macroscopic_setter_and_getter()

        self._data_handling.fill(self.density_data_name, 1.0, value_idx=self.density_data_index,
                                 ghost_layers=True, inner_ghost_layers=True)
        self._data_handling.fill(self.velocity_data_name, 0.0, ghost_layers=True, inner_ghost_layers=True)
        self.set_pdf_fields_from_macroscopic_values()

        # -- VTK output
        self.vtkWriter = self.data_handling.create_vtk_writer(name, [self.velocity_data_name, self.density_data_name])
        self.timeStepsRun = 0

    @property
    def boundary_handling(self):
        """Boundary handling instance of the scenario. Use this to change the boundary setup"""
        return self._boundary_handling

    @property
    def data_handling(self):
        return self._data_handling

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

    def _get_slice(self, data_name, slice_obj, masked):
        if slice_obj is None:
            slice_obj = make_slice[:, :] if self.dim == 2 else make_slice[:, :, 0.5]

        result = self._data_handling.gather_array(data_name, slice_obj)
        if result is None:
            return

        if masked:
            mask = self.boundary_handling.get_mask(slice_obj[:self.dim], 'domain', True)
            if len(mask.shape) < len(result.shape):
                assert len(mask.shape) + 1 == len(result.shape)
                mask = np.repeat(mask[..., np.newaxis], result.shape[-1], axis=2)

            result = np.ma.masked_array(result, mask)
        return result.squeeze()

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
            self._sync()
            self._boundary_handling(**self.kernel_params)
            self._data_handling.run_kernel(self._lbmKernels[1], **self.kernel_params)
        else:  # stream collide
            self._sync()
            self._boundary_handling(**self.kernel_params)
            self._data_handling.run_kernel(self._lbmKernels[0], **self.kernel_params)

        self._data_handling.swap(self._pdf_arr_name, self._tmp_arr_name, self._gpu)
        self.timeStepsRun += 1

    def post_run(self):
        if self._gpu:
            self._data_handling.to_cpu(self._pdf_arr_name)
        self._data_handling.run_kernel(self._getterKernel, **self.kernel_params)

    def run(self, timeSteps):
        self.pre_run()
        for i in range(timeSteps):
            self.time_step()
        self.post_run()

    def benchmark_run(self, time_steps):
        time_loop = TimeLoop()
        time_loop.add_step(self)
        duration_of_time_step = time_loop.benchmark_run(time_steps)
        mlups = self.number_of_cells / duration_of_time_step * 1e-6
        return mlups

    def benchmark(self, time_for_benchmark=5, init_time_steps=10, number_of_time_steps_for_estimation=20):
        time_loop = TimeLoop()
        time_loop.add_step(self)
        duration_of_time_step = time_loop.benchmark(time_for_benchmark, init_time_steps,
                                                    number_of_time_steps_for_estimation)
        mlups = self.number_of_cells / duration_of_time_step * 1e-6
        return mlups

    def write_vtk(self):
        self.vtkWriter(self.timeStepsRun)

    def _compile_macroscopic_setter_and_getter(self):
        lb_method = self.method
        dim = lb_method.dim
        q = len(lb_method.stencil)
        cqc = lb_method.conserved_quantity_computation
        pdf_field = self._data_handling.fields[self._pdf_arr_name]
        rho_field = self._data_handling.fields[self.density_data_name]
        rho_field = rho_field.center if self.density_data_index is None else rho_field(self.density_data_index)
        vel_field = self._data_handling.fields[self.velocity_data_name]
        pdf_symbols = [pdf_field(i) for i in range(q)]

        getter_eqs = cqc.output_equations_from_pdfs(pdf_symbols, {'density': rho_field, 'velocity': vel_field})
        getter_kernel = create_kernel(getter_eqs, target='cpu').compile()

        inp_eqs = cqc.equilibrium_input_equations_from_init_values(rho_field, [vel_field(i) for i in range(dim)])
        setter_eqs = lb_method.get_equilibrium(conserved_quantity_equations=inp_eqs)
        setter_eqs = setter_eqs.new_with_substitutions({sym: pdf_field(i)
                                                      for i, sym in enumerate(lb_method.post_collision_pdf_symbols)})

        setter_eqs = create_simplification_strategy(lb_method)(setter_eqs)
        setter_kernel = create_kernel(setter_eqs, target='cpu').compile()
        return getter_kernel, setter_kernel
