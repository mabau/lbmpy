import numpy as np

from lbmpy.creationfunctions import create_lb_function
from lbmpy.enums import Stencil
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments
from lbmpy.phasefield.analytical import force_from_phi_and_mu
from lbmpy.phasefield.cahn_hilliard_lbm import cahn_hilliard_lb_method
from lbmpy.phasefield.kerneleqs import mu_kernel
from lbmpy.stencils import LBStencil
from pystencils import Assignment, create_data_handling, create_kernel, Target
from pystencils.fd import discretize_spatial
from pystencils.fd.spatial import fd_stencils_forth_order_isotropic
from pystencils.simp import sympy_cse_on_assignment_list
from pystencils.slicing import SlicedGetterDataHandling


class PhaseFieldStepDirect:

    def __init__(self, free_energy, order_parameters, domain_size, data_handling=None, name='pfn',
                 hydro_dynamic_relaxation_rate=1.0,
                 cahn_hilliard_relaxation_rates=1.0,
                 cahn_hilliard_gammas=1,
                 optimization=None):
        if optimization is None:
            optimization = {'openmp': False, 'target': Target.CPU}
        openmp = optimization.get('openmp', False)
        target = optimization.get('target', Target.CPU)

        if not hasattr(cahn_hilliard_relaxation_rates, '__len__'):
            cahn_hilliard_relaxation_rates = [cahn_hilliard_relaxation_rates] * len(order_parameters)

        if not hasattr(cahn_hilliard_gammas, '__len__'):
            cahn_hilliard_gammas = [cahn_hilliard_gammas] * len(order_parameters)

        if data_handling is None:
            data_handling = create_data_handling(domain_size, periodicity=True, parallel=False)
        dh = data_handling
        self.data_handling = dh

        stencil = LBStencil(Stencil.D3Q19) if dh.dim == 3 else LBStencil(Stencil.D2Q9)

        self.free_energy = free_energy

        # Data Handling
        kernel_parameters = {'cpu_openmp': openmp, 'target': target, 'ghost_layers': 2}
        gpu = target == Target.GPU
        phi_size = len(order_parameters)
        gl = kernel_parameters['ghost_layers']
        self.phi_field = dh.add_array(f'{name}_phi', values_per_cell=phi_size, gpu=gpu, latex_name='φ',
                                      ghost_layers=gl)
        self.mu_field = dh.add_array(f"{name}_mu", values_per_cell=phi_size, gpu=gpu, latex_name="μ",
                                     ghost_layers=gl)
        self.vel_field = dh.add_array(f"{name}_u", values_per_cell=data_handling.dim, gpu=gpu, latex_name="u",
                                      ghost_layers=gl)

        self.force_field = dh.add_array(f"{name}_force", values_per_cell=dh.dim, gpu=gpu, latex_name="F",
                                        ghost_layers=gl)

        self.phi = SlicedGetterDataHandling(self.data_handling, self.phi_field.name)
        self.mu = SlicedGetterDataHandling(self.data_handling, self.mu_field.name)
        self.velocity = SlicedGetterDataHandling(self.data_handling, self.vel_field.name)
        self.force = SlicedGetterDataHandling(self.data_handling, self.force_field.name)

        self.ch_pdfs = []
        for i in range(len(order_parameters)):
            src = dh.add_array(f"{name}_ch_src{i}", values_per_cell=stencil.Q, ghost_layers=gl)
            dst = dh.add_array(f"{name}_ch_dst{i}", values_per_cell=stencil.Q, ghost_layers=gl)
            self.ch_pdfs.append((src, dst))
        self.hydro_pdfs = (dh.add_array(f"{name}_hydro_src", values_per_cell=stencil.Q, ghost_layers=gl),
                           dh.add_array(f"{name}_hydro_dst", values_per_cell=stencil.Q, ghost_layers=gl))

        # Compute Kernels
        mu_assignments = mu_kernel(self.free_energy, order_parameters, self.phi_field, self.mu_field)
        mu_assignments = [Assignment(a.lhs, a.rhs.doit()) for a in mu_assignments]
        mu_assignments = sympy_cse_on_assignment_list(mu_assignments)
        self.mu_kernel = create_kernel(mu_assignments, **kernel_parameters).compile()

        force_rhs = force_from_phi_and_mu(self.phi_field.center_vector, dim=dh.dim, mu=self.mu_field.center_vector)
        force_rhs = [discretize_spatial(e, dx=1, stencil=fd_stencils_forth_order_isotropic) for e in force_rhs]
        force_assignments = [Assignment(lhs, rhs)
                             for lhs, rhs in zip(self.force_field.center_vector, force_rhs)]
        self.force_kernel = create_kernel(force_assignments, **kernel_parameters).compile()

        self.ch_lb_kernels = []
        for i, (src, dst) in enumerate(self.ch_pdfs):
            ch_method = cahn_hilliard_lb_method(stencil, self.mu_field(i),

                                                relaxation_rate=cahn_hilliard_relaxation_rates[i],
                                                gamma=cahn_hilliard_gammas[i])
            opt = optimization.copy()
            opt['symbolic_field'] = src
            opt['symbolic_temporary_field'] = dst
            kernel = create_lb_function(lb_method=ch_method, optimization=opt,
                                        velocity_input=self.vel_field.center_vector,
                                        output={'density': self.phi_field(i)})
            self.ch_lb_kernels.append(kernel)

        opt = optimization.copy()
        opt['symbolic_field'] = self.hydro_pdfs[0]
        opt['symbolic_temporary_field'] = self.hydro_pdfs[1]
        self.hydro_lb_kernel = create_lb_function(stencil=stencil, relaxation_rate=hydro_dynamic_relaxation_rate,
                                                  force=self.force_field.center_vector,
                                                  output={'velocity': self.vel_field}, optimization=opt)

        # Setter Kernels
        self.init_kernels = []
        for i in range(len(order_parameters)):
            ch_method = self.ch_lb_kernels[i].method
            init_assign = pdf_initialization_assignments(lb_method=ch_method,
                                                         density=self.phi_field.center_vector[i],
                                                         velocity=self.vel_field.center_vector,
                                                         pdfs=self.ch_pdfs[i][0].center_vector)
            init_kernel = create_kernel(init_assign, **kernel_parameters).compile()
            self.init_kernels.append(init_kernel)

        init_assign = pdf_initialization_assignments(lb_method=self.hydro_lb_kernel.method, density=1,
                                                     velocity=self.vel_field.center_vector,
                                                     pdfs=self.hydro_pdfs[0].center_vector)
        self.init_kernels.append(create_kernel(init_assign, **kernel_parameters).compile())

        # Sync functions
        self.phi_sync = dh.synchronization_function([self.phi_field.name])
        self.mu_sync = dh.synchronization_function([self.mu_field.name])
        self.pdf_sync = dh.synchronization_function([self.hydro_pdfs[0].name]
                                                    + [src.name for src, _ in self.ch_pdfs])

        self.reset()

    def reset(self):
        dh = self.data_handling
        dh.fill(self.vel_field.name, 0)
        dh.fill(self.mu_field.name, 0)
        dh.fill(self.force_field.name, 0)

    def set_pdf_fields_from_macroscopic_values(self):
        self.reset()
        for k in self.init_kernels:
            self.data_handling.run_kernel(k)

    def set_concentration(self, slice_obj, concentration):
        phi = np.array(concentration)

        for b in self.data_handling.iterate(slice_obj):
            for i in range(phi.shape[-1]):
                b[self.phi_field.name][..., i] = phi[i]

    def set_single_concentration(self, slice_obj, phase_idx, value=1):
        num_phases = self.phi_field.index_shape[0]
        zeros = [0] * num_phases
        zeros[phase_idx] = value
        self.set_concentration(slice_obj, zeros)

    def smooth(self, sigma=2):
        from scipy.ndimage.filters import gaussian_filter
        dh = self.data_handling

        for block in dh.iterate(ghost_layers=True):
            c_arr = block[self.phi_field.name]
            for i in range(self.phi_field.index_shape[0]):
                gaussian_filter(c_arr[..., i], sigma=sigma, output=c_arr[..., i])

    def time_step(self):
        dh = self.data_handling

        self.phi_sync()
        dh.run_kernel(self.mu_kernel)

        self.mu_sync()
        dh.run_kernel(self.force_kernel)

        self.pdf_sync()
        dh.run_kernel(self.hydro_lb_kernel)
        dh.swap(self.hydro_pdfs[0].name, self.hydro_pdfs[1].name)

        for ch_kernel, (src, dst) in zip(self.ch_lb_kernels, self.ch_pdfs):
            dh.run_kernel(ch_kernel)
            dh.swap(src.name, dst.name)
