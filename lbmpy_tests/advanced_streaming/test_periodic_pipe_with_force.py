import numpy as np
from dataclasses import replace

from pystencils.datahandling import create_data_handling
from pystencils import create_kernel, Target, CreateKernelConfig

from lbmpy.creationfunctions import create_lb_collision_rule, create_lb_function, LBMConfig, LBMOptimisation
from lbmpy.enums import ForceModel, Method, Stencil
from lbmpy.macroscopic_value_kernels import macroscopic_values_getter, macroscopic_values_setter
from lbmpy.stencils import LBStencil

from lbmpy.advanced_streaming import LBMPeriodicityHandling
from lbmpy.boundaries import NoSlip, LatticeBoltzmannBoundaryHandling
from lbmpy.advanced_streaming.utility import is_inplace, streaming_patterns, get_timesteps

import pytest
from numpy.testing import assert_allclose

all_results = dict()

targets = [Target.CPU]

try:
    import pycuda.autoinit
    targets += [Target.GPU]
except Exception:
    pass


class PeriodicPipeFlow:
    def __init__(self, stencil, streaming_pattern, wall_boundary=None, target=Target.CPU):

        if wall_boundary is None:
            wall_boundary = NoSlip()

        self.target = target
        self.gpu = target in [Target.GPU]

        #   Stencil
        self.stencil = stencil
        self.q = stencil.Q
        self.dim = stencil.D

        #   Streaming
        self.streaming_pattern = streaming_pattern
        self.inplace = is_inplace(self.streaming_pattern)
        self.timesteps = get_timesteps(streaming_pattern)
        self.zeroth_timestep = self.timesteps[0]

        #   Domain, Data Handling and PDF fields
        self.pipe_length = 60
        self.pipe_radius = 15
        self.domain_size = (self.pipe_length, ) + (2 * self.pipe_radius,) * (self.dim - 1)
        self.periodicity = (True, ) + (False, ) * (self.dim - 1)
        self.force = (0.0001, ) + (0.0,) * (self.dim - 1)

        self.dh = create_data_handling(domain_size=self.domain_size,
                                       periodicity=self.periodicity, default_target=self.target)

        self.pdfs = self.dh.add_array('pdfs', self.q)
        if not self.inplace:
            self.pdfs_tmp = self.dh.add_array_like('pdfs_tmp', self.pdfs.name)

        #   LBM Streaming and Collision
        lbm_config = LBMConfig(stencil=stencil, method=Method.SRT, relaxation_rate=1.0,
                               force_model=ForceModel.GUO, force=self.force, streaming_pattern=streaming_pattern)

        lbm_opt = LBMOptimisation(symbolic_field=self.pdfs)
        config = CreateKernelConfig(target=self.target)

        if not self.inplace:
            lbm_opt = replace(lbm_opt, symbolic_temporary_field=self.pdfs_tmp)

        self.lb_collision = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)
        self.lb_method = self.lb_collision.method

        self.lb_kernels = []
        for t in self.timesteps:
            lbm_config = replace(lbm_config, timestep=t)
            self.lb_kernels.append(create_lb_function(collision_rule=self.lb_collision,
                                                      lbm_config=lbm_config,
                                                      lbm_optimisation=lbm_opt,
                                                      config=config))

        #   Macroscopic Values
        self.density = 1.0
        self.density_field = self.dh.add_array('rho', 1)
        u_x = 0.0
        self.velocity = (u_x,) * self.dim
        self.velocity_field = self.dh.add_array('u', self.dim)

        setter = macroscopic_values_setter(
            self.lb_method, self.density, self.velocity, self.pdfs,
            streaming_pattern=self.streaming_pattern, previous_timestep=self.zeroth_timestep)
        self.init_kernel = create_kernel(setter,
                                         config=CreateKernelConfig(target=target, ghost_layers=1)).compile()

        self.getter_kernels = []
        for t in self.timesteps:
            getter = macroscopic_values_getter(
                self.lb_method, self.density_field, self.velocity_field, self.pdfs,
                streaming_pattern=self.streaming_pattern, previous_timestep=t)
            self.getter_kernels.append(create_kernel(getter,
                                                     config=CreateKernelConfig(target=target, ghost_layers=1)).compile())

        #   Periodicity
        self.periodicity_handler = LBMPeriodicityHandling(
            self.stencil, self.dh, self.pdfs.name, streaming_pattern=self.streaming_pattern)

        #   Boundary Handling
        self.wall = wall_boundary
        self.bh = LatticeBoltzmannBoundaryHandling(
            self.lb_method, self.dh, self.pdfs.name,
            streaming_pattern=self.streaming_pattern, target=self.target)

        self.bh.set_boundary(boundary_obj=self.wall, mask_callback=self.mask_callback)

        self.current_timestep = self.zeroth_timestep

    def mask_callback(self, x, y, z=None):
        y = y - self.pipe_radius
        z = z - self.pipe_radius if z is not None else 0
        return np.sqrt(y**2 + z**2) >= self.pipe_radius

    def init(self):
        self.current_timestep = self.zeroth_timestep
        self.dh.run_kernel(self.init_kernel)

    def step(self):
        #   Order matters! First communicate, then boundaries, otherwise
        #   periodicity handling overwrites reflected populations
        # Periodicty
        self.periodicity_handler(self.current_timestep)

        # Boundaries
        self.bh(prev_timestep=self.current_timestep)

        # Here, the next time step begins
        self.current_timestep = self.current_timestep.next()

        # LBM Step
        self.dh.run_kernel(self.lb_kernels[self.current_timestep.idx])

        # Field Swaps
        if not self.inplace:
            self.dh.swap(self.pdfs.name, self.pdfs_tmp.name)

        # Macroscopic Values
        self.dh.run_kernel(self.getter_kernels[self.current_timestep.idx])

    def run(self, iterations):
        for _ in range(iterations):
            self.step()

    @property
    def velocity_array(self):
        if self.gpu:
            self.dh.to_cpu(self.velocity_field.name)
        return self.dh.gather_array(self.velocity_field.name)

    def get_trimmed_velocity_array(self):
        if self.gpu:
            self.dh.to_cpu(self.velocity_field.name)
        u = np.copy(self.dh.gather_array(self.velocity_field.name))
        mask = self.bh.get_mask(None, self.wall)
        for idx in np.ndindex(u.shape[:-1]):
            if mask[idx] != 0:
                u[idx] = np.full((self.dim, ), np.nan)
        return u


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('streaming_pattern', streaming_patterns)
@pytest.mark.parametrize('target', targets)
@pytest.mark.longrun
def test_periodic_pipe(stencil, streaming_pattern, target):
    stencil = LBStencil(stencil)
    pipeflow = PeriodicPipeFlow(stencil, streaming_pattern, target=target)
    pipeflow.init()
    pipeflow.run(100)
    u = pipeflow.get_trimmed_velocity_array()

    #   Flow must be equal up to numerical error for all streaming patterns
    global all_results
    for key, prev_u in all_results.items():
        if key[0] == stencil:
            prev_pattern = key[1]
            assert_allclose(
                u, prev_u,
                rtol=1, atol=1e-16,
                err_msg=f'Velocity field for {streaming_pattern} differed from {prev_pattern}!')
    all_results[(stencil, streaming_pattern)] = u
