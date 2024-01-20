import pytest
import numpy as np
from numpy.testing import assert_allclose

import pystencils as ps
import sympy as sp

from lbmpy.enums import Stencil, Method
from lbmpy.stencils import LBStencil
from lbmpy.creationfunctions import LBMConfig, LBMOptimisation, create_lb_function
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter, macroscopic_values_getter

@pytest.mark.parametrize('zero_centered', [False, True])
def test_density_velocity_input(zero_centered):
    stencil = LBStencil(Stencil.D2Q9)
    dh = ps.create_data_handling((5,5), default_target=ps.Target.CPU)
    rho_in = dh.add_array("rho_in", 1)
    rho_out = dh.add_array_like("rho_out", "rho_in")
    u_in = dh.add_array("u_in", 2)
    u_out = dh.add_array_like("u_out", "u_in")
    pdfs = dh.add_array("pdfs", stencil.Q)

    lb_config = LBMConfig(stencil=Stencil.D2Q9, method=Method.SRT, zero_centered=zero_centered,
                          relaxation_rate=sp.Integer(1),
                          density_input=rho_in.center, velocity_input=u_in.center_vector,
                          kernel_type="collide_only")

    lb_opt = LBMOptimisation(symbolic_field=pdfs)

    lb_func = create_lb_function(lbm_config=lb_config, lbm_optimisation=lb_opt)

    setter = macroscopic_values_setter(lb_func.method, 1, (0, 0), pdfs.center_vector)
    setter_kernel = ps.create_kernel(setter).compile()

    getter = macroscopic_values_getter(lb_func.method, rho_out.center, u_out.center_vector, pdfs.center_vector)
    getter_kernel = ps.create_kernel(getter).compile()

    dh.run_kernel(setter_kernel)

    dh.cpu_arrays[rho_in.name][1:-1, 1:-1] = 1.0 + 0.1 * np.random.random_sample((5, 5))
    dh.cpu_arrays[u_in.name][1:-1, 1:-1] = 0.05 + 0.01 * np.random.random_sample((5, 5, 2))

    dh.run_kernel(lb_func)
    dh.run_kernel(getter_kernel)

    assert_allclose(dh.cpu_arrays[rho_out.name][1:-1, 1:-1], dh.cpu_arrays[rho_in.name][1:-1, 1:-1])
    assert_allclose(dh.cpu_arrays[u_out.name][1:-1, 1:-1], dh.cpu_arrays[u_in.name][1:-1, 1:-1])

