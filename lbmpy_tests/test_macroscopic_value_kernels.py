import numpy as np

from lbmpy.creationfunctions import create_lb_method
from lbmpy.macroscopic_value_kernels import (
    compile_macroscopic_values_getter, compile_macroscopic_values_setter)


def test_set_get_density_velocity_with_fields():
    for stencil in ['D2Q9', 'D3Q19']:
        for force_model in ['guo', 'luo', 'none']:
            for compressible in [True, False]:
                force = (0.1, 0.12, -0.17)
                method = create_lb_method(stencil=stencil, force_model=force_model, method='trt',
                                          compressible=compressible, force=force)
                size = (3, 7, 4)[:method.dim]
                pdf_arr = np.zeros(size + (len(method.stencil),))
                density_input_field = 1 + 0.2 * (np.random.random_sample(size) - 0.5)
                velocity_input_field = 0.1 * (np.random.random_sample(size + (method.dim, )) - 0.5)
                setter = compile_macroscopic_values_setter(method, pdf_arr=pdf_arr,
                                                           quantities_to_set={'density': density_input_field,
                                                                              'velocity': velocity_input_field}, )
                setter(pdf_arr)

                getter = compile_macroscopic_values_getter(method, ['density', 'velocity'], pdf_arr=pdf_arr)
                density_output_field = np.empty_like(density_input_field)
                velocity_output_field = np.empty_like(velocity_input_field)
                getter(pdfs=pdf_arr, density=density_output_field, velocity=velocity_output_field)
                np.testing.assert_almost_equal(density_input_field, density_output_field)
                np.testing.assert_almost_equal(velocity_input_field, velocity_output_field)


def test_set_get_constant_velocity():
    for stencil in ['D2Q9', 'D3Q19']:
        for force_model in ['guo', 'luo', 'none']:
            for compressible in [True, False]:
                ref_velocity = [0.01, -0.2, 0.1]

                force = (0.1, 0.12, -0.17)
                method = create_lb_method(stencil=stencil, force_model=force_model, method='trt',
                                          compressible=compressible, force=force)
                size = (1, 1, 1)[:method.dim]
                pdf_arr = np.zeros(size + (len(method.stencil),))
                setter = compile_macroscopic_values_setter(method, pdf_arr=pdf_arr,
                                                           quantities_to_set={'velocity': ref_velocity[:method.dim]}, )
                setter(pdf_arr)

                getter = compile_macroscopic_values_getter(method, ['velocity'], pdf_arr=pdf_arr)
                velocity_output_field = np.zeros(size + (method.dim, ))
                getter(pdfs=pdf_arr, velocity=velocity_output_field)
                if method.dim == 2:
                    computed_velocity = velocity_output_field[0, 0, :]
                else:
                    computed_velocity = velocity_output_field[0, 0, 0, :]

                np.testing.assert_almost_equal(np.array(ref_velocity[:method.dim]), computed_velocity)
