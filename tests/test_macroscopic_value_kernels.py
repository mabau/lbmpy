import pytest
import numpy as np

from lbmpy.creationfunctions import create_lb_method, LBMConfig
from lbmpy.enums import Stencil, ForceModel, Method
from lbmpy.macroscopic_value_kernels import (
    compile_macroscopic_values_getter, compile_macroscopic_values_setter)
from lbmpy.advanced_streaming.utility import streaming_patterns, Timestep
from lbmpy.stencils import LBStencil


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19])
@pytest.mark.parametrize('force_model', [ForceModel.GUO, ForceModel.LUO, None])
@pytest.mark.parametrize('compressible', [True, False])
@pytest.mark.parametrize('zero_centered', [True, False])
@pytest.mark.parametrize('streaming_pattern', streaming_patterns)
@pytest.mark.parametrize('prev_timestep', [Timestep.EVEN, Timestep.ODD])
def test_set_get_density_velocity_with_fields(stencil, force_model, compressible, zero_centered, streaming_pattern, prev_timestep):
    force = (0.1, 0.12, -0.17)
    stencil = LBStencil(stencil)

    lbm_config = LBMConfig(stencil=stencil, force_model=force_model, method=Method.TRT,
                           compressible=compressible, zero_centered=zero_centered, force=force)

    method = create_lb_method(lbm_config=lbm_config)
    ghost_layers = 1
    inner_size = (3, 7, 4)[:method.dim]
    total_size = tuple(s + 2 * ghost_layers for s in inner_size)
    pdf_arr = np.zeros(total_size + (len(method.stencil),))

    inner_slice = (slice(ghost_layers, -ghost_layers, 1), ) * method.dim
    density_input_field = np.zeros(total_size)
    velocity_input_field = np.zeros(total_size + (method.dim, ))
    density_input_field[inner_slice] = 1 + 0.2 * (np.random.random_sample(inner_size) - 0.5)
    velocity_input_field[inner_slice] = 0.1 * \
        (np.random.random_sample(inner_size + (method.dim, )) - 0.5)

    quantities_to_set = {'density': density_input_field, 'velocity': velocity_input_field}

    setter = compile_macroscopic_values_setter(method, pdf_arr=pdf_arr, quantities_to_set=quantities_to_set,
                                               ghost_layers=ghost_layers, streaming_pattern=streaming_pattern, previous_timestep=prev_timestep)
    setter(pdf_arr)

    getter = compile_macroscopic_values_getter(method, ['density', 'density_deviation', 'velocity'], pdf_arr=pdf_arr, 
                                               ghost_layers=ghost_layers, streaming_pattern=streaming_pattern, previous_timestep=prev_timestep)

    inner_part = (slice(ghost_layers, -ghost_layers), ) * stencil.D

    density_output_field = np.zeros_like(density_input_field)
    density_deviation_output_field = np.zeros_like(density_input_field)
    velocity_output_field = np.zeros_like(velocity_input_field)
    getter(pdfs=pdf_arr, density=density_output_field, density_deviation=density_deviation_output_field, velocity=velocity_output_field)
    np.testing.assert_almost_equal(density_input_field, density_output_field)
    np.testing.assert_almost_equal(density_input_field[inner_part] - 1.0, density_deviation_output_field[inner_part])
    np.testing.assert_almost_equal(velocity_input_field, velocity_output_field)


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19])
@pytest.mark.parametrize('force_model', [ForceModel.GUO, ForceModel.LUO, None])
@pytest.mark.parametrize('compressible', [True, False])
def test_set_get_constant_velocity(stencil, force_model, compressible):
    ref_velocity = [0.01, -0.2, 0.1]

    force = (0.1, 0.12, -0.17)

    stencil = LBStencil(stencil)
    lbm_config = LBMConfig(stencil=stencil, force_model=force_model, method=Method.TRT,
                           compressible=compressible, force=force)

    method = create_lb_method(lbm_config=lbm_config)
    size = (1, 1, 1)[:method.dim]
    pdf_arr = np.zeros(size + (len(method.stencil),))
    setter = compile_macroscopic_values_setter(method, pdf_arr=pdf_arr, ghost_layers=0,
                                               quantities_to_set={'velocity': ref_velocity[:method.dim]}, )
    setter(pdf_arr)

    getter = compile_macroscopic_values_getter(method, ['velocity'], pdf_arr=pdf_arr, ghost_layers=0)
    velocity_output_field = np.zeros(size + (method.dim, ))
    getter(pdfs=pdf_arr, velocity=velocity_output_field)
    if method.dim == 2:
        computed_velocity = velocity_output_field[0, 0, :]
    else:
        computed_velocity = velocity_output_field[0, 0, 0, :]

    np.testing.assert_almost_equal(np.array(ref_velocity[:method.dim]), computed_velocity)
