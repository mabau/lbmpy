import numpy as np

from pystencils import Backend, Target, CreateKernelConfig
from lbmpy.creationfunctions import create_lb_function, LBMConfig
from lbmpy.enums import Method, Stencil
from lbmpy.stencils import LBStencil
import pytest


@pytest.mark.parametrize('setup', [(Target.CPU, Backend.C), (Target.GPU, Backend.CUDA)])
@pytest.mark.parametrize('method', [Method.SRT, Method.MRT, Method.CENTRAL_MOMENT, Method.CUMULANT])
@pytest.mark.parametrize('compressible', [False, True])
@pytest.mark.parametrize('delta_equilibrium', [False, True])
def test_simple_equilibrium_conservation(setup, method, compressible, delta_equilibrium):
    if setup[0] == Target.GPU:
        pytest.importorskip("pycuda")

    if method == Method.SRT and not delta_equilibrium:
        pytest.skip()

    if method == Method.CUMULANT and (not compressible or delta_equilibrium):
        pytest.skip()

    src = np.zeros((3, 3, 9))
    dst = np.zeros_like(src)
    config = CreateKernelConfig(target=setup[0], backend=setup[1])
    lbm_config = LBMConfig(stencil=LBStencil(Stencil.D2Q9), method=method,
                           relaxation_rate=1.8, compressible=compressible,
                           zero_centered=True, delta_equilibrium=delta_equilibrium)
    func = create_lb_function(lbm_config=lbm_config, config=config)

    if setup[0] == Target.GPU:
        import pycuda.gpuarray as gpuarray
        gpu_src, gpu_dst = gpuarray.to_gpu(src), gpuarray.to_gpu(dst)
        func(src=gpu_src, dst=gpu_dst)
        gpu_src.get(src)
        gpu_dst.get(dst)
    else:
        func(src=src, dst=dst)

    np.testing.assert_allclose(np.sum(np.abs(dst)), 0.0, atol=1e-13)
