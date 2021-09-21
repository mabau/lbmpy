import numpy as np

from pystencils import Backend, Target
from lbmpy.creationfunctions import create_lb_function
import pytest

@pytest.mark.parametrize('cuda', [False, True])
def test_srt(cuda):
    if cuda:
        pytest.importorskip("pycuda")
    src = np.zeros((3, 3, 9))
    dst = np.zeros_like(src)
    opt_params = {} if not cuda else {'target': Target.GPU, 'backend': Backend.CUDA}
    func = create_lb_function(method='srt', stencil='D2Q9', relaxation_rates=[1.8], compressible=False,
                              optimization=opt_params)

    if cuda:
        import pycuda.gpuarray as gpuarray
        gpu_src, gpu_dst = gpuarray.to_gpu(src), gpuarray.to_gpu(dst)
        func(src=gpu_src, dst=gpu_dst)
        gpu_src.get(src)
        gpu_dst.get(dst)
    else:
        func(src=src, dst=dst)

    np.testing.assert_allclose(np.sum(np.abs(dst)), 0.0, atol=1e-13)
