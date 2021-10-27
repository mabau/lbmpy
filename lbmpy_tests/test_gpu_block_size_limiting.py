from lbmpy.creationfunctions import create_lb_ast, LBMConfig
from lbmpy.enums import Method, Stencil
from lbmpy.stencils import LBStencil
import pytest
from pystencils import Target, CreateKernelConfig


def test_gpu_block_size_limiting():
    pytest.importorskip("pycuda")
    too_large = 2048*2048
    lbm_config = LBMConfig(method=Method.CUMULANT, stencil=LBStencil(Stencil.D3Q19),
                           relaxation_rate=1.8, compressible=True)
    config = CreateKernelConfig(target=Target.GPU,
                                gpu_indexing_params={'block_size': (too_large, too_large, too_large)})
    ast = create_lb_ast(lbm_config=lbm_config, config=config)
    limited_block_size = ast.indexing.call_parameters((1024, 1024, 1024))
    kernel = ast.compile()
    assert all(b < too_large for b in limited_block_size['block'])
    bs = [too_large, too_large, too_large]
    ast.indexing.limit_block_size_by_register_restriction(bs, kernel.num_regs)
    assert all(b < too_large for b in bs)
