from hashlib import sha256

from pystencils import Backend, CreateKernelConfig, Target
from lbmpy.creationfunctions import create_lb_ast
from lbmpy.enums import Stencil, Method
from lbmpy.creationfunctions import LBMConfig
from lbmpy.stencils import LBStencil


def test_hash_equivalence_llvm():
    import pytest
    pytest.importorskip("llvmlite")
    from pystencils.llvm.llvmjit import generate_llvm

    ref_value = "f1b1879e304fe8533977c885f2744516dd4964064a7e4ae64fd94b8426d995bb"

    lbm_config = LBMConfig(stencil=LBStencil(Stencil.D2Q9), method=Method.SRT)
    config = CreateKernelConfig(target=Target.CPU, backend=Backend.LLVM)
    ast = create_lb_ast(lbm_config=lbm_config, config=config)
    code = generate_llvm(ast)
    hash_value = sha256(str(code).encode()).hexdigest()
    assert hash_value == ref_value
