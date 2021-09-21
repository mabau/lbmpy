from hashlib import sha256

from pystencils import Backend, Target
from lbmpy.creationfunctions import create_lb_ast

def test_hash_equivalence_llvm():
    import pytest
    pytest.importorskip("llvmlite")
    from pystencils.llvm.llvmjit import generate_llvm

    ref_value = "a25d1507f222fc50c3900108835976445360b09ffd7f51635c441473f4baab23"
    ast = create_lb_ast(stencil='D3Q19', method='srt', optimization={'backend': Backend.LLVM})
    code = generate_llvm(ast)
    hash_value = sha256(str(code).encode()).hexdigest()
    assert hash_value == ref_value
