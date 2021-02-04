from hashlib import sha256

from lbmpy.creationfunctions import create_lb_ast

def test_hash_equivalence_llvm():

    import pytest
    pytest.importorskip("llvmlite")
    from pystencils.llvm.llvmjit import generate_llvm


    ref_value = "6db6ed9e2cbd05edae8fcaeb8168e3178dd578c2681133f3ae9228b23d2be432"
    ast = create_lb_ast(stencil='D3Q19', method='srt', optimization={'target': 'llvm'})
    code = generate_llvm(ast)
    hash_value = sha256(str(code).encode()).hexdigest()
    assert hash_value == ref_value
