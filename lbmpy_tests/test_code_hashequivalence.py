from hashlib import sha256
from pystencils.backends.cbackend import generate_c
from pystencils.llvm.llvmjit import generate_llvm
from lbmpy.creationfunctions import create_lb_ast


def test_hash_equivalence():
    """
    This test should ensure that if the Python interpreter is called multiple times to generated the same method
    exactly the same code (not only functionally equivalent code) should be produced.
    Due to undefined order in sets and dicts this may no be the case.
    """
    ref_value = "5dfbb90b02e4940f05dcca11b43e1bb885d5655566735b52ad8c64f511848420"
    ast = create_lb_ast(stencil='D3Q19', method='srt', optimization={'openmp': False})
    code = generate_c(ast)
    hash_value = sha256(code.encode()).hexdigest()
    assert hash_value == ref_value


def test_hash_equivalence_llvm():
    ref_value = "52dad2fb2c144062b524ab0b514115a1a1b22d7e7f1c8e3d0f3169e08954f8ea"
    ast = create_lb_ast(stencil='D3Q19', method='srt', optimization={'target': 'llvm'})
    code = generate_llvm(ast)
    hash_value = sha256(str(code).encode()).hexdigest()
    assert hash_value == ref_value
