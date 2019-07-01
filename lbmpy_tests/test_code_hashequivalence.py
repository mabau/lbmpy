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
    ref_value = "bd20ebb3cb8ca2aa00128a0920c5c0fa5b1b15251111e5695c1759fe29849743"
    ast = create_lb_ast(stencil='D3Q19', method='srt', optimization={'openmp': False})
    code = generate_c(ast)
    hash_value = sha256(code.encode()).hexdigest()
    assert hash_value == ref_value


def test_hash_equivalence_llvm():
    ref_value = "6db6ed9e2cbd05edae8fcaeb8168e3178dd578c2681133f3ae9228b23d2be432"
    ast = create_lb_ast(stencil='D3Q19', method='srt', optimization={'target': 'llvm'})
    code = generate_llvm(ast)
    hash_value = sha256(str(code).encode()).hexdigest()
    assert hash_value == ref_value
