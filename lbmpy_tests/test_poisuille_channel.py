"""
Test Poiseuille flow against analytical solution
"""

import pytest

from lbmpy.enums import Stencil

import pystencils as ps
from poiseuille import poiseuille_channel


@pytest.mark.parametrize('target', (ps.Target.CPU, ps.Target.GPU, ps.Target.OPENCL))
@pytest.mark.parametrize('stencil_name', (Stencil.D2Q9, Stencil.D3Q19))
def test_poiseuille_channel(target, stencil_name):
    # OpenCL and Cuda
    if target == ps.Target.OPENCL:
        import pytest
        pytest.importorskip("pyopencl")
        import pystencils.opencl.autoinit
    elif target == ps.Target.GPU:
        import pytest
        pytest.importorskip("pycuda")

    poiseuille_channel(target=target, stencil_name=stencil_name)
