"""
Test Poiseuille flow against analytical solution
"""

import pytest

from lbmpy.enums import Stencil, CollisionSpace

import pystencils as ps
from poiseuille import poiseuille_channel


@pytest.mark.parametrize('target', (ps.Target.CPU, ps.Target.GPU))
@pytest.mark.parametrize('stencil_name', (Stencil.D2Q9, Stencil.D3Q19))
@pytest.mark.parametrize('zero_centered', (False, True))
@pytest.mark.parametrize('moment_space_collision', (False, True))
def test_poiseuille_channel(target, stencil_name, zero_centered, moment_space_collision):
    # Cuda
    if target == ps.Target.GPU:
        import pytest
        pytest.importorskip("pycuda")

    cspace_info = CollisionSpace.RAW_MOMENTS if moment_space_collision else CollisionSpace.POPULATIONS
    poiseuille_channel(target=target, stencil_name=stencil_name, zero_centered=zero_centered, collision_space_info=cspace_info)
