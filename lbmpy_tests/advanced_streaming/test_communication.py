import numpy as np
from lbmpy.stencils import get_stencil
from pystencils.slicing import get_slice_before_ghost_layer, get_ghost_region_slice
from lbmpy.advanced_streaming.communication import get_communication_slices, _fix_length_one_slices
from lbmpy.advanced_streaming.utility import streaming_patterns, Timestep

import pytest

@pytest.mark.parametrize('stencil', ['D2Q9', 'D3Q19', 'D3Q27'])
@pytest.mark.parametrize('streaming_pattern', streaming_patterns)
@pytest.mark.parametrize('timestep', [Timestep.EVEN, Timestep.ODD])
def test_slices_not_empty(stencil, streaming_pattern, timestep):
    stencil = get_stencil(stencil)
    dim = len(stencil[0])
    q = len(stencil)
    arr = np.zeros( (4,) * dim + (q,) )
    slices = get_communication_slices(stencil, streaming_pattern=streaming_pattern, prev_timestep=timestep, ghost_layers=1)
    for _, slices_list in slices.items():
        for src, dst in slices_list:
            assert all(s != 0 for s in arr[src].shape)
            assert all(s != 0 for s in arr[dst].shape)


@pytest.mark.parametrize('stencil', ['D2Q9', 'D3Q19', 'D3Q27'])
@pytest.mark.parametrize('streaming_pattern', streaming_patterns)
@pytest.mark.parametrize('timestep', [Timestep.EVEN, Timestep.ODD])
def test_src_dst_same_shape(stencil, streaming_pattern, timestep):
    stencil = get_stencil(stencil)
    dim = len(stencil[0])
    q = len(stencil)
    arr = np.zeros( (4,) * dim + (q,) )
    slices = get_communication_slices(stencil, streaming_pattern=streaming_pattern, prev_timestep=timestep, ghost_layers=1)
    for _, slices_list in slices.items():
        for src, dst in slices_list:
            src_shape = arr[src].shape
            dst_shape = arr[dst].shape
            assert src_shape == dst_shape


@pytest.mark.parametrize('stencil', ['D2Q9', 'D3Q19', 'D3Q27'])
def test_pull_communication_slices(stencil):
    stencil = get_stencil(stencil)

    slices = get_communication_slices(
        stencil, streaming_pattern='pull', prev_timestep=Timestep.BOTH, ghost_layers=1)
    for i, d in enumerate(stencil):
        if i == 0:
            continue

        for s in slices[d]:
            if s[0][-1] == i:
                src = s[0][:-1]
                dst = s[1][:-1]
                break
        
        inner_slice = _fix_length_one_slices(get_slice_before_ghost_layer(d, ghost_layers=1))
        inv_dir = (-e for e in d)
        gl_slice = _fix_length_one_slices(get_ghost_region_slice(inv_dir, ghost_layers=1))
        assert src == inner_slice
        assert dst == gl_slice