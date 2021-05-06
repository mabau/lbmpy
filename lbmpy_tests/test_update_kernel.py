import pytest

import pystencils as ps

from lbmpy.stencils import get_stencil
from lbmpy.fieldaccess import StreamPullTwoFieldsAccessor, StreamPushTwoFieldsAccessor,\
    AAOddTimeStepAccessor, AAEvenTimeStepAccessor, EsoTwistOddTimeStepAccessor, EsoTwistEvenTimeStepAccessor
from lbmpy.updatekernels import create_stream_only_kernel


@pytest.mark.parametrize('accessor', [StreamPullTwoFieldsAccessor(), StreamPushTwoFieldsAccessor(),
                                      AAOddTimeStepAccessor(), AAEvenTimeStepAccessor(),
                                      EsoTwistOddTimeStepAccessor(), EsoTwistEvenTimeStepAccessor()])
def test_stream_only_kernel(accessor):
    domain_size = (4, 4)
    stencil = get_stencil("D2Q9")
    dh = ps.create_data_handling(domain_size, default_target='cpu')

    src = dh.add_array('src', values_per_cell=len(stencil))
    dh.fill('src', 0.0, ghost_layers=True)

    dst = dh.add_array_like('dst', 'src')
    dh.fill('dst', 0.0, ghost_layers=True)

    pull = create_stream_only_kernel(stencil, None, src.name, dst.name, accessor=accessor)

    for i, eq in enumerate(pull.main_assignments):
        assert eq.rhs.offsets == accessor.read(src, stencil)[i].offsets
        assert eq.lhs.offsets == accessor.write(dst, stencil)[i].offsets