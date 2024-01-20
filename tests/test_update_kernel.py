import pytest

import pystencils as ps

from lbmpy.advanced_streaming.utility import get_timesteps, streaming_patterns, get_accessor, \
    is_inplace, AccessPdfValues
from lbmpy.enums import Stencil
from lbmpy.stencils import LBStencil
from lbmpy.updatekernels import create_stream_only_kernel
from pystencils import create_kernel, Target


@pytest.mark.parametrize('streaming_pattern', streaming_patterns)
def test_stream_only_kernel(streaming_pattern):
    domain_size = (4, 4)
    stencil = LBStencil(Stencil.D2Q9)
    dh = ps.create_data_handling(domain_size, default_target=Target.CPU)
    pdfs = dh.add_array('pdfs', values_per_cell=len(stencil))
    pdfs_tmp = dh.add_array_like('pdfs_tmp', 'pdfs')

    for t in get_timesteps(streaming_pattern):
        accessor = get_accessor(streaming_pattern, t)
        src = pdfs
        dst = pdfs if is_inplace(streaming_pattern) else pdfs_tmp

        dh.fill(src.name, 0.0)
        dh.fill(dst.name, 0.0)

        stream_kernel = create_stream_only_kernel(stencil, src, dst, accessor=accessor)
        stream_func = create_kernel(stream_kernel).compile()

        #   Check functionality
        acc_in = AccessPdfValues(stencil, streaming_dir='in', accessor=accessor)
        for i in range(len(stencil)):
            acc_in.write_pdf(dh.cpu_arrays[src.name], (1,1), i, i)

        dh.run_kernel(stream_func)

        acc_out = AccessPdfValues(stencil, streaming_dir='out', accessor=accessor)
        for i in range(len(stencil)):
            assert acc_out.read_pdf(dh.cpu_arrays[dst.name], (1,1), i) == i
        
