import numpy as np
import sympy as sp

from lbmpy.advanced_streaming import Timestep
from lbmpy.boundaries import NoSlip
from lbmpy.boundaries.boundaryhandling import create_lattice_boltzmann_boundary_kernel
from lbmpy.advanced_streaming.utility import even_accessors, odd_accessors, streaming_patterns, inverse_dir_index, AccessPdfValues
from lbmpy.creationfunctions import create_lb_method
from lbmpy.stencils import get_stencil

import pystencils as ps

from pystencils.boundaries.createindexlist import numpy_data_type_for_boundary_object
from pystencils.data_types import TypedSymbol, create_type
from pystencils.field import Field, FieldType

from itertools import product

import pytest

@pytest.mark.parametrize("stencil", [ 'D2Q9', 'D3Q19', 'D3Q27'])
@pytest.mark.parametrize("streaming_pattern", streaming_patterns)
@pytest.mark.parametrize("prev_timestep", [Timestep.EVEN, Timestep.ODD])
def test_advanced_streaming_noslip_single_cell(stencil, streaming_pattern, prev_timestep):
    """
    Advanced Streaming NoSlip Test
    """

    stencil = get_stencil(stencil)
    q = len(stencil)
    dim = len(stencil[0])
    pdf_field = ps.fields(f'pdfs({q}): [{dim}D]')

    prev_pdf_access = AccessPdfValues(pdf_field, stencil, streaming_pattern, prev_timestep, 'out')
    next_pdf_access = AccessPdfValues(pdf_field, stencil, streaming_pattern, prev_timestep.next(), 'in')

    pdfs = np.zeros((3,) * dim + (q,))
    pos = (1,) * dim
    for d in range(q):
        prev_pdf_access.write_pdf(pdfs, pos, d, d)

    lb_method = create_lb_method(stencil=stencil, method='srt')
    noslip = NoSlip()

    index_struct_dtype = numpy_data_type_for_boundary_object(noslip, dim)

    index_field = Field('indexVector', FieldType.INDEXED, index_struct_dtype, layout=[0],
                    shape=(TypedSymbol("indexVectorSize", create_type(np.int64)), 1), strides=(1, 1))
    index_vector = np.array([ pos + (d,) for d in range(q) ], dtype=index_struct_dtype)

    ast = create_lattice_boltzmann_boundary_kernel(pdf_field, 
                    index_field, lb_method, noslip,
                    prev_timestep=prev_timestep,
                    streaming_pattern=streaming_pattern)

    flex_kernel = ast.compile()
    
    flex_kernel(pdfs=pdfs, indexVector=index_vector, indexVectorSize=len(index_vector))

    reflected_pdfs = [ next_pdf_access.read_pdf(pdfs, pos, d) for d in range(q)]
    inverse_pdfs = [ inverse_dir_index(stencil, d) for d in range(q) ] 
    assert reflected_pdfs == inverse_pdfs