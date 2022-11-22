from lbmpy.fieldaccess import PdfFieldAccessor, \
    StreamPullTwoFieldsAccessor, \
    StreamPushTwoFieldsAccessor, \
    AAEvenTimeStepAccessor, \
    AAOddTimeStepAccessor, \
    EsoTwistEvenTimeStepAccessor, \
    EsoTwistOddTimeStepAccessor, \
    EsoPullEvenTimeStepAccessor, \
    EsoPullOddTimeStepAccessor, \
    EsoPushEvenTimeStepAccessor, \
    EsoPushOddTimeStepAccessor

import numpy as np
import pystencils as ps
from enum import IntEnum


class Timestep(IntEnum):
    EVEN = 0
    ODD = 1
    BOTH = 2

    def next(self):
        return self if self == Timestep.BOTH else Timestep((self + 1) % 2)

    @property
    def idx(self):
        """To use this timestep as an array index"""
        return self % 2

    def __str__(self):
        if self == Timestep.EVEN:
            return 'Even'
        elif self == Timestep.ODD:
            return 'Odd'
        else:
            return 'Both'


streaming_patterns = ['push', 'pull', 'aa', 'esotwist', 'esopull', 'esopush']

even_accessors = {
    'pull': StreamPullTwoFieldsAccessor,
    'push': StreamPushTwoFieldsAccessor,
    'aa': AAEvenTimeStepAccessor,
    'esotwist': EsoTwistEvenTimeStepAccessor,
    'esopull': EsoPullEvenTimeStepAccessor,
    'esopush': EsoPushEvenTimeStepAccessor
}

odd_accessors = {
    'pull': StreamPullTwoFieldsAccessor,
    'push': StreamPushTwoFieldsAccessor,
    'aa': AAOddTimeStepAccessor,
    'esotwist': EsoTwistOddTimeStepAccessor,
    'esopull': EsoPullOddTimeStepAccessor,
    'esopush': EsoPushOddTimeStepAccessor
}


def get_accessor(streaming_pattern: str, timestep: Timestep) -> PdfFieldAccessor:
    if streaming_pattern not in streaming_patterns:
        raise ValueError(
            "Invalid value of parameter 'streaming_pattern'.", streaming_pattern)

    if timestep == Timestep.EVEN:
        return even_accessors[streaming_pattern]
    else:
        return odd_accessors[streaming_pattern]


def is_inplace(streaming_pattern):
    if streaming_pattern not in streaming_patterns:
        raise ValueError('Invalid streaming pattern', streaming_pattern)

    return streaming_pattern in ['aa', 'esotwist', 'esopull', 'esopush']


def get_timesteps(streaming_pattern):
    return (Timestep.EVEN, Timestep.ODD) if is_inplace(streaming_pattern) else (Timestep.BOTH, )


def numeric_offsets(field_access: ps.Field.Access):
    return tuple(int(o) for o in field_access.offsets)


def numeric_index(field_access: ps.Field.Access):
    return tuple(int(i) for i in field_access.index)


def inverse_dir_index(stencil, direction):
    return stencil.index(tuple(-d for d in stencil[direction]))


class AccessPdfValues:
    """Allows to access values from a PDF array correctly depending on 
    the streaming pattern."""

    def __init__(self, stencil,
                 streaming_pattern='pull', timestep=Timestep.BOTH, streaming_dir='out',
                 accessor=None):
        if streaming_dir not in ['in', 'out']:
            raise ValueError('Invalid streaming direction.', streaming_dir)

        pdf_field = ps.Field.create_generic('pdfs', len(stencil[0]), index_shape=(stencil.Q,))

        if accessor is None:
            accessor = get_accessor(streaming_pattern, timestep)
        self.accs = accessor.read(pdf_field, stencil) \
            if streaming_dir == 'in' \
            else accessor.write(pdf_field, stencil)

    def write_pdf(self, pdf_arr, pos, d, value):
        offsets = numeric_offsets(self.accs[d])
        pos = tuple(p + o for p, o in zip(pos, offsets))
        i = numeric_index(self.accs[d])[0]
        pdf_arr[pos + (i,)] = value

    def read_pdf(self, pdf_arr, pos, d):
        offsets = numeric_offsets(self.accs[d])
        pos = tuple(p + o for p, o in zip(pos, offsets))
        i = numeric_index(self.accs[d])[0]
        return pdf_arr[pos + (i,)]

    def read_multiple(self, pdf_arr, indices):
        """Returns PDF values for a list of index tuples (x, y, [z,] dir)"""
        return np.array([self.read_pdf(pdf_arr, idx[:-1], idx[-1]) for idx in indices])

    def collect_from_index_list(self, pdf_arr, index_list):
        """To collect PDF values according to an pystencils boundary handling index list"""
        def to_index_tuple(idx):
            return tuple(idx[v] for v in ('x', 'y', 'z')[:len(idx) - 1] + ('dir',))

        return self.read_multiple(pdf_arr, (to_index_tuple(idx) for idx in index_list))
