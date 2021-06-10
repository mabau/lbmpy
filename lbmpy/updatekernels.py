import numpy as np
import sympy as sp
import warnings

from lbmpy.fieldaccess import StreamPullTwoFieldsAccessor
from lbmpy.methods.abstractlbmethod import LbmCollisionRule
from pystencils import Assignment, AssignmentCollection, Field
from pystencils.field import create_numpy_array_with_layout, layout_string_to_tuple
from pystencils.simp import add_subexpressions_for_field_reads
from pystencils.sympyextensions import fast_subs

# -------------------------------------------- LBM Kernel Creation -----------------------------------------------------


def create_lbm_kernel(collision_rule, src_field, dst_field=None, accessor=StreamPullTwoFieldsAccessor()):
    """Replaces the pre- and post collision symbols in the collision rule by field accesses.

    Args:
        collision_rule:  instance of LbmCollisionRule, defining the collision step
        src_field: field used for reading pdf values
        dst_field: field used for writing pdf values if accessor.is_inplace this parameter is ignored
        accessor: instance of PdfFieldAccessor, defining where to read and write values
                  to create e.g. a fused stream-collide kernel See 'fieldaccess.PdfFieldAccessor'

    Returns:
        LbmCollisionRule where pre- and post collision symbols have been replaced
    """
    if accessor.is_inplace:
        dst_field = src_field

    if not accessor.is_inplace and dst_field is None:
        raise ValueError("For two field accessors a destination field has to be provided")

    method = collision_rule.method
    pre_collision_symbols = method.pre_collision_pdf_symbols
    post_collision_symbols = method.post_collision_pdf_symbols
    substitutions = {}

    input_accesses = accessor.read(src_field, method.stencil)
    output_accesses = accessor.write(dst_field, method.stencil)

    for (idx, offset), input_access, output_access in zip(enumerate(method.stencil), input_accesses, output_accesses):
        substitutions[pre_collision_symbols[idx]] = input_access
        substitutions[post_collision_symbols[idx]] = output_access

    result = collision_rule.new_with_substitutions(substitutions)

    if 'split_groups' in result.simplification_hints:
        new_split_groups = []
        for split_group in result.simplification_hints['split_groups']:
            new_split_groups.append([fast_subs(e, substitutions) for e in split_group])
        result.simplification_hints['split_groups'] = new_split_groups

    if accessor.is_inplace:
        result = add_subexpressions_for_field_reads(result, subexpressions=True, main_assignments=True)

    return result


def create_stream_only_kernel(stencil, src_field, dst_field=None, accessor=StreamPullTwoFieldsAccessor()):
    """Creates a stream kernel, without collision.

    Args:
        stencil: lattice Boltzmann stencil which is used in the form of a tuple of tuples
        src_field: field used for reading pdf values
        dst_field: field used for writing pdf values if accessor.is_inplace this parameter is ignored
        accessor: instance of PdfFieldAccessor, defining where to read and write values
                  to create e.g. a fused stream-collide kernel See 'fieldaccess.PdfFieldAccessor'

    Returns:
        AssignmentCollection of the stream only update rule
    """
    if accessor.is_inplace:
        dst_field = src_field

    if not accessor.is_inplace and dst_field is None:
        raise ValueError("For two field accessors a destination field has to be provided")

    temporary_symbols = sp.symbols(f'streamed_:{len(stencil)}')
    subexpressions = [Assignment(tmp, acc) for tmp, acc in zip(temporary_symbols, accessor.read(src_field, stencil))]
    main_assignments = [Assignment(acc, tmp) for acc, tmp in zip(accessor.write(dst_field, stencil), temporary_symbols)]
    return AssignmentCollection(main_assignments, subexpressions=subexpressions)


def create_stream_pull_only_kernel(stencil, numpy_arr=None, src_field_name="src", dst_field_name="dst",
                                   generic_layout='numpy', generic_field_type=np.float64):
    """Creates a stream kernel with the pull scheme, without collision.

    Args:
        stencil: lattice Boltzmann stencil which is used
        numpy_arr: numpy array which containes the pdf field data. If no numpy array is provided the symbolic field
                   accesses are created with 'Field.create_generic'. Otherwise 'Field.create_from_numpy_array' is used.
        src_field_name: name of the source field.
        dst_field_name: name of the destination field.
        generic_layout: data layout. for example 'fzyx' of 'zyxf'.
        generic_field_type: field data type.

    Returns:
        AssignmentCollection of the stream only update rule
    """
    warnings.warn("This function is depricated. Please use create_stream_only_kernel. If no PdfFieldAccessor is "
                  "provided to this function a standard StreamPullTwoFieldsAccessor is used ", DeprecationWarning)
    dim = len(stencil[0])
    if numpy_arr is None:
        src = Field.create_generic(src_field_name, dim, index_shape=(len(stencil),),
                                   layout=generic_layout, dtype=generic_field_type)
        dst = Field.create_generic(dst_field_name, dim, index_shape=(len(stencil),),
                                   layout=generic_layout, dtype=generic_field_type)
    else:
        src = Field.create_from_numpy_array(src_field_name, numpy_arr, index_dimensions=1)
        dst = Field.create_from_numpy_array(dst_field_name, numpy_arr, index_dimensions=1)
    return create_stream_only_kernel(stencil, src, dst, accessor=StreamPullTwoFieldsAccessor())


def create_stream_pull_with_output_kernel(lb_method, src_field, dst_field=None, output=None,
                                          accessor=StreamPullTwoFieldsAccessor()):
    """Creates a stream kernel, without collision but macroscopic quantaties like density or velocity can be calculated.

    Args:
        lb_method: lattice Boltzmann method see 'creationfunctions.create_lb_method'
        src_field: field used for reading pdf values
        dst_field: field used for writing pdf values if accessor.is_inplace this parameter is ignored
        output: dictonary which containes macroscopic quantities as keys which should be calculated and fields as
                values which should be used to write the data e.g.: {'density': density_field}
        accessor: instance of PdfFieldAccessor, defining where to read and write values
                  to create e.g. a fused stream-collide kernel See 'fieldaccess.PdfFieldAccessor'

    Returns:
        AssignmentCollection of the stream only update rule
    """
    if accessor.is_inplace:
        dst_field = src_field

    if not accessor.is_inplace and dst_field is None:
        raise ValueError("For two field accessors a destination field has to be provided")

    stencil = lb_method.stencil
    cqc = lb_method.conserved_quantity_computation
    streamed = sp.symbols(f"streamed_:{len(stencil)}")
    stream_assignments = [Assignment(a, b) for a, b in zip(streamed, accessor.read(src_field, stencil))]
    output_eq_collection = cqc.output_equations_from_pdfs(streamed, output) if output\
        else AssignmentCollection(main_assignments=[])
    write_eqs = [Assignment(a, b) for a, b in zip(accessor.write(dst_field, stencil), streamed)]

    subexpressions = stream_assignments + output_eq_collection.subexpressions
    main_eqs = output_eq_collection.main_assignments + write_eqs
    return LbmCollisionRule(lb_method, main_eqs, subexpressions,
                            simplification_hints=output_eq_collection.simplification_hints)


# ---------------------------------- Pdf array creation for various layouts --------------------------------------------


def create_pdf_array(size, num_directions, ghost_layers=1, layout='fzyx'):
    """Creates an empty numpy array for a pdf field with the specified memory layout.

    Examples:
        >>> create_pdf_array((3, 4, 5), 9, layout='zyxf', ghost_layers=0).shape
        (3, 4, 5, 9)
        >>> create_pdf_array((3, 4, 5), 9, layout='zyxf', ghost_layers=0).strides
        (72, 216, 864, 8)
        >>> create_pdf_array((3, 4), 9, layout='zyxf', ghost_layers=1).shape
        (5, 6, 9)
        >>> create_pdf_array((3, 4), 9, layout='zyxf', ghost_layers=1).strides
        (72, 360, 8)
    """
    size_with_gl = [s + 2 * ghost_layers for s in size]
    dim = len(size)
    if isinstance(layout, str):
        layout = layout_string_to_tuple(layout, dim + 1)
    return create_numpy_array_with_layout(size_with_gl + [num_directions], layout)
