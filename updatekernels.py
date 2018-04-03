import numpy as np
import sympy as sp

from pystencils import Field, Assignment
from pystencils.assignment_collection.assignment_collection import AssignmentCollection
from pystencils.field import create_numpy_array_with_layout, layout_string_to_tuple
from pystencils.sympyextensions import fast_subs
from lbmpy.methods.abstractlbmethod import LbmCollisionRule
from lbmpy.fieldaccess import StreamPullTwoFieldsAccessor, PeriodicTwoFieldsAccessor, CollideOnlyInplaceAccessor


# -------------------------------------------- LBM Kernel Creation -----------------------------------------------------


def create_lbm_kernel(collision_rule, input_field, output_field, accessor):
    """Replaces the pre- and post collision symbols in the collision rule by field accesses.

    Args:
        collision_rule:  instance of LbmCollisionRule, defining the collision step
        input_field: field used for reading pdf values
        output_field: field used for writing pdf values (may be the same as input field for certain accessors)
        accessor: instance of PdfFieldAccessor, defining where to read and write values
                  to create e.g. a fused stream-collide kernel

    Returns:
        LbmCollisionRule where pre- and post collision symbols have been replaced
    """
    method = collision_rule.method
    pre_collision_symbols = method.pre_collision_pdf_symbols
    post_collision_symbols = method.post_collision_pdf_symbols
    substitutions = {}

    input_accesses = accessor.read(input_field, method.stencil)
    output_accesses = accessor.write(output_field, method.stencil)

    for (idx, offset), inputAccess, outputAccess in zip(enumerate(method.stencil), input_accesses, output_accesses):
        substitutions[pre_collision_symbols[idx]] = inputAccess
        substitutions[post_collision_symbols[idx]] = outputAccess

    result = collision_rule.new_with_substitutions(substitutions)

    if 'split_groups' in result.simplification_hints:
        new_split_groups = []
        for splitGroup in result.simplification_hints['split_groups']:
            new_split_groups.append([fast_subs(e, substitutions) for e in splitGroup])
        result.simplification_hints['split_groups'] = new_split_groups

    return result


def create_stream_pull_collide_kernel(collision_rule, numpy_arr=None, src_field_name="src", dst_field_name="dst",
                                      generic_layout='numpy', generic_field_type=np.float64,
                                      builtin_periodicity=(False, False, False)):
    """Implements a stream-pull scheme, where values are read from source and written to destination field.

    Args:
        collision_rule: a collision rule created by lbm method
        numpy_arr: optional numpy field for PDFs. Used to create a kernel of fixed loop bounds and strides
                    if None, a generic kernel is created
        src_field_name: name of the pdf source field
        dst_field_name: name of the pdf destination field
        generic_layout: if no numpy_arr is given to determine the layout, a variable sized field with the given
                       generic_layout is used
        generic_field_type: if no numpy_arr is given, this data type is used for the fields
        builtin_periodicity: periodicity that should be built into the kernel

    Returns:
        lbm update rule, where generic pdf references are replaced by field accesses
    """
    dim = collision_rule.method.dim
    if numpy_arr is not None:
        assert len(numpy_arr.shape) == dim + 1, "Field dimension mismatch: dimension is %s, should be %d" % \
                                                (len(numpy_arr.shape), dim + 1)

    if numpy_arr is None:
        src = Field.create_generic(src_field_name, dim, index_dimensions=1, layout=generic_layout, dtype=generic_field_type)
        dst = Field.create_generic(dst_field_name, dim, index_dimensions=1, layout=generic_layout, dtype=generic_field_type)
    else:
        src = Field.create_from_numpy_array(src_field_name, numpy_arr, index_dimensions=1)
        dst = Field.create_from_numpy_array(dst_field_name, numpy_arr, index_dimensions=1)

    accessor = StreamPullTwoFieldsAccessor

    if any(builtin_periodicity):
        accessor = PeriodicTwoFieldsAccessor(builtin_periodicity, ghost_layers=1)
    return create_lbm_kernel(collision_rule, src, dst, accessor)


def create_collide_only_kernel(collision_rule, numpy_arr=None, field_name="src",
                               generic_layout='numpy', generic_field_type=np.float64):
    """Implements a collision only (no neighbor access) LBM kernel.

    For parameters see function ``create_stream_pull_collide_kernel``
    """
    dim = collision_rule.method.dim
    if numpy_arr is not None:
        assert len(numpy_arr.shape) == dim + 1, "Field dimension mismatch: dimension is %s, should be %d" % \
                                                (len(numpy_arr.shape), dim + 1)

    if numpy_arr is None:
        field = Field.create_generic(field_name, dim, index_dimensions=1, layout=generic_layout, dtype=generic_field_type)
    else:
        field = Field.create_from_numpy_array(field_name, numpy_arr, index_dimensions=1)

    return create_lbm_kernel(collision_rule, field, field, CollideOnlyInplaceAccessor)


def create_stream_pull_only_kernel(stencil, numpy_arr=None, src_field_name="src", dst_field_name="dst",
                                   generic_layout='numpy', generic_field_type=np.float64):
    """Creates a stream-pull kernel, without collision.

    For parameters see function ``create_stream_pull_collide_kernel``
    """

    dim = len(stencil[0])
    if numpy_arr is None:
        src = Field.create_generic(src_field_name, dim, index_dimensions=1,
                                   layout=generic_layout, dtype=generic_field_type)
        dst = Field.create_generic(dst_field_name, dim, index_dimensions=1,
                                   layout=generic_layout, dtype=generic_field_type)
    else:
        src = Field.create_from_numpy_array(src_field_name, numpy_arr, index_dimensions=1)
        dst = Field.create_from_numpy_array(dst_field_name, numpy_arr, index_dimensions=1)

    accessor = StreamPullTwoFieldsAccessor()
    eqs = [Assignment(a, b) for a, b in zip(accessor.write(dst, stencil), accessor.read(src, stencil))]
    return AssignmentCollection(eqs, [])


def create_stream_pull_with_output_kernel(lb_method, src_field, dst_field, output):
    stencil = lb_method.stencil
    cqc = lb_method.conserved_quantity_computation
    streamed = sp.symbols("streamed_:%d" % (len(stencil),))
    accessor = StreamPullTwoFieldsAccessor()
    stream_assignments = [Assignment(a, b) for a, b in zip(streamed, accessor.read(src_field, stencil))]
    output_eq_collection = cqc.output_equations_from_pdfs(streamed, output)
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


# ------------------------------------------- Add output fields to kernel ----------------------------------------------


def add_output_field_for_conserved_quantities(collision_rule, conserved_quantities_to_output_field_dict):
    method = collision_rule.method
    cqc = method.conserved_quantity_computation.output_equations_from_pdfs(method.pre_collision_pdf_symbols,
                                                                           conserved_quantities_to_output_field_dict)
    return collision_rule.new_merged(cqc)


def write_quantities_to_field(collision_rule, symbols, output_field):
    if not hasattr(symbols, "__len__"):
        symbols = [symbols]
    eqs = [Assignment(output_field(i), s) for i, s in enumerate(symbols)]
    return collision_rule.copy(collision_rule.main_assignments + eqs)
