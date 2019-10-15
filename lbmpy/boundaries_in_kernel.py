import sympy as sp

from lbmpy.boundaries.boundaryhandling import BoundaryOffsetInfo, LbmWeightInfo
from pystencils.assignment import Assignment
from pystencils.astnodes import LoopOverCoordinate
from pystencils.data_types import cast_func
from pystencils.field import Field
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.simp.simplifications import sympy_cse_on_assignment_list
from pystencils.stencil import inverse_direction
from pystencils.sympyextensions import fast_subs
from pystencils.astnodes import Block, Conditional


def direction_indices_in_direction(direction, stencil):
    for i, offset in enumerate(stencil):
        for d_i, o_i in zip(direction, offset):
            if (d_i == 1 and o_i == 1) or (d_i == -1 and o_i == -1):
                yield i
                break


def boundary_substitutions(lb_method):
    stencil = lb_method.stencil
    w = lb_method.weights
    replacements = {}
    for idx, offset in enumerate(stencil):
        symbolic_offset = BoundaryOffsetInfo.offset_from_dir(idx, dim=lb_method.dim)
        for sym, value in zip(symbolic_offset, offset):
            replacements[sym] = value

        replacements[BoundaryOffsetInfo.inv_dir(idx)] = stencil.index(inverse_direction(offset))
        replacements[LbmWeightInfo.weight_of_direction(idx)] = w[idx]
    return replacements


def type_all_numbers(expr, dtype):
    substitutions = {a: cast_func(a, dtype) for a in expr.atoms(sp.Number)}
    return expr.subs(substitutions)


def border_conditions(direction, field, ghost_layers=1):
    abs_direction = tuple(-e if e < 0 else e for e in direction)
    assert sum(abs_direction) == 1
    idx = abs_direction.index(1)
    val = direction[idx]

    loop_ctrs = [LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(len(direction))]
    loop_ctr = loop_ctrs[idx]

    gl = ghost_layers
    border_condition = sp.Eq(loop_ctr, gl if val < 0 else field.shape[idx] - gl - 1)

    if ghost_layers == 0:
        return type_all_numbers(border_condition, loop_ctr.dtype)
    else:
        other_min = [sp.Ge(c, gl)
                     for c in loop_ctrs if c != loop_ctr]
        other_max = [sp.Lt(c, field.shape[i] - gl)
                     for i, c in enumerate(loop_ctrs) if c != loop_ctr]
        result = sp.And(border_condition, *other_min, *other_max)
        return type_all_numbers(result, loop_ctr.dtype)


def transformed_boundary_rule(boundary, accessor_func, field, direction_symbol, lb_method, **kwargs):
    tmp_field = field.new_field_with_different_name("t")
    rule = boundary(tmp_field, direction_symbol, lb_method, **kwargs)
    bsubs = boundary_substitutions(lb_method)
    rule = [a.subs(bsubs) for a in rule]
    accessor_writes = accessor_func(tmp_field, lb_method.stencil)
    to_replace = set()
    for assignment in rule:
        to_replace.update({fa for fa in assignment.rhs.atoms(Field.Access) if fa.field == tmp_field})

    def compute_replacement(fa):
        f = fa.index[0]
        shift = accessor_writes[f].offsets
        new_index = tuple(a + b for a, b in zip(fa.offsets, shift))
        return field[new_index](accessor_writes[f].index[0])

    substitutions = {fa: compute_replacement(fa) for fa in to_replace}
    all_assignments = [assignment.subs(substitutions) for assignment in rule]
    main_assignments = [a for a in all_assignments if isinstance(a.lhs, Field.Access)]
    sub_expressions = [a for a in all_assignments if not isinstance(a.lhs, Field.Access)]
    assert len(main_assignments) == 1
    ac = AssignmentCollection(main_assignments, sub_expressions).new_without_subexpressions()
    return ac.main_assignments[0].rhs


def read_assignments_with_boundaries(method, pdf_field, boundary_spec, pre_stream_access, read_access):
    stencil = method.stencil
    reads = [Assignment(*v) for v in zip(method.pre_collision_pdf_symbols,
                                         read_access(pdf_field, method.stencil))]

    for direction, boundary in boundary_spec.items():
        dir_indices = direction_indices_in_direction(direction, method.stencil)
        border_cond = border_conditions(direction, pdf_field, ghost_layers=1)
        for dir_index in dir_indices:
            inv_index = stencil.index(inverse_direction(stencil[dir_index]))
            value_from_boundary = transformed_boundary_rule(boundary, pre_stream_access, pdf_field, dir_index,
                                                            method, index_field=None)
            value_without_boundary = reads[inv_index].rhs
            new_rhs = sp.Piecewise((value_from_boundary, border_cond),
                                            (value_without_boundary, True))
            reads[inv_index] = Assignment(reads[inv_index].lhs, new_rhs)

    return AssignmentCollection(reads)


def update_rule_with_boundaries(collision_rule, input_field, output_field,
                                boundaries, accessor, pre_stream_access):
    reads = read_assignments_with_boundaries(collision_rule.method, input_field, boundaries,
                                             pre_stream_access, accessor.read)

    write_substitutions = {}
    method = collision_rule.method
    post_collision_symbols = method.post_collision_pdf_symbols
    pre_collision_symbols = method.pre_collision_pdf_symbols

    output_accesses = accessor.write(output_field, method.stencil)
    input_accesses = accessor.read(input_field, method.stencil)

    for (idx, offset), output_access in zip(enumerate(method.stencil), output_accesses):
        write_substitutions[post_collision_symbols[idx]] = output_access

    result = collision_rule.new_with_substitutions(write_substitutions)
    result.subexpressions = reads.all_assignments + result.subexpressions

    if 'split_groups' in result.simplification_hints:
        all_substitutions = write_substitutions.copy()
        for (idx, offset), input_access in zip(enumerate(method.stencil), input_accesses):
            all_substitutions[pre_collision_symbols[idx]] = input_access

        new_split_groups = []
        for split_group in result.simplification_hints['split_groups']:
            new_split_groups.append([fast_subs(e, all_substitutions) for e in split_group])
        result.simplification_hints['split_groups'] = new_split_groups

    return result


def boundary_conditional(boundary, direction, read_of_next_accessor, lb_method, output_field, cse=False, **kwargs):
    stencil = lb_method.stencil
    tmp_field = output_field.new_field_with_different_name("t")

    dir_indices = direction_indices_in_direction(direction, stencil)

    assignments = []
    for direction_idx in dir_indices:
        rule = boundary(tmp_field, direction_idx, lb_method, **kwargs)
        boundary_subs = boundary_substitutions(lb_method)
        rule = [a.subs(boundary_subs) for a in rule]

        rhs_substitutions = {tmp_field(i): sym for i, sym in enumerate(lb_method.post_collision_pdf_symbols)}
        offset = stencil[direction_idx]
        inv_offset = inverse_direction(offset)
        inv_idx = stencil.index(inv_offset)

        lhs_substitutions = {
            tmp_field[offset](inv_idx): read_of_next_accessor(output_field, stencil)[inv_idx]}
        rule = [Assignment(a.lhs.subs(lhs_substitutions), a.rhs.subs(rhs_substitutions)) for a in rule]
        ac = AssignmentCollection([rule[-1]], rule[:-1]).new_without_subexpressions()
        assignments += ac.main_assignments

    border_cond = border_conditions(direction, output_field, ghost_layers=1)
    if cse:
        assignments = sympy_cse_on_assignment_list(assignments)
    return Conditional(border_cond, Block(assignments))


def update_rule_with_push_boundaries(collision_rule, field, boundary_spec, accessor, read_of_next_accessor):
    if 'split_groups' in collision_rule.simplification_hints:
        raise NotImplementedError("Split is not supported yet")
    method = collision_rule.method
    loads = [Assignment(a, b) for a, b in zip(method.pre_collision_pdf_symbols, accessor.read(field, method.stencil))]
    stores = [Assignment(a, b) for a, b in
              zip(accessor.write(field, method.stencil), method.post_collision_pdf_symbols)]

    result = loads + collision_rule.all_assignments + stores
    for direction, boundary in boundary_spec.items():
        cond = boundary_conditional(boundary, direction, read_of_next_accessor, method, field)
        result.append(cond)
    return result