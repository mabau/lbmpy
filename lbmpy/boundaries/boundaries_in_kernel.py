import sympy as sp

from lbmpy.boundaries.boundaryhandling import BoundaryOffsetInfo, LbmWeightInfo
from pystencils.assignment import Assignment
from pystencils.astnodes import Block, Conditional, LoopOverCoordinate, SympyAssignment
from pystencils.data_types import type_all_numbers
from pystencils.field import Field
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.simp.simplifications import sympy_cse_on_assignment_list
from pystencils.stencil import inverse_direction
from pystencils.sympyextensions import fast_subs


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


def boundary_conditional(boundary, direction, read_of_next_accessor, lb_method, output_field, cse=False):
    stencil = lb_method.stencil
    tmp_field = output_field.new_field_with_different_name("t")

    dir_indices = direction_indices_in_direction(direction, stencil)

    assignments = []
    for direction_idx in dir_indices:
        rule = boundary(tmp_field, direction_idx, lb_method, index_field=None)
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
    assignments = [SympyAssignment(a.lhs, a.rhs) for a in assignments]
    return Conditional(border_cond, Block(assignments))


def update_rule_with_push_boundaries(collision_rule, field, boundary_spec, accessor, read_of_next_accessor):
    method = collision_rule.method
    loads = [Assignment(a, b) for a, b in zip(method.pre_collision_pdf_symbols, accessor.read(field, method.stencil))]
    stores = [Assignment(a, b) for a, b in
              zip(accessor.write(field, method.stencil), method.post_collision_pdf_symbols)]

    result = collision_rule.copy()
    result.subexpressions = loads + result.subexpressions
    result.main_assignments += stores
    for direction, boundary in boundary_spec.items():
        cond = boundary_conditional(boundary, direction, read_of_next_accessor, method, field)
        result.main_assignments.append(cond)

    if 'split_groups' in result.simplification_hints:
        substitutions = {b: a for a, b in zip(accessor.write(field, method.stencil), method.post_collision_pdf_symbols)}
        new_split_groups = []
        for split_group in result.simplification_hints['split_groups']:
            new_split_groups.append([fast_subs(e, substitutions) for e in split_group])
        result.simplification_hints['split_groups'] = new_split_groups

    return result
