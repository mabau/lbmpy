import sympy as sp

from lbmpy.boundaries.boundaryhandling import BoundaryOffsetInfo, LbmWeightInfo
from pystencils.assignment import Assignment
from pystencils.astnodes import LoopOverCoordinate
from pystencils.data_types import cast_func
from pystencils.field import Field
from pystencils.simp.assignment_collection import AssignmentCollection
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


def transformed_boundary_rule(boundary, accessor, field, direction_symbol, lb_method, **kwargs):
    tmp_field = field.new_field_with_different_name("_tmp")
    rule = boundary(tmp_field, direction_symbol, lb_method, **kwargs)
    bsubs = boundary_substitutions(lb_method)
    rule = [a.subs(bsubs) for a in rule]
    accessor_writes = accessor.write(tmp_field, lb_method.stencil)
    to_replace = set()
    for assignment in rule:
        to_replace.update({fa for fa in assignment.atoms(Field.Access) if fa.field == tmp_field})

    def compute_replacement(fa):
        f = fa.index[0]
        shift = accessor_writes[f].offsets
        new_index = tuple(a + b for a, b in zip(fa.offsets, shift))
        return field[new_index](accessor_writes[f].index[0])

    substitutions = {fa: compute_replacement(fa) for fa in to_replace}
    all_assignments = [assignment.subs(substitutions) for assignment in rule]
    main_assignments = [a for a in all_assignments if isinstance(a.lhs, Field.Access)]
    sub_expressions = [a for a in all_assignments if not isinstance(a.lhs, Field.Access)]
    return AssignmentCollection(main_assignments, sub_expressions)


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


def read_assignments_with_boundaries(collision_rule, pdf_field,
                                     boundary_spec,
                                     prev_timestep_accessor,
                                     current_timestep_accessor):
    method = collision_rule.method
    result = {a: [b, a] for a, b in zip(current_timestep_accessor.read(pdf_field, method.stencil),
                                        method.pre_collision_pdf_symbols)}

    for direction, boundary in boundary_spec.items():
        dir_indices = direction_indices_in_direction(direction, method.stencil)
        border_cond = border_conditions(direction, pdf_field, ghost_layers=1)
        for dir_index in dir_indices:
            ac = transformed_boundary_rule(boundary, prev_timestep_accessor, pdf_field, dir_index,
                                           method, index_field=None)
            assignments = ac.new_without_subexpressions().main_assignments
            assert len(assignments) == 1
            assignment = assignments[0]
            assert assignment.lhs in result

            value_without_boundary = result[assignment.lhs][1]
            result[assignment.lhs][1] = sp.Piecewise((assignment.rhs, border_cond),
                                                     (value_without_boundary, True))

    return AssignmentCollection([Assignment(*e) for e in result.values()])


def update_rule_with_boundaries(collision_rule, input_field, output_field,
                                boundaries,
                                accessor, prev_accessor=None):
    if prev_accessor is None:
        prev_accessor = accessor

    reads = read_assignments_with_boundaries(collision_rule, input_field, boundaries,
                                             prev_timestep_accessor=prev_accessor,
                                             current_timestep_accessor=accessor)

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
