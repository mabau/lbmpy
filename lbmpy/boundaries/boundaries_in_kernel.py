import sympy as sp

from lbmpy.boundaries.boundaryhandling import LbmWeightInfo
from lbmpy.advanced_streaming.indexing import BetweenTimestepsIndexing
from lbmpy.advanced_streaming.utility import Timestep, get_accessor
from pystencils.boundaries.boundaryhandling import BoundaryOffsetInfo
from pystencils.assignment import Assignment
from pystencils.astnodes import Block, Conditional, LoopOverCoordinate, SympyAssignment
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
        return border_condition
    else:
        other_min = [sp.Ge(c, gl)
                     for c in loop_ctrs if c != loop_ctr]
        other_max = [sp.Lt(c, field.shape[i] - gl)
                     for i, c in enumerate(loop_ctrs) if c != loop_ctr]
        result = sp.And(border_condition, *other_min, *other_max)
        return result


def boundary_conditional(boundary, direction, streaming_pattern, prev_timestep, lb_method, output_field, cse=False):
    stencil = lb_method.stencil

    dir_indices = direction_indices_in_direction(direction, stencil)
    indexing = BetweenTimestepsIndexing(output_field, lb_method.stencil, prev_timestep, streaming_pattern)
    f_out, f_in = indexing.proxy_fields
    inv_dir = indexing.inverse_dir_symbol

    assignments = []
    for direction_idx in dir_indices:
        rule = boundary(f_out, f_in, direction_idx, inv_dir, lb_method, index_field=None)

        #   rhs: replace f_out by post collision symbols.
        rhs_substitutions = {f_out(i): sym for i, sym in enumerate(lb_method.post_collision_pdf_symbols)}
        rule = AssignmentCollection([rule]).new_with_substitutions(rhs_substitutions)
        rule = indexing.substitute_proxies(rule)

        ac = rule.new_without_subexpressions()
        assignments += ac.main_assignments

    border_cond = border_conditions(direction, output_field, ghost_layers=1)
    if cse:
        assignments = sympy_cse_on_assignment_list(assignments)
    assignments = [SympyAssignment(a.lhs, a.rhs) for a in assignments]
    return Conditional(border_cond, Block(assignments))


def update_rule_with_push_boundaries(collision_rule, field, boundary_spec, 
                                     streaming_pattern='pull', timestep=Timestep.BOTH):
    method = collision_rule.method
    accessor = get_accessor(streaming_pattern, timestep)
    loads = [Assignment(a, b) for a, b in zip(method.pre_collision_pdf_symbols, accessor.read(field, method.stencil))]
    stores = [Assignment(a, b) for a, b in
              zip(accessor.write(field, method.stencil), method.post_collision_pdf_symbols)]

    result = collision_rule.copy()
    result.subexpressions = loads + result.subexpressions
    result.main_assignments += stores
    for direction, boundary in boundary_spec.items():
        cond = boundary_conditional(boundary, direction, streaming_pattern, timestep, method, field)
        result.main_assignments.append(cond)

    if 'split_groups' in result.simplification_hints:
        substitutions = {b: a for a, b in zip(accessor.write(field, method.stencil), method.post_collision_pdf_symbols)}
        new_split_groups = []
        for split_group in result.simplification_hints['split_groups']:
            new_split_groups.append([fast_subs(e, substitutions) for e in split_group])
        result.simplification_hints['split_groups'] = new_split_groups

    return result
