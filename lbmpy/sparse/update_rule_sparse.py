from pystencils import Assignment, AssignmentCollection
# noinspection PyProtectedMember
from pystencils.field import Field, FieldType, compute_strides

AC = AssignmentCollection


def create_symbolic_list(name, num_cells, values_per_cell, dtype, layout='AoS'):
    assert layout in ('AoS', 'SoA')
    layout = (0, 1) if layout == 'AoS' else (1, 0)
    if values_per_cell > 1:
        shape = (num_cells, values_per_cell)
        spatial_layout = (0,)
    else:
        shape = (num_cells,)
        spatial_layout = (0,)
        layout = (0,)
    strides = compute_strides(shape, layout)
    return Field(name, FieldType.CUSTOM, dtype, spatial_layout, shape, strides)


def create_lb_update_rule_sparse(collision_rule, src, dst, idx, kernel_type='stream_pull_collide') -> AC:
    """Creates a update rule from a collision rule using compressed pdf storage and two (src/dst) arrays.

    Args:
        collision_rule: arbitrary collision rule, e.g. created with create_lb_collision_rule
        src: symbolic field to read from
        dst: symbolic field to write to
        idx: symbolic index field
        kernel_type: one of 'stream_pull_collide', 'collide_only' or 'stream_pull_only'
    Returns:
        update rule
    """
    assert kernel_type in ('stream_pull_collide', 'collide_only', 'stream_pull_only')
    method = collision_rule.method
    q = len(method.stencil)

    symbol_subs = _list_substitutions(method, src, idx)

    if kernel_type == 'stream_pull_only':
        assignments = []
        for i in range(q):
            lhs = dst(i)
            rhs = symbol_subs[method.pre_collision_pdf_symbols[i]]
            if lhs - rhs != 0:
                assignments.append(Assignment(lhs, rhs))
        return AssignmentCollection(assignments, subexpressions=[])
    else:
        write_target = src if kernel_type == 'collide_only' else dst
        symbol_subs.update({sym: write_target(i) for i, sym in enumerate(method.post_collision_pdf_symbols)})
        return collision_rule.new_with_substitutions(symbol_subs)


def create_macroscopic_value_getter_sparse(method, pdfs, output_descriptor) -> AC:
    """Returns assignment collection with assignments to compute density and/or velocity.

    Args:
        method: lb method
        pdfs: symbolic pdf field
        output_descriptor: see `output_equations_from_pdfs`
    """
    cqc = method.conserved_quantity_computation
    getter_eqs = cqc.output_equations_from_pdfs(pdfs.center_vector, output_descriptor)
    return getter_eqs


def create_macroscopic_value_setter_sparse(method, pdfs, density, velocity) -> AC:
    """Returns assignment collection to set a pdf array to equilibrium with given density and velocity.

    Args:
        method: see `create_macroscopic_value_getter_sparse`
        pdfs: symbolic pdf field
        density: True to read density from array, or for spatially constant density a single symbol/value
        velocity: similar to density parameter
    """
    cqc = method.conserved_quantity_computation
    inp_eqs = cqc.equilibrium_input_equations_from_init_values(density, velocity, force_substitution=False)
    result = method.get_equilibrium(conserved_quantity_equations=inp_eqs)
    substitutions = {a: b for a, b in zip(method.post_collision_pdf_symbols, pdfs.center_vector)}
    return result.new_with_substitutions(substitutions).new_without_subexpressions()


# ---------------------------------------- Helper Functions ------------------------------------------------------------


def _list_substitutions(method, src, idx, store_center=False):
    if store_center:
        result = {sym: src.absolute_access((idx(i),), ())
                  for i, sym in enumerate(method.pre_collision_pdf_symbols)}
    else:
        result = {sym: src.absolute_access((idx(i - 1),), ())
                  for i, sym in enumerate(method.pre_collision_pdf_symbols)}
        result[method.pre_collision_pdf_symbols[0]] = src(0)

    return result
