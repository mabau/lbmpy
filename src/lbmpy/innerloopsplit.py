from collections import defaultdict

import sympy as sp

from lbmpy.methods.abstractlbmethod import LbmCollisionRule
from pystencils import Field


def create_lbm_split_groups(cr: LbmCollisionRule, opposing_directions=True):
    """
    Creates split groups for LBM collision equations. For details about split groups see
    :func:`pystencils.transformation.split_inner_loop` .
    The split groups are added as simplification hint 'split_groups'

    Split groups are created in the following way: Opposing directions are put
    into a single group if opposing_directions, else all stores are put into separate loops
    The velocity subexpressions are pre-computed as well as all subexpressions which are used in all
    non-center collision equations, and depend on at least one pdf.

    Required simplification hints:
        - velocity: sequence of velocity symbols
    """
    sh = cr.simplification_hints
    assert 'velocity' in sh, "Needs simplification hint 'velocity': Sequence of velocity symbols"

    pre_collision_symbols = set(cr.method.pre_collision_pdf_symbols)
    non_center_post_collision_symbols = set(cr.method.post_collision_pdf_symbols[1:])
    post_collision_symbols = set(cr.method.post_collision_pdf_symbols)

    stencil = cr.method.stencil

    important_sub_expressions = {e.lhs for e in cr.subexpressions
                                 if pre_collision_symbols.intersection(cr.dependent_symbols([e.lhs]))}

    other_written_fields = []
    for eq in cr.main_assignments:
        if eq.lhs not in post_collision_symbols and isinstance(eq.lhs, Field.Access):
            other_written_fields.append(eq.lhs)
        if eq.lhs not in non_center_post_collision_symbols:
            continue
        important_sub_expressions.intersection_update(eq.rhs.atoms(sp.Symbol))

    important_sub_expressions.update(sh['velocity'])

    subexpressions_to_pre_compute = list(important_sub_expressions)
    subexpressions_to_pre_compute.sort(key=lambda e: e.name)  # ensures that exactly the same code is produced (caching)
    split_groups = [subexpressions_to_pre_compute + other_written_fields, ]

    direction_groups = defaultdict(list)
    dim = len(stencil[0])

    if opposing_directions:
        for direction, eq in zip(stencil, cr.main_assignments):
            if direction == tuple([0] * dim):
                split_groups[0].append(eq.lhs)
                continue

            inverse_dir = tuple([-i for i in direction])

            if inverse_dir in direction_groups:
                direction_groups[inverse_dir].append(eq.lhs)
            else:
                direction_groups[direction].append(eq.lhs)
        split_groups += direction_groups.values()
    else:
        for e in cr.main_assignments:
            split_groups.append([e.lhs])

    cr.simplification_hints['split_groups'] = split_groups
    return cr
