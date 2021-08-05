import sympy as sp

from pystencils.simp import (
    SimplificationStrategy, apply_to_all_assignments,
    insert_aliases, insert_zeros, insert_constants)


def create_phasefield_simplification_strategy(lb_method):
    s = SimplificationStrategy()
    expand = apply_to_all_assignments(sp.expand)

    s.add(expand)

    s.add(insert_zeros)
    s.add(insert_aliases)
    s.add(insert_constants)
    s.add(lambda ac: ac.new_without_unused_subexpressions())

    return s
