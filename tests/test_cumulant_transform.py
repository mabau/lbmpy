import pytest
from itertools import chain
import sympy as sp

from pystencils import AssignmentCollection

from lbmpy.moment_transforms import (
    CentralMomentsToCumulantsByGeneratingFunc, PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT,
    PRE_COLLISION_CUMULANT, PRE_COLLISION_MONOMIAL_CUMULANT
)
from lbmpy.methods import cascaded_moment_sets_literature
from lbmpy.stencils import Stencil, LBStencil

@pytest.mark.parametrize("stencil", [Stencil.D2Q9, Stencil.D3Q19])
def test_identity(stencil):
    stencil = LBStencil(stencil)
    polys = list(chain.from_iterable(cascaded_moment_sets_literature(stencil)))
    rho = sp.Symbol('rho')
    u = sp.symbols('u_:2')
    transform = CentralMomentsToCumulantsByGeneratingFunc(stencil, polys, rho, u,
                    post_collision_symbol_base=PRE_COLLISION_CUMULANT)

    forward_eqs = transform.forward_transform()
    backward_eqs = transform.backward_transform(central_moment_base=PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT)

    subexpressions = forward_eqs.all_assignments + backward_eqs.subexpressions
    main_assignments = backward_eqs.main_assignments
    ac = AssignmentCollection(main_assignments, subexpressions=subexpressions)
    ac = ac.new_without_subexpressions()

    for lhs, rhs in ac.main_assignments_dict.items():
        assert (lhs - rhs).expand() == 0
