import sympy as sp
from pystencils.simp.subexpression_insertion import insert_constants

from lbmpy import create_lb_collision_rule, LBMConfig, LBStencil, Stencil, Method


def test_smagorinsky_with_constant_omega():
    stencil = LBStencil(Stencil.D2Q9)

    config = LBMConfig(stencil=stencil, method=Method.SRT, smagorinsky=True, relaxation_rate=sp.Symbol("omega"))
    collision_rule = create_lb_collision_rule(lbm_config=config)

    config = LBMConfig(stencil=stencil, method=Method.SRT, smagorinsky=True, relaxation_rate=1.5)
    collision_rule2 = create_lb_collision_rule(lbm_config=config)

    collision_rule = collision_rule.subs({sp.Symbol("omega"): 1.5})
    collision_rule = insert_constants(collision_rule)

    assert collision_rule == collision_rule2