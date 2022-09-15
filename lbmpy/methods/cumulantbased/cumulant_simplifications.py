import sympy as sp
from pystencils.simp.subexpression_insertion import insert_subexpressions

from warnings import warn


def insert_logs(ac, **kwargs):
    def callback(exp):
        rhs = exp.rhs
        logs = rhs.atoms(sp.log)
        return len(logs) > 0

    return insert_subexpressions(ac, callback, **kwargs)


def insert_log_products(ac, **kwargs):
    def callback(asm):
        rhs = asm.rhs
        if isinstance(rhs, sp.log):
            return True
        if isinstance(rhs, sp.Mul):
            if any(isinstance(arg, sp.log) for arg in rhs.args):
                return True
        return False

    return insert_subexpressions(ac, callback, **kwargs)


def expand_post_collision_central_moments(ac):
    if 'post_collision_monomial_central_moments' in ac.simplification_hints:
        subexpr_dict = ac.subexpressions_dict
        cm_symbols = ac.simplification_hints['post_collision_monomial_central_moments']
        for s in cm_symbols:
            if s in subexpr_dict:
                subexpr_dict[s] = subexpr_dict[s].expand()
        ac = ac.copy()
        ac.set_sub_expressions_from_dict(subexpr_dict)
    return ac


def check_for_logarithms(ac):
    logs = ac.atoms(sp.log)
    if len(logs) > 0:
        warn("""There are logarithms remaining in your cumulant-based collision operator!
                This will let your kernel's performance and numerical accuracy deterioate severly.
                Either you have disabled simplification, or it unexpectedly failed.
                If the presence of logarithms is intended, please inspect the kernel to make sure
                if this warning can be ignored.
                Otherwise, if setting `simplification='auto'` in your optimization config does not resolve
                the problem, try a different parametrization, or contact the developers.""")
