import sympy as sp
from pystencils.sympyextensions import is_constant

#   Subexpression Insertion


def insert_subexpressions(ac, selection_callback, skip=set()):
    i = 0
    while i < len(ac.subexpressions):
        exp = ac.subexpressions[i]
        if exp.lhs not in skip and selection_callback(exp):
            ac = ac.new_with_inserted_subexpression(exp.lhs)
        else:
            i += 1

    return ac


def insert_aliases(ac, **kwargs):
    return insert_subexpressions(ac, lambda x: isinstance(x.rhs, sp.Symbol), **kwargs)


def insert_zeros(ac, **kwargs):
    zero = sp.Integer(0)
    return insert_subexpressions(ac, lambda x: x.rhs == zero, **kwargs)


def insert_constants(ac, **kwargs):
    return insert_subexpressions(ac, lambda x: is_constant(x.rhs), **kwargs)


def insert_symbol_times_minus_one(ac, **kwargs):
    def callback(exp):
        rhs = exp.rhs
        minus_one = sp.Integer(-1)
        return isinstance(rhs, sp.Mul) and \
            len(rhs.args) == 2 and \
            (rhs.args[0] == minus_one or rhs.args[1] == minus_one)
    return insert_subexpressions(ac, callback, **kwargs)


def insert_constant_multiples(ac, **kwargs):
    def callback(exp):
        rhs = exp.rhs
        return isinstance(rhs, sp.Mul) and \
            len(rhs.args) == 2 and \
            (is_constant(rhs.args[0]) or is_constant(rhs.args[1]))
    return insert_subexpressions(ac, callback, **kwargs)


def insert_constant_additions(ac, **kwargs):
    def callback(exp):
        rhs = exp.rhs
        return isinstance(rhs, sp.Add) and \
            len(rhs.args) == 2 and \
            (is_constant(rhs.args[0]) or is_constant(rhs.args[1]))
    return insert_subexpressions(ac, callback, **kwargs)


def insert_squares(ac, **kwargs):
    two = sp.Integer(2)

    def callback(exp):
        rhs = exp.rhs
        return isinstance(rhs, sp.Pow) and rhs.args[1] == two
    return insert_subexpressions(ac, callback, **kwargs)


def bind_symbols_to_skip(insertion_function, skip):
    return lambda ac: insertion_function(ac, skip=skip)
