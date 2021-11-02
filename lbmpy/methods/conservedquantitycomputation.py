import abc
from collections import OrderedDict

import sympy as sp

from pystencils import Assignment, AssignmentCollection, Field


class AbstractConservedQuantityComputation(abc.ABC):
    r"""

    This class defines how conserved quantities are computed as functions of the pdfs.
    Conserved quantities are used for output and as input to the equilibrium in the collision step

    Depending on the method they might also be computed slightly different, e.g. due to a force model.

    An additional method describes how to get the conserved quantities for the equilibrium for initialization.
    In most cases the inputs can be used directly, but for some methods they have to be altered slightly.
    For example in zero centered hydrodynamic schemes with force model, the density has
    to be decreased by one, and the given velocity has to be shifted dependent on the force.

    .. image:: /img/moment_shift.svg

    """

    @property
    @abc.abstractmethod
    def conserved_quantities(self):
        """
        Dict, mapping names (symbol) to dimensionality (int)
        For example: {'density' : 1, 'velocity' : 3}
        The naming strings can be used in :func:`output_equations_from_pdfs`
        and :func:`equilibrium_input_equations_from_init_values`
        """

    def defined_symbols(self, order='all'):
        """
        Returns a dict, mapping names of conserved quantities to their symbols
        """

    @property
    @abc.abstractmethod
    def default_values(self):
        """
        Returns a dict of symbol to default value, where "default" means that
        the equilibrium simplifies to the weights if these values are inserted.
        Hydrodynamic example: rho=1, u_i = 0
        """

    @abc.abstractmethod
    def equilibrium_input_equations_from_pdfs(self, pdfs):
        """
        Returns an equation collection that defines all necessary quantities to compute the equilibrium as functions
        of the pdfs.
        For hydrodynamic LBM schemes this is usually the density and velocity.

        Args:
            pdfs: values or symbols for the pdf values
        """

    @abc.abstractmethod
    def output_equations_from_pdfs(self, pdfs, output_quantity_names_to_symbols):
        """
        Returns an equation collection that defines conserved quantities for output. These conserved quantities might
        be slightly different that the ones used as input for the equilibrium e.g. due to a force model.

        Args:
            pdfs: values for the pdf entries
            output_quantity_names_to_symbols: dict mapping of conserved quantity names
             (See :func:`conserved_quantities`) to symbols or field accesses where they should be written to
        """

    @abc.abstractmethod
    def equilibrium_input_equations_from_init_values(self, **kwargs):
        """
        Returns an equation collection that defines all necessary quantities to compute the equilibrium as function of
        given conserved quantities. Parameters can be names that are given by
        symbol names of :func:`conserved_quantities`.
        For all parameters not specified each implementation should use sensible defaults. For example hydrodynamic
        schemes use density=1 and velocity=0.
        """


class DensityVelocityComputation(AbstractConservedQuantityComputation):
    def __init__(self, stencil, compressible, force_model=None,
                 zeroth_order_moment_symbol=sp.Symbol("rho"),
                 first_order_moment_symbols=sp.symbols("u_:3"),
                 second_order_moment_symbols=sp.symbols("p_:9")):
        dim = stencil.D
        self._stencil = stencil
        self._compressible = compressible
        self._forceModel = force_model
        self._symbolOrder0 = zeroth_order_moment_symbol
        self._symbolsOrder1 = first_order_moment_symbols[:dim]
        self._symbolsOrder2 = second_order_moment_symbols[:(dim * dim)]

    @property
    def conserved_quantities(self):
        return {'density': 1,
                'velocity': self._stencil.D}

    @property
    def compressible(self):
        return self._compressible

    def defined_symbols(self, order='all'):
        if order == 'all':
            return {'density': self._symbolOrder0,
                    'velocity': self._symbolsOrder1}
        elif order == 0:
            return 'density', self._symbolOrder0
        elif order == 1:
            return 'velocity', self._symbolsOrder1
        else:
            return None

    @property
    def zero_centered_pdfs(self):
        return not self._compressible

    @property
    def zeroth_order_moment_symbol(self):
        return self._symbolOrder0

    @property
    def first_order_moment_symbols(self):
        return self._symbolsOrder1

    @property
    def default_values(self):
        result = {self._symbolOrder0: 1}
        for s in self._symbolsOrder1:
            result[s] = 0
        return result

    def equilibrium_input_equations_from_pdfs(self, pdfs, force_substitution=True):
        dim = self._stencil.D
        eq_coll = get_equations_for_zeroth_and_first_order_moment(self._stencil, pdfs, self._symbolOrder0,
                                                                  self._symbolsOrder1[:dim])
        if self._compressible:
            eq_coll = divide_first_order_moments_by_rho(eq_coll, dim)

        if self._forceModel is not None:
            eq_coll = apply_force_model_shift(self._forceModel.equilibrium_velocity_shift,
                                              dim, eq_coll, self._compressible)
            if force_substitution:
                eq_coll = add_symbolic_force_substitutions(eq_coll, self._forceModel._subs_dict_force)
        return eq_coll

    def equilibrium_input_equations_from_init_values(self, density=1, velocity=(0, 0, 0), force_substitution=True):
        dim = self._stencil.D
        zeroth_order_moment = density if self._compressible else density - sp.Rational(1, 1)
        first_order_moments = velocity[:dim]
        vel_offset = [0] * dim

        if self._compressible:
            if self._forceModel is not None:
                vel_offset = self._forceModel.macroscopic_velocity_shift(zeroth_order_moment)
        else:
            if self._forceModel is not None:
                vel_offset = self._forceModel.macroscopic_velocity_shift(sp.Rational(1, 1))
        eqs = [Assignment(self._symbolOrder0, zeroth_order_moment)]

        first_order_moments = [a - b for a, b in zip(first_order_moments, vel_offset)]
        eqs += [Assignment(l, r) for l, r in zip(self._symbolsOrder1, first_order_moments)]

        result = AssignmentCollection(eqs, [])

        if self._forceModel is not None and force_substitution:
            result = add_symbolic_force_substitutions(result, self._forceModel._subs_dict_force)

        return result

    def output_equations_from_pdfs(self, pdfs, output_quantity_names_to_symbols, force_substitution=True):
        dim = self._stencil.D

        ac = get_equations_for_zeroth_and_first_order_moment(self._stencil, pdfs,
                                                             self._symbolOrder0, self._symbolsOrder1,
                                                             self._symbolsOrder2)

        if self._compressible:
            ac = divide_first_order_moments_by_rho(ac, dim)
        else:
            ac = add_density_offset(ac)

        if self._forceModel is not None:
            ac = apply_force_model_shift(self._forceModel.macroscopic_velocity_shift, dim, ac, self._compressible)

        main_assignments = []
        eqs = OrderedDict([(eq.lhs, eq.rhs) for eq in ac.all_assignments])

        if 'density' in output_quantity_names_to_symbols:
            density_output_symbol = output_quantity_names_to_symbols['density']
            if isinstance(density_output_symbol, Field):
                density_output_symbol = density_output_symbol.center
            if density_output_symbol != self._symbolOrder0:
                main_assignments.append(Assignment(density_output_symbol, self._symbolOrder0))
            else:
                main_assignments.append(Assignment(self._symbolOrder0, eqs[self._symbolOrder0]))
                del eqs[self._symbolOrder0]
        if 'velocity' in output_quantity_names_to_symbols:
            vel_output_symbols = output_quantity_names_to_symbols['velocity']
            if isinstance(vel_output_symbols, Field):
                vel_output_symbols = vel_output_symbols.center_vector
            if tuple(vel_output_symbols) != tuple(self._symbolsOrder1):
                main_assignments += [Assignment(a, b) for a, b in zip(vel_output_symbols, self._symbolsOrder1)]
            else:
                for u_i in self._symbolsOrder1:
                    main_assignments.append(Assignment(u_i, eqs[u_i]))
                    del eqs[u_i]
        if 'momentum_density' in output_quantity_names_to_symbols:
            # get zeroth and first moments again - force-shift them if necessary
            # and add their values directly to the main equations assuming that subexpressions are already in
            # main equation collection
            # Is not optimal when velocity and momentum_density are calculated together,
            # but this is usually not the case
            momentum_density_output_symbols = output_quantity_names_to_symbols['momentum_density']
            mom_density_eq_coll = get_equations_for_zeroth_and_first_order_moment(self._stencil, pdfs,
                                                                                  self._symbolOrder0,
                                                                                  self._symbolsOrder1)
            if self._forceModel is not None:
                mom_density_eq_coll = apply_force_model_shift(self._forceModel.macroscopic_momentum_density_shift, dim,
                                                              mom_density_eq_coll, self._compressible)

            for sym, val in zip(momentum_density_output_symbols, mom_density_eq_coll.main_assignments[1:]):
                main_assignments.append(Assignment(sym, val.rhs))
        if 'moment0' in output_quantity_names_to_symbols:
            moment0_output_symbol = output_quantity_names_to_symbols['moment0']
            if isinstance(moment0_output_symbol, Field):
                moment0_output_symbol = moment0_output_symbol.center
            main_assignments.append(Assignment(moment0_output_symbol, sum(pdfs)))
        if 'moment1' in output_quantity_names_to_symbols:
            moment1_output_symbol = output_quantity_names_to_symbols['moment1']
            if isinstance(moment1_output_symbol, Field):
                moment1_output_symbol = moment1_output_symbol.center_vector
            main_assignments.extend([Assignment(lhs, sum(d[i] * pdf for d, pdf in zip(self._stencil, pdfs)))
                                     for i, lhs in enumerate(moment1_output_symbol)])
        if 'moment2' in output_quantity_names_to_symbols:
            moment2_output_symbol = output_quantity_names_to_symbols['moment2']
            if isinstance(moment2_output_symbol, Field):
                moment2_output_symbol = moment2_output_symbol.center_vector
            for i, p in enumerate(moment2_output_symbol):
                main_assignments.append(Assignment(p, eqs[self._symbolsOrder2[i]]))
                del eqs[self._symbolsOrder2[i]]

        ac = ac.copy(main_assignments, [Assignment(a, b) for a, b in eqs.items()])
        if self._forceModel is not None and force_substitution:
            ac = add_symbolic_force_substitutions(ac, self._forceModel._subs_dict_force)

        return ac.new_without_unused_subexpressions()

    def __repr__(self):
        return "ConservedValueComputation for %s" % (", ".join(self.conserved_quantities.keys()),)


# -----------------------------------------  Helper functions ----------------------------------------------------------


def get_equations_for_zeroth_and_first_order_moment(stencil, symbolic_pdfs, symbolic_zeroth_moment,
                                                    symbolic_first_moments, symbolic_second_moments=None):
    r"""
    Returns an equation system that computes the zeroth and first order moments with the least amount of operations

    The first equation of the system is equivalent to

    .. math :

        \rho = \sum_{d \in S} f_d
        u_j = \sum_{d \in S} f_d u_jd
        p_j = \sum_{d \in S} {d \in S} f_d u_jd

    Args:
        stencil: called :math:`S` above
        symbolic_pdfs: called :math:`f` above
        symbolic_zeroth_moment:  called :math:`\rho` above
        symbolic_first_moments: called :math:`u` above
        symbolic_second_moments: called :math:`p` above
    """

    def filter_out_plus_terms(expr):
        result = 0
        for term in expr.args:
            if not type(term) is sp.Mul:
                result += term
        return result

    dim = stencil.D

    subexpressions = list()
    pdf_sum = sum(symbolic_pdfs)
    u = [0] * dim
    for f, offset in zip(symbolic_pdfs, stencil):
        for i in range(dim):
            u[i] += f * int(offset[i])

    p = [0] * dim * dim
    for f, offset in zip(symbolic_pdfs, stencil):
        for i in range(dim):
            for j in range(dim):
                p[dim * i + j] += f * int(offset[i]) * int(offset[j])

    plus_terms = [set(filter_out_plus_terms(u_i).args) for u_i in u]

    velo_terms = sp.symbols(f"vel:{dim}Term")
    for i in range(dim):
        rhs = plus_terms[i]
        for j in range(i):
            rhs -= plus_terms[j]
        eq = Assignment(velo_terms[i], sum(rhs))
        subexpressions.append(eq)
        if len(rhs) == 0:  # if one of the substitutions is not found the simplification can not be applied
            subexpressions = []
            break

    for subexpression in subexpressions:
        pdf_sum = pdf_sum.subs(subexpression.rhs, subexpression.lhs)

    if len(subexpressions) > 0:
        for i in range(dim):
            u[i] = u[i].subs(subexpressions[i].rhs, subexpressions[i].lhs)

    equations = []
    equations += [Assignment(symbolic_zeroth_moment, pdf_sum)]
    equations += [Assignment(u_i_sym, u_i) for u_i_sym, u_i in zip(symbolic_first_moments, u)]
    if symbolic_second_moments:
        equations += [Assignment(symbolic_second_moments[i], p[i]) for i in range(dim ** 2)]

    return AssignmentCollection(equations, subexpressions)


def divide_first_order_moments_by_rho(assignment_collection, dim):
    r"""
    Assumes that the equations of the passed equation collection are the following
        - rho = f_0  + f_1 + ...
        - u_0 = ...
        - u_1 = ...
    Returns a new equation collection where the u terms (first order moments) are divided by rho.
    The dim parameter specifies the number of first order moments. All subsequent equations are just copied over.
    """
    old_eqs = assignment_collection.main_assignments
    rho = old_eqs[0].lhs
    new_first_order_moment_eq = [Assignment(eq.lhs, eq.rhs / rho) for eq in old_eqs[1:dim + 1]]
    new_eqs = [old_eqs[0]] + new_first_order_moment_eq + old_eqs[dim + 1:]
    return assignment_collection.copy(new_eqs)


def add_density_offset(assignment_collection, offset=sp.Rational(1, 1)):
    r"""
    Assumes that first equation is the density (zeroth moment). Changes the density equations by adding offset to it.
    """
    old_eqs = assignment_collection.main_assignments
    new_density = Assignment(old_eqs[0].lhs, old_eqs[0].rhs + offset)
    return assignment_collection.copy([new_density] + old_eqs[1:])


def apply_force_model_shift(shift_func, dim, assignment_collection, compressible, reverse=False):
    """
    Modifies the first order moment equations in assignment collection according to the force model shift.
    It is applied if force model has a method named shift_member_name. The equations 1: dim+1 of the passed
    equation collection are assumed to be the velocity equations.

    Args:
        shift_func: shift function which is applied. See lbmpy.forcemodels.AbstractForceModel for details
        dim: number of spatial dimensions
        assignment_collection: assignment collection containing the conserved quantity computation
        compressible: True if a compressible LB method is used. Otherwise the Helmholtz decomposition was applied
                      for rho
        reverse: If True the sign of the shift is flipped
    """
    old_eqs = assignment_collection.main_assignments
    density = old_eqs[0].lhs if compressible else sp.Rational(1, 1)
    old_vel_eqs = old_eqs[1:dim + 1]
    vel_offsets = shift_func(density)
    if reverse:
        vel_offsets = [-v for v in vel_offsets]
    shifted_velocity_eqs = [Assignment(old_eq.lhs, old_eq.rhs + offset)
                            for old_eq, offset in zip(old_vel_eqs, vel_offsets)]
    new_eqs = [old_eqs[0]] + shifted_velocity_eqs + old_eqs[dim + 1:]
    return assignment_collection.copy(new_eqs)


def add_symbolic_force_substitutions(assignment_collection, subs_dict):
    """
    Every force model holds a symbolic representation of the forcing terms internally. This function adds the
    equations for the D-dimensional force vector to the symbolic replacements

    Args:
        assignment_collection: assignment collection which will be modified
        subs_dict: substitution dict which can be obtained from the force model
    """
    old_eqs = assignment_collection.subexpressions
    subs_equations = []
    for key, value in zip(subs_dict.keys(), subs_dict.values()):
        subs_equations.append(Assignment(key, value))

    new_eqs = subs_equations + old_eqs
    return assignment_collection.copy(main_assignments=None, subexpressions=new_eqs)
