import pystencils as ps
import sympy as sp
import numpy as np

from pystencils.boundaries.boundaryconditions import Boundary
from pystencils.stencil import inverse_direction_string, direction_string_to_offset


class OldroydB:
    def __init__(self, dim, u, tau, F, tauflux, tauface, lambda_p, eta_p, vof=True):
        assert not ps.FieldType.is_staggered(u)
        assert not ps.FieldType.is_staggered(tau)
        assert not ps.FieldType.is_staggered(F)
        assert ps.FieldType.is_staggered(tauflux)
        assert ps.FieldType.is_staggered(tauface)
        
        self.dim = dim
        self.u = u
        self.tau = tau
        self.F = F
        self.tauflux = tauflux
        self.tauface_field = tauface
        self.lambda_p = lambda_p
        self.eta_p = eta_p
        
        full_stencil = ["C"] + self.tauflux.staggered_stencil + \
            list(map(inverse_direction_string, self.tauflux.staggered_stencil))
        self.stencil = tuple(map(lambda d: tuple(ps.stencil.direction_string_to_offset(d, self.dim)), full_stencil))
        full_stencil = ["C"] + self.tauface_field.staggered_stencil + \
            list(map(inverse_direction_string, self.tauface_field.staggered_stencil))
        self.force_stencil = tuple(map(lambda d: tuple(ps.stencil.direction_string_to_offset(d, self.dim)),
                                   full_stencil))
        
        self.disc = ps.fd.FVM1stOrder(self.tau, self._flux(), self._source())
        if vof:
            self.vof = ps.fd.VOF(self.tauflux, self.u, self.tau)
        else:
            self.vof = None
    
    def _flux(self):
        return [self.tau.center_vector.applyfunc(lambda t: t * self.u.center_vector[i]) for i in range(self.dim)]
    
    def _source(self):
        gradu = sp.Matrix([[ps.fd.diff(self.u.center_vector[j], i) for j in range(self.dim)] for i in range(self.dim)])
        gamma = gradu + gradu.transpose()
        return self.tau.center_vector * gradu + gradu.transpose() * self.tau.center_vector + \
            (self.eta_p * gamma - self.tau.center_vector) / self.lambda_p
    
    def tauface(self):
        return ps.AssignmentCollection([ps.Assignment(self.tauface_field.staggered_vector_access(d),
                                        (self.tau.center_vector + self.tau.neighbor_vector(d)) / 2)
                                        for d in self.tauface_field.staggered_stencil])
    
    def force(self):
        full_stencil = self.tauface_field.staggered_stencil + \
            list(map(inverse_direction_string, self.tauface_field.staggered_stencil))
        dtau = sp.Matrix([sum([sum([
                          self.tauface_field.staggered_access(d, (i, j)) * direction_string_to_offset(d)[i]
                          for i in range(self.dim)]) / sp.Matrix(direction_string_to_offset(d)).norm()
                         for d in full_stencil]) for j in range(self.dim)])
        A0 = sum([sp.Matrix(direction_string_to_offset(d)).norm() for d in full_stencil])
        return ps.AssignmentCollection(ps.Assignment(self.F.center_vector, dtau / A0 * 2 * self.dim))
    
    def flux(self):
        if self.vof is not None:
            return self.vof
        else:
            return self.disc.discrete_flux(self.tauflux)
    
    def continuity(self):
        cont = self.disc.discrete_continuity(self.tauflux)
        tau_copy = sp.Matrix(self.dim, self.dim, lambda i, j: sp.Symbol("tau_old_%d_%d" % (i, j)))
        tau_subs = {self.tau.center_vector[i, j]: tau_copy[i, j] for i in range(self.dim) for j in range(self.dim)}
        return [ps.Assignment(tau_copy[i, j], self.tau.center_vector[i, j])
                for i in range(self.dim) for j in range(self.dim)] + \
               [ps.Assignment(a.lhs, a.rhs.subs(tau_subs)) for a in cont]


class Flux(Boundary):
    inner_or_boundary = True  # call the boundary condition with the fluid cell
    single_link = False  # needs to be called for all directional fluxes
    
    def __init__(self, stencil, value=None):
        self.stencil = stencil
        self.value = value

    def __call__(self, field, direction_symbol, **kwargs):
        assert ps.FieldType.is_staggered(field)
        
        assert all([s == 0 for s in self.stencil[0]])
        accesses = [field.staggered_vector_access(ps.stencil.offset_to_direction_string(d))
                    for d in self.stencil[1:]]
        conds = [sp.Equality(direction_symbol, d + 1) for d in range(len(accesses))]
        
        if self.value is None:
            val = sp.Matrix(np.zeros(accesses[0].shape, dtype=int))
        else:
            val = self.value
        
        # use conditional
        conditional = None
        for a, c, d in zip(accesses, conds, self.stencil[1:]):
            d = ps.stencil.offset_to_direction_string(d)
            assignments = []
            for i in range(len(a)):
                fac = 1
                if ps.FieldType.is_staggered_flux(field) and type(a[i]) is sp.Mul and a[i].args[0] == -1:
                    fac = a[i].args[0]
                    a[i] *= a[i].args[0]
                assignments.append(ps.Assignment(a[i], fac * val[i]))
            if len(assignments) > 0:
                conditional = ps.astnodes.Conditional(c,
                                                      ps.astnodes.Block(assignments),
                                                      conditional)
        return [conditional]

    def __hash__(self):
        return hash((Flux, self.stencil, self.value))

    def __eq__(self, other):
        return type(other) == Flux and other.stencil == self.stencil and self.value == other.value


class Extrapolation(Boundary):
    inner_or_boundary = True  # call the boundary condition with the fluid cell
    single_link = False  # needs to be called for all directional fluxes

    def __init__(self, stencil, src_field, order):
        self.stencil = stencil
        self.src = src_field
        if order == 0:
            self.weights = (1,)
        elif order == 1:
            self.weights = (sp.Rational(3, 2), - sp.Rational(1, 2))
        elif order == 2:
            self.weights = (sp.Rational(15, 8), - sp.Rational(10, 8), sp.Rational(3, 8))
        else:
            raise NotImplementedError("weights are not known for extrapolation orders > 2")

    def __call__(self, field, direction_symbol, **kwargs):
        assert ps.FieldType.is_staggered(field)
        
        assert all([s == 0 for s in self.stencil[0]])
        accesses = [field.staggered_vector_access(ps.stencil.offset_to_direction_string(d))
                    for d in self.stencil[1:]]
        conds = [sp.Equality(direction_symbol, d + 1) for d in range(len(accesses))]
        
        # use conditional
        conditional = None
        for a, c, o in zip(accesses, conds, self.stencil[1:]):
            assignments = []
            src = [self.src.neighbor_vector(tuple([-1 * n * i for i in o])) for n in range(len(self.weights))]
            interp = self.weights[0] * src[0]
            for i in range(1, len(self.weights)):
                interp += self.weights[i] * src[i]
            for i in range(len(a)):
                fac = 1
                if ps.FieldType.is_staggered_flux(field) and type(a[i]) is sp.Mul and a[i].args[0] == -1:
                    fac = a[i].args[0]
                    a[i] *= a[i].args[0]
                assignments.append(ps.Assignment(a[i], fac * interp[i]))
            if len(assignments) > 0:
                conditional = ps.astnodes.Conditional(ps.data_types.type_all_numbers(c, "int"),
                                                      ps.astnodes.Block(assignments),
                                                      conditional)
        return [conditional]

    def __hash__(self):
        return hash((Extrapolation, self.stencil, self.src, self.weights))

    def __eq__(self, other):
        return type(other) == Extrapolation and other.stencil == self.stencil and \
            other.src == self.src and other.weights == self.weights


class ForceOnBoundary(Boundary):
    inner_or_boundary = False  # call the boundary condition with the boundary cell
    single_link = False  # needs to be called for all directional fluxes
    
    def __init__(self, stencil, force_field):
        self.stencil = stencil
        self.force_field = force_field
        
        assert not ps.FieldType.is_staggered(force_field)

    def __call__(self, face_stress_field, direction_symbol, **kwargs):
        assert ps.FieldType.is_staggered(face_stress_field)
        
        assert all([s == 0 for s in self.stencil[0]])
        accesses = [face_stress_field.staggered_vector_access(ps.stencil.offset_to_direction_string(d))
                    for d in self.stencil[1:]]
        conds = [sp.Equality(direction_symbol, d + 1) for d in range(len(accesses))]
        
        # use conditional
        conditional = None
        for a, c, o in zip(accesses, conds, self.stencil[1:]):
            assignments = ps.Assignment(self.force_field.center_vector,
                                        self.force_field.center_vector + 1 * a.transpose() * sp.Matrix(o))
            conditional = ps.astnodes.Conditional(ps.data_types.type_all_numbers(c, "int"),
                                                  ps.astnodes.Block(assignments),
                                                  conditional)
        return [conditional]

    def __hash__(self):
        return hash((ForceOnBoundary, self.stencil, self.force_field))

    def __eq__(self, other):
        return type(other) == ForceOnBoundary and other.stencil == self.stencil and \
            other.force_field == self.force_field
