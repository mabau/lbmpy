from warnings import warn
from dataclasses import dataclass
from typing import Type

import itertools
import operator
from collections import OrderedDict
from functools import reduce

import sympy as sp

from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.equilibrium import ContinuousHydrodynamicMaxwellian, DiscreteHydrodynamicMaxwellian

from lbmpy.methods.default_moment_sets import cascaded_moment_sets_literature

from lbmpy.moment_transforms import CentralMomentsToCumulantsByGeneratingFunc

from .conservedquantitycomputation import DensityVelocityComputation

from .momentbased.momentbasedmethod import MomentBasedLbMethod
from .momentbased.centralmomentbasedmethod import CentralMomentBasedLbMethod
from .cumulantbased import CumulantBasedLbMethod
from lbmpy.moment_transforms import (
    AbstractMomentTransform, BinomialChimeraTransform, PdfsToMomentsByChimeraTransform)
from lbmpy.moment_transforms.rawmomenttransforms import AbstractRawMomentTransform
from lbmpy.moment_transforms.centralmomenttransforms import AbstractCentralMomentTransform

from lbmpy.moments import (
    MOMENT_SYMBOLS, exponents_to_polynomial_representations,
    get_default_moment_set_for_stencil, gram_schmidt, is_even, moments_of_order,
    moments_up_to_component_order, sort_moments_into_groups_of_same_order,
    is_bulk_moment, is_shear_moment, get_order)

from lbmpy.relaxationrates import relaxation_rate_from_magic_number
from lbmpy.enums import Stencil, CollisionSpace
from lbmpy.stencils import LBStencil
from pystencils.sympyextensions import common_denominator


@dataclass
class CollisionSpaceInfo:
    """
    This class encapsulates necessary details about a method's collision space.
    """
    collision_space: CollisionSpace
    """
    The method's collision space.
    """
    raw_moment_transform_class: Type[AbstractRawMomentTransform] = None
    """
    Python class that determines how PDFs are transformed to raw moment space. If left as 'None', this parameter
    will be inferred from `collision_space`, defaulting to 
    :class:`lbmpy.moment_transforms.PdfsToMomentsByChimeraTransform`
    if `CollisionSpace.RAW_MOMENTS` is given, or `None` otherwise.
    """
    central_moment_transform_class: Type[AbstractCentralMomentTransform] = None
    """
    Python class that determines how PDFs are transformed to central moment space. If left as 'None', this parameter
    will be inferred from `collision_space`, defaulting to 
    :class:`lbmpy.moment_transforms.BinomialChimeraTransform`
    if `CollisionSpace.CENTRAL_MOMENTS` or `CollisionSpace.CUMULANTS` is given, or `None` otherwise.
    """
    cumulant_transform_class: Type[AbstractMomentTransform] = None
    """
    Python class that determines how central moments are transformed to cumulant space. If left as 'None', this 
    parameter will be inferred from `collision_space`, defaulting to 
    :class:`lbmpy.moment_transforms.CentralMomentsToCumulantsByGeneratingFunc`
    if `CollisionSpace.CUMULANTS` is given, or `None` otherwise.
    """

    def __post_init__(self):
        if self.collision_space == CollisionSpace.RAW_MOMENTS and self.raw_moment_transform_class is None:
            self.raw_moment_transform_class = PdfsToMomentsByChimeraTransform
        if self.collision_space in (CollisionSpace.CENTRAL_MOMENTS, CollisionSpace.CUMULANTS) \
                and self.central_moment_transform_class is None:
            self.central_moment_transform_class = BinomialChimeraTransform
        if self.collision_space == CollisionSpace.CUMULANTS and self.cumulant_transform_class is None:
            self.cumulant_transform_class = CentralMomentsToCumulantsByGeneratingFunc


def create_with_discrete_maxwellian_equilibrium(stencil, moment_to_relaxation_rate_dict,
                                                compressible=False, zero_centered=False, delta_equilibrium=False,
                                                force_model=None, equilibrium_order=2, c_s_sq=sp.Rational(1, 3),
                                                **kwargs):
    r"""Creates a moment-based LBM by taking a dictionary of moments with corresponding relaxation rates.

    These moments are relaxed against the moments of the discrete Maxwellian distribution
    (see :class:`lbmpy.equilibrium.DiscreteHydrodynamicMaxwellian`).

    Internally, this method calls :func:`lbmpy.methods.create_from_equilibrium`.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStenil`
        moment_to_relaxation_rate_dict: dict that has as many entries as the stencil. Each moment, which can be
                                        represented by an exponent tuple or in polynomial form
                                        (see `lbmpy.moments`), is mapped to a relaxation rate.
        compressible: If `False`, the incompressible equilibrium formulation is used to better approximate the
                      incompressible Navier-Stokes equations. Otherwise, the default textbook equilibrium is used.
        zero_centered: If `True`, the zero-centered storage format for PDFs is used, storing only their deviation from
                       the background distribution (given by the lattice weights).
        delta_equilibrium: Takes effect only if zero-centered storage is used. If `True`, the equilibrium distribution
                           is also given only as its deviation from the background distribution.
        force_model: instance of :class:`lbmpy.forcemodels.AbstractForceModel`, or None if no external forces are 
                     present.
        equilibrium_order: approximation order of macroscopic velocity :math:`\mathbf{u}` in the equilibrium
        c_s_sq: Speed of sound squared
        kwargs: See :func:`lbmpy.methods.create_from_equilibrium`

    Returns:
        Instance of a subclass of :class:`lbmpy.methods.AbstractLbMethod`.
    """
    cqc = DensityVelocityComputation(stencil, compressible, zero_centered, force_model=force_model, c_s_sq=c_s_sq)
    equilibrium = DiscreteHydrodynamicMaxwellian(stencil, compressible=compressible,
                                                 deviation_only=delta_equilibrium,
                                                 order=equilibrium_order,
                                                 c_s_sq=c_s_sq)
    return create_from_equilibrium(stencil, equilibrium, cqc, moment_to_relaxation_rate_dict,
                                   zero_centered=zero_centered, force_model=force_model, **kwargs)


def create_with_continuous_maxwellian_equilibrium(stencil, moment_to_relaxation_rate_dict,
                                                  compressible=False, zero_centered=False, delta_equilibrium=False,
                                                  force_model=None, equilibrium_order=2, c_s_sq=sp.Rational(1, 3),
                                                  **kwargs):
    r"""
    Creates a moment-based LBM by taking a dictionary of moments with corresponding relaxation rates. 
    These moments are relaxed against the moments of the continuous Maxwellian distribution
    (see :class:`lbmpy.equilibrium.ContinuousHydrodynamicMaxwellian`).

    Internally, this method calls :func:`lbmpy.methods.create_from_equilibrium`.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStenil`
        moment_to_relaxation_rate_dict: dict that has as many entries as the stencil. Each moment, which can be
                                        represented by an exponent tuple or in polynomial form
                                        (see `lbmpy.moments`), is mapped to a relaxation rate.
        compressible: If `False`, the incompressible equilibrium formulation is used to better approximate the
                      incompressible Navier-Stokes equations. Otherwise, the default textbook equilibrium is used.
        zero_centered: If `True`, the zero-centered storage format for PDFs is used, storing only their deviation from 
                       the background distribution (given by the lattice weights).
        delta_equilibrium: Takes effect only if zero-centered storage is used. If `True`, the equilibrium 
                           distribution is also given only as its deviation from the background distribution.
        force_model: Instance of :class:`lbmpy.forcemodels.AbstractForceModel`, or None if no external forces 
                     are present.
        equilibrium_order: approximation order of macroscopic velocity :math:`\mathbf{u}` in the equilibrium
        c_s_sq: Speed of sound squared
        kwargs: See :func:`lbmpy.methods.create_from_equilibrium`

    Returns:
        Instance of a subclass of :class:`lbmpy.methods.AbstractLbMethod`.
    """
    cqc = DensityVelocityComputation(stencil, compressible, zero_centered, force_model=force_model, c_s_sq=c_s_sq)
    equilibrium = ContinuousHydrodynamicMaxwellian(dim=stencil.D, compressible=compressible,
                                                   deviation_only=delta_equilibrium,
                                                   order=equilibrium_order,
                                                   c_s_sq=c_s_sq)
    return create_from_equilibrium(stencil, equilibrium, cqc, moment_to_relaxation_rate_dict,
                                   zero_centered=zero_centered, force_model=force_model, **kwargs)


def create_from_equilibrium(stencil, equilibrium, conserved_quantity_computation,
                            moment_to_relaxation_rate_dict,
                            collision_space_info=CollisionSpaceInfo(CollisionSpace.POPULATIONS),
                            zero_centered=False, force_model=None):
    r"""
    Creates a lattice Boltzmann method in either population, moment, or central moment space, from a given
    discrete velocity set and equilibrium distribution. 

    This function takes a stencil, an equilibrium distribution, an appropriate conserved quantity computation
    instance, a dictionary mapping moment polynomials to their relaxation rates, and a collision space info
    determining the desired collision space. It returns a method instance relaxing the given moments against
    their equilibrium values computed from the given distribution, in the given collision space.

    Args:
        stencil: Instance of :class:`lbmpy.stencils.LBStencil`
        equilibrium: Instance of a subclass of :class:`lbmpy.equilibrium.AbstractEquilibrium`.
        conserved_quantity_computation: Instance of a subclass of 
                                        :class:`lbmpy.methods.AbstractConservedQuantityComputation`,
                                        which must provide equations for the conserved quantities used in
                                        the equilibrium.
        moment_to_relaxation_rate_dict: Dictionary mapping moment polynomials to relaxation rates.
        collision_space_info: Instance of :class:`CollisionSpaceInfo`, defining the method's desired collision space
                              and the manner of transformation to this space. Cumulant-based methods are not supported
                              yet.
        zero_centered: Whether or not the zero-centered storage format should be used. If `True`, the given equilibrium
                       must either be a deviation-only formulation, or must provide a background distribution for PDFs
                       to be centered around.
        force_model: Instance of :class:`lbmpy.forcemodels.AbstractForceModel`, or None if no external forces are
                     present.
    """
    if any(isinstance(moment_to_relaxation_rate_dict, t) for t in (sp.Symbol, float, int)):
        moment_to_relaxation_rate_dict = {m: moment_to_relaxation_rate_dict
                                          for m in get_default_moment_set_for_stencil(stencil)}

    mom_to_rr_dict = OrderedDict(moment_to_relaxation_rate_dict)
    assert len(mom_to_rr_dict) == stencil.Q, "The number of moments has to be equal to the number of stencil entries"

    cqc = conserved_quantity_computation
    cspace = collision_space_info

    if cspace.collision_space == CollisionSpace.POPULATIONS:
        return MomentBasedLbMethod(stencil, equilibrium, mom_to_rr_dict, conserved_quantity_computation=cqc,
                                   force_model=force_model, zero_centered=zero_centered,
                                   moment_transform_class=None)
    elif cspace.collision_space == CollisionSpace.RAW_MOMENTS:
        return MomentBasedLbMethod(stencil, equilibrium, mom_to_rr_dict, conserved_quantity_computation=cqc,
                                   force_model=force_model, zero_centered=zero_centered,
                                   moment_transform_class=cspace.raw_moment_transform_class)
    elif cspace.collision_space == CollisionSpace.CENTRAL_MOMENTS:
        return CentralMomentBasedLbMethod(stencil, equilibrium, mom_to_rr_dict, conserved_quantity_computation=cqc,
                                          force_model=force_model, zero_centered=zero_centered,
                                          central_moment_transform_class=cspace.central_moment_transform_class)
    elif cspace.collision_space == CollisionSpace.CUMULANTS:
        return CumulantBasedLbMethod(stencil, equilibrium, mom_to_rr_dict, conserved_quantity_computation=cqc,
                                     force_model=force_model, zero_centered=zero_centered,
                                     central_moment_transform_class=cspace.central_moment_transform_class,
                                     cumulant_transform_class=cspace.cumulant_transform_class)


# ------------------------------------ SRT / TRT/ MRT Creators ---------------------------------------------------------


def create_srt(stencil, relaxation_rate, continuous_equilibrium=True, **kwargs):
    r"""Creates a single relaxation time (SRT) lattice Boltzmann model also known as BGK model.

    Internally calls either :func:`create_with_discrete_maxwellian_equilibrium`
    or :func:`create_with_continuous_maxwellian_equilibrium`, depending on the value of `continuous_equilibrium`.

    If not specified otherwise, collision equations will be derived in population space.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rate: relaxation rate (inverse of the relaxation time)
                        usually called :math:`\omega` in LBM literature
        continuous_equilibrium: determines if the discrete or continuous maxwellian equilibrium is
                        used to compute the equilibrium moments

    Returns:
        :class:`lbmpy.methods.momentbased.MomentBasedLbMethod` instance
    """
    continuous_equilibrium = _deprecate_maxwellian_moments(continuous_equilibrium, kwargs)
    check_and_set_mrt_space(CollisionSpace.POPULATIONS)
    moments = get_default_moment_set_for_stencil(stencil)
    rr_dict = OrderedDict([(m, relaxation_rate) for m in moments])
    if continuous_equilibrium:
        return create_with_continuous_maxwellian_equilibrium(stencil, rr_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_equilibrium(stencil, rr_dict, **kwargs)


def create_trt(stencil, relaxation_rate_even_moments, relaxation_rate_odd_moments,
               continuous_equilibrium=True, **kwargs):
    """
    Creates a two relaxation time (TRT) lattice Boltzmann model, where even and odd moments are relaxed differently.
    In the SRT model the exact wall position of no-slip boundaries depends on the viscosity, the TRT method does not
    have this problem.

    Parameters are similar to :func:`lbmpy.methods.create_srt`, but instead of one relaxation rate there are
    two relaxation rates: one for even moments (determines viscosity) and one for odd moments.
    If unsure how to choose the odd relaxation rate, use the function :func:`lbmpy.methods.create_trt_with_magic_number`

    Internally calls either :func:`create_with_discrete_maxwellian_equilibrium`
    or :func:`create_with_continuous_maxwellian_equilibrium`, depending on the value of `continuous_equilibrium`.

    If not specified otherwise, collision equations will be derived in population space.

    Returns:
        :class:`lbmpy.methods.momentbased.MomentBasedLbMethod` instance
    """
    continuous_equilibrium = _deprecate_maxwellian_moments(continuous_equilibrium, kwargs)
    check_and_set_mrt_space(CollisionSpace.POPULATIONS)
    moments = get_default_moment_set_for_stencil(stencil)
    rr_dict = OrderedDict([(m, relaxation_rate_even_moments if is_even(m) else relaxation_rate_odd_moments)
                           for m in moments])
    if continuous_equilibrium:
        return create_with_continuous_maxwellian_equilibrium(stencil, rr_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_equilibrium(stencil, rr_dict, **kwargs)


def create_trt_with_magic_number(stencil, relaxation_rate, magic_number=sp.Rational(3, 16), **kwargs):
    r"""
    Creates a two relaxation time (TRT) lattice Boltzmann method, where the relaxation time for odd moments is
    determines from the even moment relaxation time and a "magic number".
    For possible parameters see :func:`lbmpy.methods.create_trt`

    Internally calls either :func:`create_with_discrete_maxwellian_equilibrium`
    or :func:`create_with_continuous_maxwellian_equilibrium`, depending on the value of `continuous_equilibrium`.

    If not specified otherwise, collision equations will be derived in population space.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rate: relaxation rate (inverse of the relaxation time)
                        usually called :math:`\omega` in LBM literature
        magic_number: magic number which is used to calculate the relxation rate for the odd moments.

    Returns:
        :class:`lbmpy.methods.momentbased.MomentBasedLbMethod` instance
    """
    rr_odd = relaxation_rate_from_magic_number(relaxation_rate, magic_number)
    return create_trt(stencil, relaxation_rate_even_moments=relaxation_rate,
                      relaxation_rate_odd_moments=rr_odd, **kwargs)


def create_mrt_raw(stencil, relaxation_rates, continuous_equilibrium=True, **kwargs):
    r"""
    Creates a MRT method using non-orthogonalized moments.

    Internally calls either :func:`create_with_discrete_maxwellian_equilibrium`
    or :func:`create_with_continuous_maxwellian_equilibrium`, depending on the value of `continuous_equilibrium`.

    If not specified otherwise, collision equations will be derived in raw moment space.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates (inverse of the relaxation times) for each moment
        continuous_equilibrium: determines if the discrete or continuous maxwellian equilibrium is
                        used to compute the equilibrium moments.
    Returns:
        :class:`lbmpy.methods.momentbased.MomentBasedLbMethod` instance
    """
    continuous_equilibrium = _deprecate_maxwellian_moments(continuous_equilibrium, kwargs)
    check_and_set_mrt_space(CollisionSpace.RAW_MOMENTS)
    moments = get_default_moment_set_for_stencil(stencil)
    nested_moments = [(c,) for c in moments]
    rr_dict = _get_relaxation_info_dict(relaxation_rates, nested_moments, stencil.D)
    if continuous_equilibrium:
        return create_with_continuous_maxwellian_equilibrium(stencil, rr_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_equilibrium(stencil, rr_dict, **kwargs)


def create_central_moment(stencil, relaxation_rates, nested_moments=None,
                          continuous_equilibrium=True, **kwargs):
    r"""
    Creates moment based LB method where the collision takes place in the central moment space.

    Internally calls either :func:`create_with_discrete_maxwellian_equilibrium`
    or :func:`create_with_continuous_maxwellian_equilibrium`, depending on the value of `continuous_equilibrium`.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates (inverse of the relaxation times) for each moment
        nested_moments: a list of lists of modes, grouped by common relaxation times.
        continuous_equilibrium: determines if the discrete or continuous maxwellian equilibrium is
                        used to compute the equilibrium moments.
    Returns:
        :class:`lbmpy.methods.momentbased.CentralMomentBasedLbMethod` instance
    """
    continuous_equilibrium = _deprecate_maxwellian_moments(continuous_equilibrium, kwargs)

    kwargs.setdefault('collision_space_info', CollisionSpaceInfo(CollisionSpace.CENTRAL_MOMENTS))
    if kwargs['collision_space_info'].collision_space != CollisionSpace.CENTRAL_MOMENTS:
        raise ValueError("Central moment-based methods can only be derived in central moment space.")

    if nested_moments and not isinstance(nested_moments[0], list):
        nested_moments = list(sort_moments_into_groups_of_same_order(nested_moments).values())
        second_order_moments = nested_moments[2]
        bulk_moment = [m for m in second_order_moments if is_bulk_moment(m, stencil.D)]
        shear_moments = [m for m in second_order_moments if is_shear_moment(m, stencil.D)]
        assert len(shear_moments) + len(bulk_moment) == len(second_order_moments)
        nested_moments[2] = shear_moments
        nested_moments.insert(3, bulk_moment)

    if not nested_moments:
        nested_moments = cascaded_moment_sets_literature(stencil)

    rr_dict = _get_relaxation_info_dict(relaxation_rates, nested_moments, stencil.D)

    if continuous_equilibrium:
        return create_with_continuous_maxwellian_equilibrium(stencil, rr_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_equilibrium(stencil, rr_dict, **kwargs)


def create_trt_kbc(dim, shear_relaxation_rate, higher_order_relaxation_rate, method_name='KBC-N4',
                   continuous_equilibrium=True, **kwargs):
    """
    Creates a method with two relaxation rates, one for lower order moments which determines the viscosity and
    one for higher order moments. In entropic models this second relaxation rate is chosen subject to an entropy
    condition. Which moments are relaxed by which rate is determined by the method_name

    Internally calls either :func:`create_with_discrete_maxwellian_equilibrium`
    or :func:`create_with_continuous_maxwellian_equilibrium`, depending on the value of `continuous_equilibrium`.

    If not specified otherwise, collision equations will be derived in population space.

    Args:
        dim: 2 or 3, leads to stencil D2Q9 or D3Q27
        shear_relaxation_rate: relaxation rate that determines viscosity
        higher_order_relaxation_rate: relaxation rate for higher order moments
        method_name: string 'KBC-Nx' where x can be an number from 1 to 4, for details see
                    "Karlin 2015: Entropic multi relaxation lattice Boltzmann models for turbulent flows"
        continuous_equilibrium: determines if the discrete or continuous maxwellian equilibrium is
                        used to compute the equilibrium moments.
    """
    continuous_equilibrium = _deprecate_maxwellian_moments(continuous_equilibrium, kwargs)
    check_and_set_mrt_space(CollisionSpace.POPULATIONS)

    def product(iterable):
        return reduce(operator.mul, iterable, 1)

    the_moment = MOMENT_SYMBOLS[:dim]

    rho = [sp.Rational(1, 1)]
    velocity = list(the_moment)

    shear_tensor_off_diagonal = [product(t) for t in itertools.combinations(the_moment, 2)]
    shear_tensor_diagonal = [m_i * m_i for m_i in the_moment]
    shear_tensor_trace = sum(shear_tensor_diagonal)
    shear_tensor_trace_free_diagonal = [dim * d - shear_tensor_trace for d in shear_tensor_diagonal]

    poly_repr = exponents_to_polynomial_representations
    energy_transport_tensor = list(poly_repr([a for a in moments_of_order(3, dim, True)
                                              if 3 not in a]))

    explicitly_defined = set(rho + velocity + shear_tensor_off_diagonal
                             + shear_tensor_diagonal + energy_transport_tensor)
    rest = list(set(poly_repr(moments_up_to_component_order(2, dim))) - explicitly_defined)
    assert len(rest) + len(explicitly_defined) == 3 ** dim

    # naming according to paper Karlin 2015: Entropic multi relaxation lattice Boltzmann models for turbulent flows
    d = shear_tensor_off_diagonal + shear_tensor_trace_free_diagonal[:-1]
    t = [shear_tensor_trace]
    q = energy_transport_tensor
    if method_name == 'KBC-N1':
        decomposition = [d, t + q + rest]
    elif method_name == 'KBC-N2':
        decomposition = [d + t, q + rest]
    elif method_name == 'KBC-N3':
        decomposition = [d + q, t + rest]
    elif method_name == 'KBC-N4':
        decomposition = [d + t + q, rest]
    else:
        raise ValueError("Unknown model. Supported models KBC-Nx where x in (1,2,3,4)")

    omega_s, omega_h = shear_relaxation_rate, higher_order_relaxation_rate
    shear_part, rest_part = decomposition

    relaxation_rates = [omega_s] + \
                       [omega_s] * len(velocity) + \
                       [omega_s] * len(shear_part) + \
                       [omega_h] * len(rest_part)

    stencil = LBStencil(Stencil.D2Q9) if dim == 2 else LBStencil(Stencil.D3Q27)
    all_moments = rho + velocity + shear_part + rest_part
    moment_to_rr = OrderedDict((m, rr) for m, rr in zip(all_moments, relaxation_rates))

    if continuous_equilibrium:
        return create_with_continuous_maxwellian_equilibrium(stencil, moment_to_rr, **kwargs)
    else:
        return create_with_discrete_maxwellian_equilibrium(stencil, moment_to_rr, **kwargs)


def create_mrt_orthogonal(stencil, relaxation_rates, continuous_equilibrium=True, weighted=None,
                          nested_moments=None, **kwargs):
    r"""
    Returns an orthogonal multi-relaxation time model for the stencils D2Q9, D3Q15, D3Q19 and D3Q27.
    These MRT methods are just one specific version - there are many MRT methods possible for all these stencils
    which differ by the linear combination of moments and the grouping into equal relaxation times.
    To create a generic MRT method use `create_with_discrete_maxwellian_equilibrium`

    Internally calls either :func:`create_with_discrete_maxwellian_equilibrium`
    or :func:`create_with_continuous_maxwellian_equilibrium`, depending on the value of `continuous_equilibrium`.

    If not specified otherwise, collision equations will be derived in raw moment space.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates for the moments
        continuous_equilibrium: determines if the discrete or continuous maxwellian equilibrium is
                                               used to compute the equilibrium moments
        weighted: whether to use weighted or unweighted orthogonality
        nested_moments: a list of lists of modes, grouped by common relaxation times. If this argument is not provided,
                        Gram-Schmidt orthogonalization of the default modes is performed. The default modes equal the
                        raw moments except for the separation of the shear and bulk viscosity.
    """
    continuous_equilibrium = _deprecate_maxwellian_moments(continuous_equilibrium, kwargs)
    check_and_set_mrt_space(CollisionSpace.RAW_MOMENTS)

    if weighted:
        weights = get_weights(stencil, sp.Rational(1, 3))
    else:
        weights = None

    if not nested_moments:
        moments = get_default_moment_set_for_stencil(stencil)
        x, y, z = MOMENT_SYMBOLS
        if stencil.D == 2:
            diagonal_viscous_moments = [x ** 2 + y ** 2, x ** 2]
        else:
            diagonal_viscous_moments = [x ** 2 + y ** 2 + z ** 2, x ** 2, y ** 2 - z ** 2]

        for i, d in enumerate(MOMENT_SYMBOLS[:stencil.D]):
            if d ** 2 in moments:
                moments[moments.index(d ** 2)] = diagonal_viscous_moments[i]
        orthogonal_moments = gram_schmidt(moments, stencil, weights)
        orthogonal_moments_scaled = [e * common_denominator(e) for e in orthogonal_moments]
        nested_moments = list(sort_moments_into_groups_of_same_order(orthogonal_moments_scaled).values())
        # second order moments: separate bulk from shear
        second_order_moments = nested_moments[2]
        bulk_moment = [m for m in second_order_moments if is_bulk_moment(m, stencil.D)]
        shear_moments = [m for m in second_order_moments if is_shear_moment(m, stencil.D)]
        assert len(shear_moments) + len(bulk_moment) == len(second_order_moments)
        nested_moments[2] = shear_moments
        nested_moments.insert(3, bulk_moment)

    moment_to_relaxation_rate_dict = _get_relaxation_info_dict(relaxation_rates, nested_moments, stencil.D)

    if continuous_equilibrium:
        return create_with_continuous_maxwellian_equilibrium(stencil,
                                                             moment_to_relaxation_rate_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_equilibrium(stencil,
                                                           moment_to_relaxation_rate_dict, **kwargs)


# ----------------------------------------- Cumulant method creators ---------------------------------------------------

def create_cumulant(stencil, relaxation_rates, cumulant_groups, **kwargs):
    r"""Creates a cumulant-based lattice Boltzmann method.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates for each cumulant group. If None are provided a list of symbolic relaxation
                          rates is created and used. If only a list with one entry is provided this relaxation rate is
                          used for determine the viscosity of the simulation. All other cumulants are relaxed with one.
                          If a cumulant force model is provided the first order cumulants are relaxed with two to ensure
                          that the force is applied correctly to the momentum groups
        cumulant_groups: Nested sequence of polynomial expressions defining the cumulants to be relaxed. All cumulants 
                         within one group are relaxed with the same relaxation rate.
        kwargs: See :func:`create_with_continuous_maxwellian_equilibrium`

    Returns:
        :class:`lbmpy.methods.cumulantbased.CumulantBasedLbMethod` instance
    """
    cumulant_to_rr_dict = _get_relaxation_info_dict(relaxation_rates, cumulant_groups, stencil.D)
    kwargs.setdefault('collision_space_info', CollisionSpaceInfo(CollisionSpace.CUMULANTS))

    if kwargs['collision_space_info'].collision_space != CollisionSpace.CUMULANTS:
        raise ValueError("Cumulant-based methods can only be derived in cumulant space.")

    return create_with_continuous_maxwellian_equilibrium(stencil, cumulant_to_rr_dict, 
                                                         compressible=True, delta_equilibrium=False,
                                                         **kwargs)


def create_with_monomial_cumulants(stencil, relaxation_rates, **kwargs):
    r"""Creates a cumulant lattice Boltzmann model using the given stencil's canonical monomial cumulants.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates for each cumulant group. If None are provided a list of symbolic relaxation
                          rates is created and used. If only a list with one entry is provided this relaxation rate is
                          used for determine the viscosity of the simulation. All other cumulants are relaxed with one.
                          If a cumulant force model is provided the first order cumulants are relaxed with two to ensure
                          that the force is applied correctly to the momentum groups
        kwargs: See :func:`create_cumulant`

    Returns:
        :class:`lbmpy.methods.cumulantbased.CumulantBasedLbMethod` instance
    """
    # Get monomial moments
    cumulants = get_default_moment_set_for_stencil(stencil)
    cumulant_groups = [(c,) for c in cumulants]
    return create_cumulant(stencil, relaxation_rates, cumulant_groups, **kwargs)


def create_with_default_polynomial_cumulants(stencil, relaxation_rates, **kwargs):
    r"""Creates a cumulant lattice Boltzmann model based on the default polynomial set of :cite:`geier2015`.

    Args: See :func:`create_cumulant`.

    Returns:
        :class:`lbmpy.methods.cumulantbased.CumulantBasedLbMethod` instance
    """
    # Get polynomial groups
    cumulant_groups = cascaded_moment_sets_literature(stencil)
    return create_cumulant(stencil, relaxation_rates, cumulant_groups, **kwargs)


def _get_relaxation_info_dict(relaxation_rates, nested_moments, dim):
    r"""Creates a dictionary where each moment is mapped to a relaxation rate.

    Args:
        relaxation_rates: list of relaxation rates which should be used. This can also be a function which
                          takes a moment group in the list of nested moments and returns a list of relaxation rates.
                          This list has to have the length of the moment group and is then used for the moments
                          in the moment group.
        nested_moments: list of lists containing the moments.
        dim: dimension
    """
    result = OrderedDict()

    if callable(relaxation_rates):
        for group in nested_moments:
            rr = iter(relaxation_rates(group))
            for moment in group:
                result[moment] = next(rr)

        return result

    number_of_moments = 0
    shear_moments = 0
    bulk_moments = 0

    for group in nested_moments:
        for moment in group:
            number_of_moments += 1
            if is_shear_moment(moment, dim):
                shear_moments += 1
            if is_bulk_moment(moment, dim):
                bulk_moments += 1

    # if only one relaxation rate is specified it is used as the shear relaxation rate
    if len(relaxation_rates) == 1:
        for group in nested_moments:
            for moment in group:
                if get_order(moment) <= 1:
                    result[moment] = 0.0
                elif is_shear_moment(moment, dim):
                    result[moment] = relaxation_rates[0]
                else:
                    result[moment] = 1.0

    # if relaxation rate for each moment is specified they are all used
    if len(relaxation_rates) == number_of_moments:
        rr_iter = iter(relaxation_rates)
        for group in nested_moments:
            for moment in group:
                rr = next(rr_iter)
                result[moment] = rr

    # Fallback case, relaxes each group with the same relaxation rate and separates shear and bulk moments
    next_rr = True
    if len(relaxation_rates) != 1 and len(relaxation_rates) != number_of_moments:
        try:
            rr_iter = iter(relaxation_rates)
            if shear_moments > 0:
                shear_rr = next(rr_iter)
            if bulk_moments > 0:
                bulk_rr = next(rr_iter)
            for group in nested_moments:
                if next_rr:
                    rr = next(rr_iter)
                next_rr = False
                for moment in group:
                    if get_order(moment) <= 1:
                        result[moment] = 0.0
                    elif is_shear_moment(moment, dim):
                        result[moment] = shear_rr
                    elif is_bulk_moment(moment, dim):
                        result[moment] = bulk_rr
                    else:
                        next_rr = True
                        result[moment] = rr
        except StopIteration:
            raise ValueError("Not enough relaxation rates are specified. You can either specify one relaxation rate, "
                             "which is used as the shear relaxation rate. In this case, conserved moments are "
                             "relaxed with 0, and higher-order moments are relaxed with 1. Another "
                             "possibility is to specify a relaxation rate for shear, bulk and one for each order "
                             "moment. In this case, conserved moments are also "
                             "relaxed with 0. The last possibility is to specify a relaxation rate for each moment, "
                             "including conserved moments")
    return result


def check_and_set_mrt_space(default, **kwargs):
    kwargs.setdefault('collision_space_info', CollisionSpaceInfo(default))

    if kwargs['collision_space_info'].collision_space not in (CollisionSpace.RAW_MOMENTS, CollisionSpace.POPULATIONS):
        raise ValueError("Raw moment-based methods can only be derived in population or raw moment space.")

# ----------------------------------------- Comparison view for notebooks ----------------------------------------------


def compare_moment_based_lb_methods(reference, other, show_deviations_only=False):
    import ipy_table
    table = []
    caption_rows = [len(table)]
    table.append(['Shared Moment', 'ref', 'other', 'difference'])

    reference_moments = set(reference.moments)
    other_moments = set(other.moments)
    for moment in reference_moments.intersection(other_moments):
        reference_value = reference.relaxation_info_dict[moment].equilibrium_value
        other_value = other.relaxation_info_dict[moment].equilibrium_value
        diff = sp.simplify(reference_value - other_value)
        if show_deviations_only and diff == 0:
            pass
        else:
            table.append([f"${sp.latex(moment)}$",
                          f"${sp.latex(reference_value)}$",
                          f"${sp.latex(other_value)}$",
                          f"${sp.latex(diff)}$"])

    only_in_ref = reference_moments - other_moments
    if only_in_ref:
        caption_rows.append(len(table))
        table.append(['Only in Ref', 'value', '', ''])
        for moment in only_in_ref:
            val = reference.relaxation_info_dict[moment].equilibrium_value
            table.append([f"${sp.latex(moment)}$",
                          f"${sp.latex(val)}$",
                          " ", " "])

    only_in_other = other_moments - reference_moments
    if only_in_other:
        caption_rows.append(len(table))
        table.append(['Only in Other', '', 'value', ''])
        for moment in only_in_other:
            val = other.relaxation_info_dict[moment].equilibrium_value
            table.append([f"${sp.latex(moment)}$",
                          " ",
                          f"${sp.latex(val)}$",
                          " "])

    table_display = ipy_table.make_table(table)
    for row_idx in caption_rows:
        for col in range(4):
            ipy_table.set_cell_style(row_idx, col, color='#bbbbbb')
    return table_display

# ----------------------------------------- Deprecation of Maxwellian Moments -----------------------------------------


def _deprecate_maxwellian_moments(continuous_equilibrium, kwargs):
    if 'maxwellian_moments' in kwargs:
        warn("Argument 'maxwellian_moments' is deprecated and will be removed by version 0.5."
             "Use `continuous_equilibrium` instead.",
             stacklevel=2)
        return kwargs['maxwellian_moments']
    else:
        return continuous_equilibrium
