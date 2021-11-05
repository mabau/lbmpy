import itertools
import operator
from collections import OrderedDict
from functools import reduce

import sympy as sp

from lbmpy.maxwellian_equilibrium import (
    compressible_to_incompressible_moment_value, get_equilibrium_values_of_maxwell_boltzmann_function,
    get_moments_of_discrete_maxwellian_equilibrium, get_weights)

from lbmpy.methods.abstractlbmethod import RelaxationInfo
from lbmpy.methods.default_moment_sets import cascaded_moment_sets_literature

from lbmpy.methods.centeredcumulant import CenteredCumulantBasedLbMethod
from lbmpy.methods.centeredcumulant.cumulant_transform import CentralMomentsToCumulantsByGeneratingFunc

from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation

from lbmpy.methods.momentbased.momentbasedmethod import MomentBasedLbMethod
from lbmpy.methods.momentbased.centralmomentbasedmethod import CentralMomentBasedLbMethod
from lbmpy.moment_transforms import PdfsToCentralMomentsByShiftMatrix, PdfsToMomentsByChimeraTransform

from lbmpy.moments import (
    MOMENT_SYMBOLS, discrete_moment, exponents_to_polynomial_representations,
    get_default_moment_set_for_stencil, gram_schmidt, is_even, moments_of_order,
    moments_up_to_component_order, sort_moments_into_groups_of_same_order,
    is_bulk_moment, is_shear_moment, get_order, set_up_shift_matrix)

from lbmpy.relaxationrates import relaxation_rate_from_magic_number
from lbmpy.enums import Stencil
from lbmpy.stencils import LBStencil
from pystencils.sympyextensions import common_denominator


def create_with_discrete_maxwellian_eq_moments(stencil, moment_to_relaxation_rate_dict, compressible=False,
                                               force_model=None, equilibrium_order=2, c_s_sq=sp.Rational(1, 3),
                                               central_moment_space=False,
                                               moment_transform_class=None,
                                               central_moment_transform_class=PdfsToCentralMomentsByShiftMatrix):
    r"""Creates a moment-based LBM by taking a list of moments with corresponding relaxation rate.

    These moments are relaxed against the moments of the discrete Maxwellian distribution.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStenil`
        moment_to_relaxation_rate_dict: dict that has as many entries as the stencil. Each moment, which can be
                                        represented by an exponent tuple or in polynomial form
                                        (see `lbmpy.moments`), is mapped to a relaxation rate.
        compressible: incompressible LBM methods split the density into :math:`\rho = \rho_0 + \Delta \rho`
                      where :math:`\rho_0` is chosen as one, and the first moment of the pdfs is :math:`\Delta \rho` .
                      This approximates the incompressible Navier-Stokes equations better than the standard
                      compressible model.
        force_model: force model instance, or None if no external forces
        equilibrium_order: approximation order of macroscopic velocity :math:`\mathbf{u}` in the equilibrium
        c_s_sq: Speed of sound squared
        central_moment_space: If set to True, an instance of 
                              :class:`lbmpy.methods.momentbased.CentralMomentBasedLbMethod` is returned, 
                              and the the collision will be performed in the central moment space.
        moment_transform_class: Class implementing the transform from populations to moment space.
        central_moment_transform_class: Class implementing the transform from populations to central moment space.

    Returns:
        Instance of either :class:`lbmpy.methods.momentbased.MomentBasedLbMethod` or 
        :class:`lbmpy.methods.momentbased.CentralMomentBasedLbMethod` 
    """
    mom_to_rr_dict = OrderedDict(moment_to_relaxation_rate_dict)
    assert len(mom_to_rr_dict) == stencil.Q, \
        "The number of moments has to be the same as the number of stencil entries"

    density_velocity_computation = DensityVelocityComputation(stencil, compressible, force_model)

    moments = tuple(mom_to_rr_dict.keys())
    eq_values = get_moments_of_discrete_maxwellian_equilibrium(stencil, moments,
                                                               c_s_sq=c_s_sq, compressible=compressible,
                                                               order=equilibrium_order)
    if central_moment_space:
        N = set_up_shift_matrix(moments, stencil)
        eq_values = sp.simplify(N * sp.Matrix(eq_values))

    rr_dict = OrderedDict([(mom, RelaxationInfo(eq_mom, rr))
                           for mom, rr, eq_mom in zip(mom_to_rr_dict.keys(), mom_to_rr_dict.values(), eq_values)])

    if central_moment_space:
        return CentralMomentBasedLbMethod(stencil, rr_dict, density_velocity_computation,
                                          force_model, central_moment_transform_class)
    else:
        return MomentBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model, moment_transform_class)


def create_with_continuous_maxwellian_eq_moments(stencil, moment_to_relaxation_rate_dict, compressible=False,
                                                 force_model=None, equilibrium_order=2, c_s_sq=sp.Rational(1, 3),
                                                 central_moment_space=False,
                                                 moment_transform_class=None,
                                                 central_moment_transform_class=PdfsToCentralMomentsByShiftMatrix):
    r"""
    Creates a moment-based LBM by taking a list of moments with corresponding relaxation rate. These moments are
    relaxed against the moments of the continuous Maxwellian distribution.
    For parameter description see :func:`lbmpy.methods.create_with_discrete_maxwellian_eq_moments`.
    By using the continuous Maxwellian we automatically get a compressible model.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        moment_to_relaxation_rate_dict: dict that has as many entries as the stencil. Each moment, which can be
                                        represented by an exponent tuple or in polynomial form
                                        (see `lbmpy.moments`), is mapped to a relaxation rate.
        compressible: incompressible LBM methods split the density into :math:`\rho = \rho_0 + \Delta \rho`
                      where :math:`\rho_0` is chosen as one, and the first moment of the pdfs is :math:`\Delta \rho` .
                      This approximates the incompressible Navier-Stokes equations better than the standard
                      compressible model.
        force_model: force model instance, or None if no external forces
        equilibrium_order: approximation order of macroscopic velocity :math:`\mathbf{u}` in the equilibrium
        c_s_sq: Speed of sound squared
        central_moment_space: If set to True, an instance of 
                              :class:`lbmpy.methods.momentbased.CentralMomentBasedLbMethod` is returned, 
                              and the the collision will be performed in the central moment space.
        moment_transform_class: Class implementing the transform from populations to moment space.
        central_moment_transform_class: Class implementing the transform from populations to central moment space.

    Returns:
        Instance of either :class:`lbmpy.methods.momentbased.MomentBasedLbMethod` or 
        :class:`lbmpy.methods.momentbased.CentralMomentBasedLbMethod` 
    """
    mom_to_rr_dict = OrderedDict(moment_to_relaxation_rate_dict)
    assert len(mom_to_rr_dict) == stencil.Q, "The number of moments has to be equal to the number of stencil entries"
    dim = stencil.D
    density_velocity_computation = DensityVelocityComputation(stencil, compressible, force_model)
    moments = tuple(mom_to_rr_dict.keys())

    if compressible and central_moment_space:
        eq_values = get_equilibrium_values_of_maxwell_boltzmann_function(moments, dim, c_s_sq=c_s_sq,
                                                                         order=equilibrium_order,
                                                                         space="central moment")
    else:
        eq_values = get_equilibrium_values_of_maxwell_boltzmann_function(moments, dim, c_s_sq=c_s_sq,
                                                                         order=equilibrium_order, space="moment")

    if not compressible:
        rho = density_velocity_computation.defined_symbols(order=0)[1]
        u = density_velocity_computation.defined_symbols(order=1)[1]
        eq_values = [compressible_to_incompressible_moment_value(em, rho, u) for em in eq_values]
        if central_moment_space:
            N = set_up_shift_matrix(moments, stencil)
            eq_values = sp.simplify(N * sp.Matrix(eq_values))

    rr_dict = OrderedDict([(mom, RelaxationInfo(eq_mom, rr))
                           for mom, rr, eq_mom in zip(mom_to_rr_dict.keys(), mom_to_rr_dict.values(), eq_values)])

    if central_moment_space:
        return CentralMomentBasedLbMethod(stencil, rr_dict, density_velocity_computation,
                                          force_model, central_moment_transform_class)
    else:
        return MomentBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model, moment_transform_class)


def create_generic_mrt(stencil, moment_eq_value_relaxation_rate_tuples, compressible=False,
                       force_model=None, moment_transform_class=PdfsToMomentsByChimeraTransform):
    r"""
    Creates a generic moment-based LB method.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        moment_eq_value_relaxation_rate_tuples: sequence of tuples containing (moment, equilibrium value, relax. rate)
        compressible: compressibility, determines calculation of velocity for force models
        force_model: see create_with_discrete_maxwellian_eq_moments
        moment_transform_class: class to define the transformation to the moment space
    """
    density_velocity_computation = DensityVelocityComputation(stencil, compressible, force_model)

    rr_dict = OrderedDict()
    for moment, eq_value, rr in moment_eq_value_relaxation_rate_tuples:
        moment = sp.sympify(moment)
        rr_dict[moment] = RelaxationInfo(eq_value, rr)
    return MomentBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model, moment_transform_class)


def create_from_equilibrium(stencil, equilibrium, moment_to_relaxation_rate_dict,
                            compressible=False, force_model=None,
                            moment_transform_class=PdfsToMomentsByChimeraTransform):
    r"""
    Creates a moment-based LB method using a given equilibrium distribution function

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        equilibrium: list of equilibrium terms, dependent on rho and u, one for each stencil direction
        moment_to_relaxation_rate_dict: relaxation rate for each moment, or a symbol/float if all should relaxed with
                                        the same rate
        compressible: see create_with_discrete_maxwellian_eq_moments
        force_model: see create_with_discrete_maxwellian_eq_moments
        moment_transform_class: class to define the transformation to the moment space
    """
    if any(isinstance(moment_to_relaxation_rate_dict, t) for t in (sp.Symbol, float, int)):
        moment_to_relaxation_rate_dict = {m: moment_to_relaxation_rate_dict
                                          for m in get_default_moment_set_for_stencil(stencil)}

    mom_to_rr_dict = OrderedDict(moment_to_relaxation_rate_dict)
    assert len(mom_to_rr_dict) == stencil.Q, "The number of moments has to be equal to the number of stencil entries"
    density_velocity_computation = DensityVelocityComputation(stencil, compressible, force_model)

    rr_dict = OrderedDict([(mom, RelaxationInfo(discrete_moment(equilibrium, mom, stencil).expand(), rr))
                           for mom, rr in zip(mom_to_rr_dict.keys(), mom_to_rr_dict.values())])
    return MomentBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model, moment_transform_class)


# ------------------------------------ SRT / TRT/ MRT Creators ---------------------------------------------------------


def create_srt(stencil, relaxation_rate, maxwellian_moments=False, **kwargs):
    r"""Creates a single relaxation time (SRT) lattice Boltzmann model also known as BGK model.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rate: relaxation rate (inverse of the relaxation time)
                        usually called :math:`\omega` in LBM literature
        maxwellian_moments: determines if the discrete or continuous maxwellian equilibrium is
                        used to compute the equilibrium moments

    Returns:
        :class:`lbmpy.methods.momentbased.MomentBasedLbMethod` instance
    """
    moments = get_default_moment_set_for_stencil(stencil)
    rr_dict = OrderedDict([(m, relaxation_rate) for m in moments])
    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, rr_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, rr_dict, **kwargs)


def create_trt(stencil, relaxation_rate_even_moments, relaxation_rate_odd_moments,
               maxwellian_moments=False, **kwargs):
    """
    Creates a two relaxation time (TRT) lattice Boltzmann model, where even and odd moments are relaxed differently.
    In the SRT model the exact wall position of no-slip boundaries depends on the viscosity, the TRT method does not
    have this problem.

    Parameters are similar to :func:`lbmpy.methods.create_srt`, but instead of one relaxation rate there are
    two relaxation rates: one for even moments (determines viscosity) and one for odd moments.
    If unsure how to choose the odd relaxation rate, use the function :func:`lbmpy.methods.create_trt_with_magic_number`
    """
    moments = get_default_moment_set_for_stencil(stencil)
    rr_dict = OrderedDict([(m, relaxation_rate_even_moments if is_even(m) else relaxation_rate_odd_moments)
                           for m in moments])
    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, rr_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, rr_dict, **kwargs)


def create_trt_with_magic_number(stencil, relaxation_rate, magic_number=sp.Rational(3, 16), **kwargs):
    r"""
    Creates a two relaxation time (TRT) lattice Boltzmann method, where the relaxation time for odd moments is
    determines from the even moment relaxation time and a "magic number".
    For possible parameters see :func:`lbmpy.methods.create_trt`

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


def create_mrt_raw(stencil, relaxation_rates, maxwellian_moments=False, **kwargs):
    r"""
    Creates a MRT method using non-orthogonalized moments.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates (inverse of the relaxation times) for each moment
        maxwellian_moments: determines if the discrete or continuous maxwellian equilibrium is
                        used to compute the equilibrium moments.
    Returns:
        :class:`lbmpy.methods.momentbased.MomentBasedLbMethod` instance
    """
    moments = get_default_moment_set_for_stencil(stencil)
    nested_moments = [(c,) for c in moments]
    rr_dict = _get_relaxation_info_dict(relaxation_rates, nested_moments, stencil.D)
    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, rr_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, rr_dict, **kwargs)


def create_central_moment(stencil, relaxation_rates, nested_moments=None,
                          maxwellian_moments=False, **kwargs):
    r"""
    Creates moment based LB method where the collision takes place in the central moment space.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates (inverse of the relaxation times) for each moment
        nested_moments: a list of lists of modes, grouped by common relaxation times.
        maxwellian_moments: determines if the discrete or continuous maxwellian equilibrium is
                        used to compute the equilibrium moments.
    Returns:
        :class:`lbmpy.methods.momentbased.CentralMomentBasedLbMethod` instance
    """
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

    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, rr_dict, central_moment_space=True, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, rr_dict, central_moment_space=True, **kwargs)


def create_trt_kbc(dim, shear_relaxation_rate, higher_order_relaxation_rate, method_name='KBC-N4',
                   maxwellian_moments=False, **kwargs):
    """
    Creates a method with two relaxation rates, one for lower order moments which determines the viscosity and
    one for higher order moments. In entropic models this second relaxation rate is chosen subject to an entropy
    condition. Which moments are relaxed by which rate is determined by the method_name

    Args:
        dim: 2 or 3, leads to stencil D2Q9 or D3Q27
        shear_relaxation_rate: relaxation rate that determines viscosity
        higher_order_relaxation_rate: relaxation rate for higher order moments
        method_name: string 'KBC-Nx' where x can be an number from 1 to 4, for details see
                    "Karlin 2015: Entropic multi relaxation lattice Boltzmann models for turbulent flows"
        maxwellian_moments: determines if the discrete or continuous maxwellian equilibrium is
                        used to compute the equilibrium moments.
    """

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

    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, moment_to_rr, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, moment_to_rr, **kwargs)


def create_mrt_orthogonal(stencil, relaxation_rates, maxwellian_moments=False, weighted=None,
                          nested_moments=None, **kwargs):
    r"""
    Returns an orthogonal multi-relaxation time model for the stencils D2Q9, D3Q15, D3Q19 and D3Q27.
    These MRT methods are just one specific version - there are many MRT methods possible for all these stencils
    which differ by the linear combination of moments and the grouping into equal relaxation times.
    To create a generic MRT method use `create_with_discrete_maxwellian_eq_moments`

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates for the moments
        maxwellian_moments: determines if the discrete or continuous maxwellian equilibrium is
                                               used to compute the equilibrium moments
        weighted: whether to use weighted or unweighted orthogonality
        nested_moments: a list of lists of modes, grouped by common relaxation times. If this argument is not provided,
                        Gram-Schmidt orthogonalization of the default modes is performed. The default modes equal the
                        raw moments except for the separation of the shear and bulk viscosity.
    """
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

    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil,
                                                            moment_to_relaxation_rate_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil,
                                                          moment_to_relaxation_rate_dict, **kwargs)


# ----------------------------------------- Cumulant method creators ---------------------------------------------------


def create_centered_cumulant_model(stencil, cumulant_to_rr_dict, force_model=None,
                                   equilibrium_order=None, c_s_sq=sp.Rational(1, 3),
                                   galilean_correction=False,
                                   central_moment_transform_class=PdfsToCentralMomentsByShiftMatrix,
                                   cumulant_transform_class=CentralMomentsToCumulantsByGeneratingFunc):
    r"""Creates a cumulant lattice Boltzmann model.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        cumulant_to_rr_dict: dict that has as many entries as the stencil. Each cumulant, which can be
                             represented by an exponent tuple or in polynomial form is mapped to a relaxation rate.
                             See :func:`lbmpy.methods.default_moment_sets.cascaded_moment_sets_literature`
        force_model: force model used for the collision. For cumulant LB method a good choice is
                     `lbmpy.methods.centeredcumulant.CenteredCumulantForceModel`
        equilibrium_order: approximation order of macroscopic velocity :math:`\mathbf{u}` in the equilibrium
        c_s_sq: Speed of sound squared
        galilean_correction: special correction for D3Q27 cumulant collisions. See Appendix H in
                             :cite:`geier2015`. Implemented in :mod:`lbmpy.methods.centeredcumulant.galilean_correction`
        central_moment_transform_class: Class which defines the transformation to the central moment space
                                        (see :mod:`lbmpy.moment_transforms`)
        cumulant_transform_class: Class which defines the transformation from the central moment space to the
                                  cumulant space (see :mod:`lbmpy.methods.centeredcumulant.cumulant_transform`)

    Returns:
        :class:`lbmpy.methods.centeredcumulant.CenteredCumulantBasedLbMethod` instance
    """

    one = sp.Integer(1)

    assert len(cumulant_to_rr_dict) == stencil.Q, \
        "The number of moments has to be equal to the number of stencil entries"

    # CQC
    cqc = DensityVelocityComputation(stencil, True, force_model=force_model)
    density_symbol = cqc.zeroth_order_moment_symbol
    velocity_symbols = cqc.first_order_moment_symbols

    #   Equilibrium Values
    higher_order_polynomials = list(filter(lambda x: get_order(x) > 1, cumulant_to_rr_dict.keys()))

    #   Lower Order Equilibria
    cumulants_to_relaxation_info_dict = {one: RelaxationInfo(density_symbol, cumulant_to_rr_dict[one])}
    for d in MOMENT_SYMBOLS[:stencil.D]:
        cumulants_to_relaxation_info_dict[d] = RelaxationInfo(0, cumulant_to_rr_dict[d])

    #   Polynomial Cumulant Equilibria
    polynomial_equilibria = get_equilibrium_values_of_maxwell_boltzmann_function(
        higher_order_polynomials, stencil.D, rho=density_symbol, u=velocity_symbols,
        c_s_sq=c_s_sq, order=equilibrium_order, space="cumulant")
    polynomial_equilibria = [density_symbol * v for v in polynomial_equilibria]

    for i, c in enumerate(higher_order_polynomials):
        cumulants_to_relaxation_info_dict[c] = RelaxationInfo(polynomial_equilibria[i], cumulant_to_rr_dict[c])

    return CenteredCumulantBasedLbMethod(stencil, cumulants_to_relaxation_info_dict, cqc, force_model,
                                         galilean_correction=galilean_correction,
                                         central_moment_transform_class=central_moment_transform_class,
                                         cumulant_transform_class=cumulant_transform_class)


def create_with_polynomial_cumulants(stencil, relaxation_rates, cumulant_groups, **kwargs):
    r"""Creates a cumulant lattice Boltzmann model based on a default polynomial set.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates for each cumulant group. If None are provided a list of symbolic relaxation
                          rates is created and used. If only a list with one entry is provided this relaxation rate is
                          used for determine the viscosity of the simulation. All other cumulants are relaxed with one.
                          If a cumulant force model is provided the first order cumulants are relaxed with two to ensure
                          that the force is applied correctly to the momentum groups
        cumulant_groups: Nested sequence of polynomial expressions defining the cumulants to be relaxed. All cumulants 
                         within one group are relaxed with the same relaxation rate.
        kwargs: See :func:`create_centered_cumulant_model`

    Returns:
        :class:`lbmpy.methods.centeredcumulant.CenteredCumulantBasedLbMethod` instance
    """
    cumulant_to_rr_dict = _get_relaxation_info_dict(relaxation_rates, cumulant_groups, stencil.D)
    return create_centered_cumulant_model(stencil, cumulant_to_rr_dict, **kwargs)


def create_with_monomial_cumulants(stencil, relaxation_rates, **kwargs):
    r"""Creates a cumulant lattice Boltzmann model based on a default polinomial set.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`
        relaxation_rates: relaxation rates for each cumulant group. If None are provided a list of symbolic relaxation
                          rates is created and used. If only a list with one entry is provided this relaxation rate is
                          used for determine the viscosity of the simulation. All other cumulants are relaxed with one.
                          If a cumulant force model is provided the first order cumulants are relaxed with two to ensure
                          that the force is applied correctly to the momentum groups
        kwargs: See :func:`create_centered_cumulant_model`

    Returns:
        :class:`lbmpy.methods.centeredcumulant.CenteredCumulantBasedLbMethod` instance
    """
    # Get monomial moments
    cumulants = get_default_moment_set_for_stencil(stencil)
    cumulant_groups = [(c,) for c in cumulants]

    return create_with_polynomial_cumulants(stencil, relaxation_rates, cumulant_groups, **kwargs)


def create_with_default_polynomial_cumulants(stencil, relaxation_rates, **kwargs):
    r"""Creates a cumulant lattice Boltzmann model based on a default polynomial set.

    Args: See :func:`create_with_polynomial_cumulants`.

    Returns:
        :class:`lbmpy.methods.centeredcumulant.CenteredCumulantBasedLbMethod` instance
    """
    # Get polynomial groups
    cumulant_groups = cascaded_moment_sets_literature(stencil)
    return create_with_polynomial_cumulants(stencil, relaxation_rates, cumulant_groups, **kwargs)


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
