import itertools
import operator
from collections import OrderedDict
from functools import reduce
from warnings import warn

import sympy as sp

from lbmpy.maxwellian_equilibrium import (
    compressible_to_incompressible_moment_value, get_cumulants_of_continuous_maxwellian_equilibrium,
    get_cumulants_of_discrete_maxwellian_equilibrium,
    get_moments_of_continuous_maxwellian_equilibrium,
    get_moments_of_discrete_maxwellian_equilibrium, get_weights)
from lbmpy.methods.abstractlbmethod import RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod
from lbmpy.methods.momentbased import MomentBasedLbMethod
from lbmpy.moments import (
    MOMENT_SYMBOLS, discrete_moment, exponents_to_polynomial_representations,
    get_default_moment_set_for_stencil, get_order, gram_schmidt, is_even, moments_of_order,
    moments_up_to_component_order, sort_moments_into_groups_of_same_order, is_bulk_moment, is_shear_moment)
from lbmpy.relaxationrates import relaxation_rate_from_magic_number
from lbmpy.stencils import get_stencil
from pystencils.stencil import have_same_entries
from pystencils.sympyextensions import common_denominator


def create_with_discrete_maxwellian_eq_moments(stencil, moment_to_relaxation_rate_dict, compressible=False,
                                               force_model=None, equilibrium_order=2,
                                               cumulant=False, c_s_sq=sp.Rational(1, 3)):
    r"""Creates a moment-based LBM by taking a list of moments with corresponding relaxation rate.

    These moments are relaxed against the moments of the discrete Maxwellian distribution.

    Args:
        stencil: nested tuple defining the discrete velocity space. See `get_stencil`
        moment_to_relaxation_rate_dict: dict that has as many entries as the stencil. Each moment, which can be
                                        represented by an exponent tuple or in polynomial form
                                        (see `lbmpy.moments`), is mapped to a relaxation rate.
        compressible: incompressible LBM methods split the density into :math:`\rho = \rho_0 + \Delta \rho`
                      where :math:`\rho_0` is chosen as one, and the first moment of the pdfs is :math:`\Delta \rho` .
                      This approximates the incompressible Navier-Stokes equations better than the standard
                      compressible model.
        force_model: force model instance, or None if no external forces
        equilibrium_order: approximation order of macroscopic velocity :math:`\mathbf{u}` in the equilibrium
        cumulant: if True relax cumulants instead of moments
        c_s_sq: Speed of sound squared

    Returns:
        `lbmpy.methods.MomentBasedLbMethod` instance
    """
    if isinstance(stencil, str):
        stencil = get_stencil(stencil)
    mom_to_rr_dict = OrderedDict(moment_to_relaxation_rate_dict)
    assert len(mom_to_rr_dict) == len(stencil), \
        "The number of moments has to be the same as the number of stencil entries"

    density_velocity_computation = DensityVelocityComputation(stencil, compressible, force_model)

    if cumulant:
        warn("Cumulant methods should be created with maxwellian_moments=True")
        eq_values = get_cumulants_of_discrete_maxwellian_equilibrium(stencil, tuple(mom_to_rr_dict.keys()),
                                                                     c_s_sq=c_s_sq, compressible=compressible,
                                                                     order=equilibrium_order)
    else:
        eq_values = get_moments_of_discrete_maxwellian_equilibrium(stencil, tuple(mom_to_rr_dict.keys()),
                                                                   c_s_sq=c_s_sq, compressible=compressible,
                                                                   order=equilibrium_order)

    rr_dict = OrderedDict([(mom, RelaxationInfo(eq_mom, rr))
                           for mom, rr, eq_mom in zip(mom_to_rr_dict.keys(), mom_to_rr_dict.values(), eq_values)])
    if cumulant:
        return CumulantBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model)
    else:
        return MomentBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model)


def create_with_continuous_maxwellian_eq_moments(stencil, moment_to_relaxation_rate_dict, compressible=False,
                                                 force_model=None, equilibrium_order=2,
                                                 cumulant=False, c_s_sq=sp.Rational(1, 3)):
    r"""
    Creates a moment-based LBM by taking a list of moments with corresponding relaxation rate. These moments are
    relaxed against the moments of the continuous Maxwellian distribution.
    For parameter description see :func:`lbmpy.methods.create_with_discrete_maxwellian_eq_moments`.
    By using the continuous Maxwellian we automatically get a compressible model.
    """
    if isinstance(stencil, str):
        stencil = get_stencil(stencil)
    mom_to_rr_dict = OrderedDict(moment_to_relaxation_rate_dict)
    assert len(mom_to_rr_dict) == len(stencil), "The number of moments has to be equal to the number of stencil entries"
    dim = len(stencil[0])
    density_velocity_computation = DensityVelocityComputation(stencil, compressible, force_model)

    if cumulant:
        eq_values = get_cumulants_of_continuous_maxwellian_equilibrium(tuple(mom_to_rr_dict.keys()), dim, c_s_sq=c_s_sq,
                                                                       order=equilibrium_order)
    else:
        eq_values = get_moments_of_continuous_maxwellian_equilibrium(tuple(mom_to_rr_dict.keys()), dim, c_s_sq=c_s_sq,
                                                                     order=equilibrium_order)

    if not compressible:
        if not compressible and cumulant:
            raise NotImplementedError("Incompressible cumulants not yet supported")
        rho = density_velocity_computation.defined_symbols(order=0)[1]
        u = density_velocity_computation.defined_symbols(order=1)[1]
        eq_values = [compressible_to_incompressible_moment_value(em, rho, u) for em in eq_values]

    rr_dict = OrderedDict([(mom, RelaxationInfo(eq_mom, rr))
                          for mom, rr, eq_mom in zip(mom_to_rr_dict.keys(), mom_to_rr_dict.values(), eq_values)])
    if cumulant:
        return CumulantBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model)
    else:
        return MomentBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model)


def create_generic_mrt(stencil, moment_eq_value_relaxation_rate_tuples, compressible=False,
                       force_model=None, cumulant=False):
    r"""
    Creates a generic moment-based LB method.

    Args:
        stencil: sequence of lattice velocities
        moment_eq_value_relaxation_rate_tuples: sequence of tuples containing (moment, equilibrium value, relax. rate)
        compressible: compressibility, determines calculation of velocity for force models
        force_model: see create_with_discrete_maxwellian_eq_moments
        cumulant: true for cumulant methods, False for moment-based methods
    """
    density_velocity_computation = DensityVelocityComputation(stencil, compressible, force_model)

    rr_dict = OrderedDict()
    for moment, eq_value, rr in moment_eq_value_relaxation_rate_tuples:
        moment = sp.sympify(moment)
        rr_dict[moment] = RelaxationInfo(eq_value, rr)
    if cumulant:
        return CumulantBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model)
    else:
        return MomentBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model)


def create_from_equilibrium(stencil, equilibrium, moment_to_relaxation_rate_dict, compressible=False, force_model=None):
    r"""
    Creates a moment-based LB method using a given equilibrium distribution function

    Args:
        stencil: see create_with_discrete_maxwellian_eq_moments
        equilibrium: list of equilibrium terms, dependent on rho and u, one for each stencil direction
        moment_to_relaxation_rate_dict: relaxation rate for each moment, or a symbol/float if all should relaxed with
                                        the same rate
        compressible: see create_with_discrete_maxwellian_eq_moments
        force_model: see create_with_discrete_maxwellian_eq_moments
    """
    if isinstance(stencil, str):
        stencil = get_stencil(stencil)
    if any(isinstance(moment_to_relaxation_rate_dict, t) for t in (sp.Symbol, float, int)):
        moment_to_relaxation_rate_dict = {m: moment_to_relaxation_rate_dict
                                          for m in get_default_moment_set_for_stencil(stencil)}

    mom_to_rr_dict = OrderedDict(moment_to_relaxation_rate_dict)
    assert len(mom_to_rr_dict) == len(stencil), "The number of moments has to be equal to the number of stencil entries"
    density_velocity_computation = DensityVelocityComputation(stencil, compressible, force_model)

    rr_dict = OrderedDict([(mom, RelaxationInfo(discrete_moment(equilibrium, mom, stencil).expand(), rr))
                          for mom, rr in zip(mom_to_rr_dict.keys(), mom_to_rr_dict.values())])
    return MomentBasedLbMethod(stencil, rr_dict, density_velocity_computation, force_model)


# ------------------------------------ SRT / TRT/ MRT Creators ---------------------------------------------------------


def create_srt(stencil, relaxation_rate, maxwellian_moments=False, **kwargs):
    r"""Creates a single relaxation time (SRT) lattice Boltzmann model also known as BGK model.

    Args:
        stencil: nested tuple defining the discrete velocity space. See :func:`lbmpy.stencils.get_stencil`
        relaxation_rate: relaxation rate (inverse of the relaxation time)
                        usually called :math:`\omega` in LBM literature
        maxwellian_moments: determines if the discrete or continuous maxwellian equilibrium is
                        used to compute the equilibrium moments

    Returns:
        :class:`lbmpy.methods.MomentBasedLbMethod` instance
    """
    if isinstance(stencil, str):
        stencil = get_stencil(stencil)
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
    if isinstance(stencil, str):
        stencil = get_stencil(stencil)
    moments = get_default_moment_set_for_stencil(stencil)
    rr_dict = OrderedDict([(m, relaxation_rate_even_moments if is_even(m) else relaxation_rate_odd_moments)
                           for m in moments])
    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, rr_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, rr_dict, **kwargs)


def create_trt_with_magic_number(stencil, relaxation_rate, magic_number=sp.Rational(3, 16), **kwargs):
    """
    Creates a two relaxation time (TRT) lattice Boltzmann method, where the relaxation time for odd moments is
    determines from the even moment relaxation time and a "magic number".
    For possible parameters see :func:`lbmpy.methods.create_trt`
    """
    rr_odd = relaxation_rate_from_magic_number(relaxation_rate, magic_number)
    return create_trt(stencil, relaxation_rate_even_moments=relaxation_rate,
                      relaxation_rate_odd_moments=rr_odd, **kwargs)


def create_mrt_raw(stencil, relaxation_rates, maxwellian_moments=False, **kwargs):
    """Creates a MRT method using non-orthogonalized moments."""
    if isinstance(stencil, str):
        stencil = get_stencil(stencil)
    moments = get_default_moment_set_for_stencil(stencil)
    rr_dict = OrderedDict(zip(moments, relaxation_rates))
    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, rr_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, rr_dict, **kwargs)


def create_mrt3(stencil, relaxation_rates, maxwellian_moments=False, **kwargs):
    """Creates a MRT with three relaxation times.

    The first rate controls viscosity, the second the bulk viscosity and the last is used to relax higher order moments.
    """
    warn("create_mrt3 is deprecated. It uses non-orthogonal moments, use create_mrt instead", DeprecationWarning)

    def product(iterable):
        return reduce(operator.mul, iterable, 1)

    if isinstance(stencil, str):
        stencil = get_stencil(stencil)

    dim = len(stencil[0])
    the_moment = MOMENT_SYMBOLS[:dim]

    shear_tensor_off_diagonal = [product(t) for t in itertools.combinations(the_moment, 2)]
    shear_tensor_diagonal = [m_i * m_i for m_i in the_moment]
    shear_tensor_trace = sum(shear_tensor_diagonal)
    shear_tensor_trace_free_diagonal = [dim * d - shear_tensor_trace for d in shear_tensor_diagonal]

    rest = [default_moment for default_moment in get_default_moment_set_for_stencil(stencil) 
            if get_order(default_moment) != 2]

    d = shear_tensor_off_diagonal + shear_tensor_trace_free_diagonal[:-1]
    t = [shear_tensor_trace]
    q = rest

    if 'magic_number' in kwargs:
        magic_number = kwargs['magic_number']
    else:
        magic_number = sp.Rational(3, 16)

    if len(relaxation_rates) == 1:
        relaxation_rates = [relaxation_rates[0],
                            relaxation_rate_from_magic_number(relaxation_rates[0], magic_number=magic_number),
                            1]
    elif len(relaxation_rates) == 2:
        relaxation_rates = [relaxation_rates[0],
                            relaxation_rate_from_magic_number(relaxation_rates[0], magic_number=magic_number),
                            relaxation_rates[1]]

    relaxation_rates = [relaxation_rates[0]] * len(d) + \
                       [relaxation_rates[1]] * len(t) + \
                       [relaxation_rates[2]] * len(q)

    all_moments = d + t + q
    moment_to_rr = OrderedDict((m, rr) for m, rr in zip(all_moments, relaxation_rates))

    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, moment_to_rr, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, moment_to_rr, **kwargs)


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
                        used to compute the equilibrium moments
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
    assert len(rest) + len(explicitly_defined) == 3**dim

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

    stencil = get_stencil("D2Q9") if dim == 2 else get_stencil("D3Q27")
    all_moments = rho + velocity + shear_part + rest_part
    moment_to_rr = OrderedDict((m, rr) for m, rr in zip(all_moments, relaxation_rates))

    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, moment_to_rr, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, moment_to_rr, **kwargs)


def create_mrt_orthogonal(stencil, relaxation_rate_getter, maxwellian_moments=False, weighted=None,
                          nested_moments=None, **kwargs):
    r"""
    Returns an orthogonal multi-relaxation time model for the stencils D2Q9, D3Q15, D3Q19 and D3Q27.
    These MRT methods are just one specific version - there are many MRT methods possible for all these stencils
    which differ by the linear combination of moments and the grouping into equal relaxation times.
    To create a generic MRT method use `create_with_discrete_maxwellian_eq_moments`

    Args:
        stencil: nested tuple defining the discrete velocity space. See `get_stencil`
        relaxation_rate_getter: function getting a list of moments as argument, returning the associated relaxation
        maxwellian_moments: determines if the discrete or continuous maxwellian equilibrium is
                                               used to compute the equilibrium moments
        weighted: whether to use weighted or unweighted orthogonality
        nested_moments: a list of lists of modes, grouped by common relaxation times. This is usually used in
                        conjunction with `mrt_orthogonal_modes_literature`. If this argument is not provided,
                        Gram-Schmidt orthogonalization of the default modes is performed.
    """
    dim = len(stencil[0])
    if isinstance(stencil, str):
        stencil = get_stencil(stencil)

    if weighted is None and not nested_moments:
        raise ValueError("Please specify whether you want weighted or unweighted orthogonality using 'weighted='")
    elif weighted:
        weights = get_weights(stencil, sp.Rational(1, 3))
    else:
        weights = None

    moment_to_relaxation_rate_dict = OrderedDict()
    if not nested_moments:
        moments = get_default_moment_set_for_stencil(stencil)
        x, y, z = MOMENT_SYMBOLS
        if dim == 2:
            diagonal_viscous_moments = [x ** 2 + y ** 2, x ** 2]
        else:
            diagonal_viscous_moments = [x ** 2 + y ** 2 + z ** 2, x ** 2, y ** 2 - z ** 2]
        for i, d in enumerate(MOMENT_SYMBOLS[:dim]):
            moments[moments.index(d**2)] = diagonal_viscous_moments[i]
        orthogonal_moments = gram_schmidt(moments, stencil, weights)
        orthogonal_moments_scaled = [e * common_denominator(e) for e in orthogonal_moments]
        nested_moments = list(sort_moments_into_groups_of_same_order(orthogonal_moments_scaled).values())
        # second order moments: separate bulk from shear
        second_order_moments = nested_moments[2]
        bulk_moment = [m for m in second_order_moments if is_bulk_moment(m, dim)]
        shear_moments = [m for m in second_order_moments if is_shear_moment(m, dim)]
        assert len(shear_moments) + len(bulk_moment) == len(second_order_moments)
        nested_moments[2] = shear_moments
        nested_moments.insert(3, bulk_moment)
    for moment_list in nested_moments:
        rr = relaxation_rate_getter(moment_list)
        for m in moment_list:
            moment_to_relaxation_rate_dict[m] = rr

    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, moment_to_relaxation_rate_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, moment_to_relaxation_rate_dict, **kwargs)


def mrt_orthogonal_modes_literature(stencil, is_weighted, is_cumulant):
    """
    Returns a list of lists of modes, grouped by common relaxation times.
    This is for commonly used MRT models found in literature.

    Args:
        stencil: nested tuple defining the discrete velocity space. See `get_stencil`
        is_weighted: whether to use weighted or unweighted orthogonality
        is_cumulant: whether a moment-based or cumulant-based model is desired

    MRT schemes as described in the following references are used
    """
    x, y, z = MOMENT_SYMBOLS
    one = sp.Rational(1, 1)

    if have_same_entries(stencil, get_stencil("D2Q9")) and is_weighted:
        # Reference:
        # Duenweg, B., Schiller, U. D., & Ladd, A. J. (2007). Statistical mechanics of the fluctuating
        # lattice Boltzmann equation. Physical Review E, 76(3)
        sq = x ** 2 + y ** 2
        all_moments = [one, x, y, 3 * sq - 2, 2 * x ** 2 - sq, x * y,
                       (3 * sq - 4) * x, (3 * sq - 4) * y, 9 * sq ** 2 - 15 * sq + 2]
        nested_moments = list(sort_moments_into_groups_of_same_order(all_moments).values())
        return nested_moments
    elif have_same_entries(stencil, get_stencil("D3Q15")) and is_weighted:
        sq = x ** 2 + y ** 2 + z ** 2
        nested_moments = [
            [one, x, y, z],  # [0, 3, 5, 7]
            [sq - 1],  # [1]
            [3 * sq ** 2 - 9 * sq + 4],  # [2]
            [(3 * sq - 5) * x, (3 * sq - 5) * y, (3 * sq - 5) * z],  # [4, 6, 8]
            [3 * x ** 2 - sq, y ** 2 - z ** 2, x * y, y * z, x * z],  # [9, 10, 11, 12, 13]
            [x * y * z]
        ]
    elif have_same_entries(stencil, get_stencil("D3Q19")) and is_weighted:
        # This MRT variant mentioned in the dissertation of Ulf Schiller 
        # "Thermal fluctuations and boundary conditions in the lattice Boltzmann method" (2008), p. 24ff
        # There are some typos in the moment matrix on p.27
        # The here implemented ordering of the moments is however different from that reference (Eq. 2.61-2.63)
        # The moments are weighted-orthogonal (Eq. 2.58)
        
        # Further references:
        # Duenweg, B., Schiller, U. D., & Ladd, A. J. (2007). Statistical mechanics of the fluctuating
        # lattice Boltzmann equation. Physical Review E, 76(3)
        # Chun, B., & Ladd, A. J. (2007). Interpolated boundary condition for lattice Boltzmann simulations of
        # flows in narrow gaps. Physical review E, 75(6)
        if is_cumulant:
            nested_moments = [
                [sp.sympify(1), x, y, z],  # conserved
                [x * y,
                 x * z,
                 y * z,
                 x ** 2 - y ** 2,
                 x ** 2 - z ** 2],  # shear

                [x ** 2 + y ** 2 + z ** 2],  # bulk

                [x * y ** 2 + x * z ** 2,
                 x ** 2 * y + y * z ** 2,
                 x ** 2 * z + y ** 2 * z],
                [x * y ** 2 - x * z ** 2,
                 x ** 2 * y - y * z ** 2,
                 x ** 2 * z - y ** 2 * z],

                [x ** 2 * y ** 2 - 2 * x ** 2 * z ** 2 + y ** 2 * z ** 2,
                 x ** 2 * y ** 2 + x ** 2 * z ** 2 - 2 * y ** 2 * z ** 2],
                [x ** 2 * y ** 2 + x ** 2 * z ** 2 + y ** 2 * z ** 2],
            ]
        else:
            sq = x ** 2 + y ** 2 + z ** 2
            nested_moments = [
                [one, x, y, z],  # [0, 3, 5, 7]
                [sq - 1],  # [1]
                [3 * sq ** 2 - 6 * sq + 1],  # [2]
                [(3 * sq - 5) * x, (3 * sq - 5) * y, (3 * sq - 5) * z],  # [4, 6, 8]
                [3 * x ** 2 - sq, y ** 2 - z ** 2, x * y, y * z, x * z],  # [9, 11, 13, 14, 15]
                [(2 * sq - 3) * (3 * x ** 2 - sq), (2 * sq - 3) * (y ** 2 - z ** 2)],  # [10, 12]
                [(y ** 2 - z ** 2) * x, (z ** 2 - x ** 2) * y, (x ** 2 - y ** 2) * z]  # [16, 17, 18]
            ]
    elif have_same_entries(stencil, get_stencil("D3Q27")) and not is_weighted:
        if is_cumulant:
            nested_moments = [
                [sp.sympify(1), x, y, z],  # conserved
                [x * y,
                 x * z,
                 y * z,
                 x ** 2 - y ** 2,
                 x ** 2 - z ** 2],  # shear

                [x ** 2 + y ** 2 + z ** 2],  # bulk

                [x * y ** 2 + x * z ** 2,
                 x ** 2 * y + y * z ** 2,
                 x ** 2 * z + y ** 2 * z],
                [x * y ** 2 - x * z ** 2,
                 x ** 2 * y - y * z ** 2,
                 x ** 2 * z - y ** 2 * z],
                [x * y * z],

                [x ** 2 * y ** 2 - 2 * x ** 2 * z ** 2 + y ** 2 * z ** 2,
                 x ** 2 * y ** 2 + x ** 2 * z ** 2 - 2 * y ** 2 * z ** 2],
                [x ** 2 * y ** 2 + x ** 2 * z ** 2 + y ** 2 * z ** 2],

                [x ** 2 * y * z,
                 x * y ** 2 * z,
                 x * y * z ** 2],

                [x ** 2 * y ** 2 * z,
                 x ** 2 * y * z ** 2,
                 x * y ** 2 * z ** 2],

                [x ** 2 * y ** 2 * z ** 2],
            ]
        else:
            xsq, ysq, zsq = x ** 2, y ** 2, z ** 2
            all_moments = [
                sp.Rational(1, 1),  # 0
                x, y, z,  # 1, 2, 3
                x * y, x * z, y * z,  # 4, 5, 6
                xsq - ysq,  # 7
                (xsq + ysq + zsq) - 3 * zsq,  # 8
                (xsq + ysq + zsq) - 2,  # 9
                3 * (x * ysq + x * zsq) - 4 * x,  # 10
                3 * (xsq * y + y * zsq) - 4 * y,  # 11
                3 * (xsq * z + ysq * z) - 4 * z,  # 12
                x * ysq - x * zsq,  # 13
                xsq * y - y * zsq,  # 14
                xsq * z - ysq * z,  # 15
                x * y * z,  # 16
                3 * (xsq * ysq + xsq * zsq + ysq * zsq) - 4 * (xsq + ysq + zsq) + 4,  # 17
                3 * (xsq * ysq + xsq * zsq - 2 * ysq * zsq) - 2 * (2 * xsq - ysq - zsq),  # 18
                3 * (xsq * ysq - xsq * zsq) - 2 * (ysq - zsq),  # 19
                3 * (xsq * y * z) - 2 * (y * z),  # 20
                3 * (x * ysq * z) - 2 * (x * z),  # 21
                3 * (x * y * zsq) - 2 * (x * y),  # 22
                9 * (x * ysq * zsq) - 6 * (x * ysq + x * zsq) + 4 * x,  # 23
                9 * (xsq * y * zsq) - 6 * (xsq * y + y * zsq) + 4 * y,  # 24
                9 * (xsq * ysq * z) - 6 * (xsq * z + ysq * z) + 4 * z,  # 25
                27 * (xsq * ysq * zsq) - 18 * (xsq * ysq + xsq * zsq + ysq * zsq) + 12 * (xsq + ysq + zsq) - 8,  # 26
            ]
            nested_moments = list(sort_moments_into_groups_of_same_order(all_moments).values())
    else:
        raise NotImplementedError("No MRT model is available (yet) for this stencil. "
                                  "Create a custom MRT using 'create_with_discrete_maxwellian_eq_moments'")

    return nested_moments


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
            table.append(["$%s$" % (sp.latex(moment), ),
                          "$%s$" % (sp.latex(reference_value), ),
                          "$%s$" % (sp.latex(other_value), ),
                          "$%s$" % (sp.latex(diff),)])

    only_in_ref = reference_moments - other_moments
    if only_in_ref:
        caption_rows.append(len(table))
        table.append(['Only in Ref', 'value', '', ''])
        for moment in only_in_ref:
            val = reference.relaxation_info_dict[moment].equilibrium_value
            table.append(["$%s$" % (sp.latex(moment),),
                          "$%s$" % (sp.latex(val),),
                          " ", " "])

    only_in_other = other_moments - reference_moments
    if only_in_other:
        caption_rows.append(len(table))
        table.append(['Only in Other', '', 'value', ''])
        for moment in only_in_other:
            val = other.relaxation_info_dict[moment].equilibrium_value
            table.append(["$%s$" % (sp.latex(moment),),
                          " ",
                          "$%s$" % (sp.latex(val),),
                          " "])

    table_display = ipy_table.make_table(table)
    for row_idx in caption_rows:
        for col in range(4):
            ipy_table.set_cell_style(row_idx, col, color='#bbbbbb')
    return table_display
