from warnings import warn

import sympy as sp
from collections import OrderedDict
from functools import reduce
import operator
import itertools
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod
from lbmpy.methods.momentbased import MomentBasedLbMethod
from lbmpy.stencils import stencils_have_same_entries, get_stencil
from lbmpy.moments import is_even, gram_schmidt, get_default_moment_set_for_stencil, MOMENT_SYMBOLS, \
    exponents_to_polynomial_representations, moments_of_order, moments_up_to_component_order, sort_moments_into_groups_of_same_order, \
    get_order, discrete_moment
from pystencils.sympyextensions import common_denominator
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation
from lbmpy.methods.abstractlbmethod import RelaxationInfo
from lbmpy.maxwellian_equilibrium import get_moments_of_discrete_maxwellian_equilibrium, \
    get_moments_of_continuous_maxwellian_equilibrium, get_cumulants_of_discrete_maxwellian_equilibrium, \
    get_cumulants_of_continuous_maxwellian_equilibrium, compressible_to_incompressible_moment_value
from lbmpy.relaxationrates import relaxation_rate_from_magic_number, default_relaxation_rate_names


def create_with_discrete_maxwellian_eq_moments(stencil, moment_to_relaxation_rate_dict, compressible=False,
                                               force_model=None, equilibrium_order=2,
                                               cumulant=False, c_s_sq=sp.Rational(1, 3)):
    r"""
    Creates a moment-based LBM by taking a list of moments with corresponding relaxation rate. These moments are
    relaxed against the moments of the discrete Maxwellian distribution.

    Args:
        stencil: nested tuple defining the discrete velocity space. See `func:lbmpy.stencils.get_stencil`
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
        :class:`lbmpy.methods.MomentBasedLbMethod` instance
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

    rr_dict = OrderedDict([(mom, RelaxationInfo(eqMom, rr))
                           for mom, rr, eqMom in zip(mom_to_rr_dict.keys(), mom_to_rr_dict.values(), eq_values)])
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

    rr_dict = OrderedDict([(mom, RelaxationInfo(eqMom, rr))
                          for mom, rr, eqMom in zip(mom_to_rr_dict.keys(), mom_to_rr_dict.values(), eq_values)])
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
    for moment, eqValue, rr in moment_eq_value_relaxation_rate_tuples:
        moment = sp.sympify(moment)
        rr_dict[moment] = RelaxationInfo(eqValue, rr)
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

    rest = [defaultMoment for defaultMoment in get_default_moment_set_for_stencil(stencil) if get_order(defaultMoment) != 2]

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

    :param dim: 2 or 3, leads to stencil D2Q9 or D3Q27
    :param shear_relaxation_rate: relaxation rate that determines viscosity
    :param higher_order_relaxation_rate: relaxation rate for higher order moments
    :param method_name: string 'KBC-Nx' where x can be an number from 1 to 4, for details see
                       "Karlin 2015: Entropic multi relaxation lattice Boltzmann models for turbulent flows"
    :param maxwellian_moments: determines if the discrete or continuous maxwellian equilibrium is
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

    energy_transport_tensor = list(exponents_to_polynomial_representations([a for a in moments_of_order(3, dim, True)
                                                                            if 3 not in a]))

    explicitly_defined = set(rho + velocity + shear_tensor_off_diagonal +
                             shear_tensor_diagonal + energy_transport_tensor)
    rest = list(set(exponents_to_polynomial_representations(moments_up_to_component_order(2, dim))) - explicitly_defined)
    assert len(rest) + len(explicitly_defined) == 3**dim

    # naming according to paper Karlin 2015: Entropic multirelaxation lattice Boltzmann models for turbulent flows
    d = shear_tensor_off_diagonal + shear_tensor_trace_free_diagonal[:-1]
    t = [shear_tensor_trace]
    q = energy_transport_tensor
    if method_name == 'KBC-N1':
        decomposition = [d, t+q+rest]
    elif method_name == 'KBC-N2':
        decomposition = [d+t, q+rest]
    elif method_name == 'KBC-N3':
        decomposition = [d+q, t+rest]
    elif method_name == 'KBC-N4':
        decomposition = [d+t+q, rest]
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


def create_mrt_orthogonal(stencil, relaxation_rate_getter=None, maxwellian_moments=False, **kwargs):
    r"""
    Returns a orthogonal multi-relaxation time model for the stencils D2Q9, D3Q15, D3Q19 and D3Q27.
    These MRT methods are just one specific version - there are many MRT methods possible for all these stencils
    which differ by the linear combination of moments and the grouping into equal relaxation times.
    To create a generic MRT method use :func:`lbmpy.methods.create_with_discrete_maxwellian_eq_moments`

    :param stencil: nested tuple defining the discrete velocity space. See `func:lbmpy.stencils.get_stencil`
    :param relaxation_rate_getter: function getting a list of moments as argument, returning the associated relaxation
                                 time. The default returns:

                                    - 0 for moments of order 0 and 1 (conserved)
                                    - :math:`\omega`: from moments of order 2 (rate that determines viscosity)
                                    - numbered :math:`\omega_i` for the rest
    :param maxwellian_moments: determines if the discrete or continuous maxwellian equilibrium is
                                               used to compute the equilibrium moments
    """
    if relaxation_rate_getter is None:
        relaxation_rate_getter = default_relaxation_rate_names()
    if isinstance(stencil, str):
        stencil = get_stencil(stencil)

    x, y, z = MOMENT_SYMBOLS
    one = sp.Rational(1, 1)

    moment_to_relaxation_rate_dict = OrderedDict()
    if stencils_have_same_entries(stencil, get_stencil("D2Q9")):
        moments = get_default_moment_set_for_stencil(stencil)
        orthogonal_moments = gram_schmidt(moments, stencil)
        orthogonal_moments_scaled = [e * common_denominator(e) for e in orthogonal_moments]
        nested_moments = list(sort_moments_into_groups_of_same_order(orthogonal_moments_scaled).values())
    elif stencils_have_same_entries(stencil, get_stencil("D3Q15")):
        sq = x ** 2 + y ** 2 + z ** 2
        nested_moments = [
            [one, x, y, z],  # [0, 3, 5, 7]
            [sq - 1],  # [1]
            [3 * sq ** 2 - 6 * sq + 1],  # [2]
            [(3 * sq - 5) * x, (3 * sq - 5) * y, (3 * sq - 5) * z],  # [4, 6, 8]
            [3 * x ** 2 - sq, y ** 2 - z ** 2, x * y, y * z, x * z],  # [9, 10, 11, 12, 13]
            [x * y * z]
        ]
    elif stencils_have_same_entries(stencil, get_stencil("D3Q19")):
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
    elif stencils_have_same_entries(stencil, get_stencil("D3Q27")):
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

    for momentList in nested_moments:
        rr = relaxation_rate_getter(momentList)
        for m in momentList:
            moment_to_relaxation_rate_dict[m] = rr

    if maxwellian_moments:
        return create_with_continuous_maxwellian_eq_moments(stencil, moment_to_relaxation_rate_dict, **kwargs)
    else:
        return create_with_discrete_maxwellian_eq_moments(stencil, moment_to_relaxation_rate_dict, **kwargs)


# ----------------------------------------- Comparison view for notebooks ----------------------------------------------


def compare_moment_based_lb_methods(reference, other, show_deviations_only=False):
    import ipy_table
    table = []
    caption_rows = [len(table)]
    table.append(['Shared Moment', 'ref', 'other', 'difference'])

    reference_moments = set(reference.moments)
    other_moments = set(other.moments)
    for moment in reference_moments.intersection(other_moments):
        reference_value = reference.relaxation_info_dict[moment].equilibriumValue
        other_value = other.relaxation_info_dict[moment].equilibriumValue
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
            val = reference.relaxation_info_dict[moment].equilibriumValue
            table.append(["$%s$" % (sp.latex(moment),),
                          "$%s$" % (sp.latex(val),),
                          " ", " "])

    only_in_other = other_moments - reference_moments
    if only_in_other:
        caption_rows.append(len(table))
        table.append(['Only in Other', '', 'value', ''])
        for moment in only_in_other:
            val = other.relaxation_info_dict[moment].equilibriumValue
            table.append(["$%s$" % (sp.latex(moment),),
                          " ",
                          "$%s$" % (sp.latex(val),),
                          " "])

    table_display = ipy_table.make_table(table)
    for row_idx in caption_rows:
        for col in range(4):
            ipy_table.set_cell_style(row_idx, col, color='#bbbbbb')
    return table_display
