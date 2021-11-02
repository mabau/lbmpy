from enum import Enum, auto


class Stencil(Enum):
    """
    The Stencil enumeration represents all possible lattice Boltzmann stencils that are available in lbmpy.
    It should be passed to :class:`lbmpy.stencils.LBStenil`. This class then creates a stencils representation
    containing the concrete neighbour directions as a tuple of tuples.

    The number of spatial dimensions *d* and the number of discrete velocities *q* are stated in the DdQq notation
    """
    D2Q9 = auto()
    """
    A two dimensional stencil using 9 discrete velocities.
    """
    D2V17 = auto()
    """
    A two dimensional stencil using 17 discrete velocities. (long range stencil).
    """
    D2V37 = auto()
    """
    A two dimensional stencil using 37 discrete velocities. (long range stencil).
    """
    D3Q7 = auto()
    """
    A three dimensional stencil using 7 discrete velocities.
    """
    D3Q15 = auto()
    """
    A three dimensional stencil using 15 discrete velocities.
    """
    D3Q19 = auto()
    """
    A three dimensional stencil using 19 discrete velocities.
    """
    D3Q27 = auto()
    """
    A three dimensional stencil using 27 discrete velocities.
    """


class Method(Enum):
    """
    The Method enumeration represents all possible lattice Boltzmann collision operators that are available in lbmpy.
    It should be passed to :class:`lbmpy.creationfunctions.LBMConfig`. The LBM configuration *dataclass* then derives
    the respective collision equations when passed to the creations functions in the `lbmpy.creationfunctions`
    module of lbmpy.

    Note here, when using a specific enumeration to derive a particular LBM collision operator,
    different parameters of the :class:`lbmpy.creationfunctions.LBMConfig` might become necessary.
    For example, it does not make sense to define *relaxation_rates* for a single relaxation rate method, which
    is essential for multiple relaxation rate methods. Important specific parameters are listed below to the enum value.
    A specific creation function is stated for each case which explains these parameters in detail.
    """
    SRT = auto()
    """
    See :func:`lbmpy.methods.create_srt`, 
    Single relaxation time method
    """
    TRT = auto()
    """
    See :func:`lbmpy.methods.create_trt`, 
    Two relaxation time, the first relaxation rate is for even moments and determines the
    viscosity (as in SRT). The second relaxation rate is used for relaxing odd moments and controls the
    bulk viscosity. For details in the TRT collision operator see :cite:`TRT`
    """
    MRT_RAW = auto()
    """
    See :func:`lbmpy.methods.create_mrt_raw`, 
    Non-orthogonal MRT where all relaxation rates can be specified independently, i.e. there are as many relaxation 
    rates as stencil entries. Look at the generated method in Jupyter to see which moment<->relaxation rate mapping.
    Originally defined in :cite:`raw_moments`
    """
    MRT = auto()
    """
    See :func:`lbmpy.methods.create_mrt_orthogonal`
    Orthogonal multi relaxation time model, relaxation rates are used in this order for *shear modes*, *bulk modes*,
    *third-order modes*, *fourth-order modes*, etc. Requires also a parameter *weighted* that should be `True` if the
    moments should be orthogonal w.r.t. weighted scalar product using the lattice weights. If `False`, the normal
    scalar product is used. For custom definition of the method, a *nested_moments* can be passed.
    For example: [ [1, x, y], [x*y, x**2, y**2], ... ] that groups all moments together that should be relaxed
    at the same rate. Literature values of this list can be obtained through 
    :func:`lbmpy.methods.creationfunctions.mrt_orthogonal_modes_literature`.
    WMRT collision operators are reported to be numerically more stable and more accurate, 
    whilst also having a lower computational cos :cite:`FAKHARI201722`
    """
    CENTRAL_MOMENT = auto()
    """
    See :func:`lbmpy.methods.create_central_moment`
    Creates moment based LB method where the collision takes place in the central moment space. By default, 
    a raw-moment set is used where the bulk and the shear viscosity are separated. An original derivation can be 
    found in :cite:`Geier2006`
    """
    TRT_KBC_N1 = auto()
    """
    See :func:`lbmpy.methods.create_trt_kbc`
    Particular two-relaxation rate method. This is not the entropic method yet, only the relaxation pattern. 
    To get the entropic method also *entropic* needs to be set to `True`. 
    There are four KBC methods available in lbmpy. The naming is according to :cite:`karlin2015entropic`
    """
    TRT_KBC_N2 = auto()
    """
    See :func:`lbmpy.methods.create_trt_kbc`
    Particular two-relaxation rate method. This is not the entropic method yet, only the relaxation pattern. 
    To get the entropic method also *entropic* needs to be set to `True`. 
    There are four KBC methods available in lbmpy. The naming is according to :cite:`karlin2015entropic`
    """
    TRT_KBC_N3 = auto()
    """
    See :func:`lbmpy.methods.create_trt_kbc`
    Particular two-relaxation rate method. This is not the entropic method yet, only the relaxation pattern. 
    To get the entropic method also *entropic* needs to be set to `True`. 
    There are four KBC methods available in lbmpy. The naming is according to :cite:`karlin2015entropic`
    """
    TRT_KBC_N4 = auto()
    """
    See :func:`lbmpy.methods.create_trt_kbc`
    Particular two-relaxation rate method. This is not the entropic method yet, only the relaxation pattern. 
    To get the entropic method also *entropic* needs to be set to `True`. 
    There are four KBC methods available in lbmpy. The naming is according to :cite:`karlin2015entropic`
    """
    ENTROPIC_SRT = auto()
    """
    See :func:`lbmpy.methods.create_srt_entropic`,
    An entropic version of the isothermal lattice Boltzmann method with the simplicity and 
    computational efficiency of the standard lattice Boltzmann model. For details see :cite:`Ansumali2003`
    """
    CUMULANT = auto()
    """
    See :func:`lbmpy.methods.create_with_default_polynomial_cumulants`
    Cumulant-based LB method which relaxes groups of polynomial cumulants chosen to optimize rotational invariance.
    For details on the method see :cite:`geier2015`
    """
    MONOMIAL_CUMULANT = auto()
    """
    See :func:`lbmpy.methods.create_with_monomial_cumulants`
    Cumulant-based LB method which relaxes monomial cumulants.
    For details on the method see :cite:`geier2015` and :cite:`Coreixas2019`
    """


class ForceModel(Enum):
    """
    The ForceModel enumeration defines which force model is used to introduce forcing terms in the collision operator
    of the lattice Boltzmann method. A short summary of the theory behind is shown in `lbmpy.forcemodels`.
    More precise definitions are given in Chapter 6 and 10 of :cite:`lbm_book`
    """
    SIMPLE = auto()
    """
    See :class:`lbmpy.forcemodels.Simple`
    """
    LUO = auto()
    """
    See :class:`lbmpy.forcemodels.Luo`
    """
    GUO = auto()
    """
    See :class:`lbmpy.forcemodels.Guo`
    """
    BUICK = auto()
    """
    See :class:`lbmpy.forcemodels.Buick`
    """
    SILVA = auto()
    """
    See :class:`lbmpy.forcemodels.Buick`
    """
    EDM = auto()
    """
    See :class:`lbmpy.forcemodels.EDM`
    """
    KUPERSHTOKH = auto()
    """
    See :class:`lbmpy.forcemodels.EDM`
    """
    CUMULANT = auto()
    """
    See :class:`lbmpy.methods.centeredcumulant.CenteredCumulantForceModel`
    """
    HE = auto()
    """
    See :class:`lbmpy.forcemodels.He`
    """
    SHANCHEN = auto()
    """
    See :class:`lbmpy.forcemodels.ShanChen`
    """
