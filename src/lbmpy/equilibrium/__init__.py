r"""
This module contains various classes encapsulating equilibrium distributions used in the lattice Boltzmann
method. These include both the continuous and the discretized variants of the Maxwellian equilibrium of
hydrodynamics. Furthermore, a lightweight wrapper class for custom discrete equilibria is provided.
Custom equilibria may also be implemented by manually overriding the abstract base class 
:class:`lbmpy.equilibrium.AbstractEquilibrium`.
"""

from .abstract_equilibrium import AbstractEquilibrium
from .continuous_hydro_maxwellian import ContinuousHydrodynamicMaxwellian, default_background_distribution
from .generic_discrete_equilibrium import GenericDiscreteEquilibrium, discrete_equilibrium_from_matching_moments
from .discrete_hydro_maxwellian import DiscreteHydrodynamicMaxwellian

__all__ = [
    "AbstractEquilibrium",
    "ContinuousHydrodynamicMaxwellian", "default_background_distribution",
    "GenericDiscreteEquilibrium", "discrete_equilibrium_from_matching_moments",
    "DiscreteHydrodynamicMaxwellian"
]
