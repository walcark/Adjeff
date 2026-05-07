"""Analytical 5S scene modules (no GPU required).

These modules implement the closed-form 5S radiative transfer equations
directly, without Monte-Carlo simulation:

- :class:`Unif2Toa` — forward model: ``rho_unif`` → ``rho_toa``.
- :class:`Toa2Unif` — inverse model: ``rho_toa`` → ``rho_unif``.

Both assume a uniform Lambertian surface
(``rho_s = rho_env = rho_unif``) and require the six radiative
quantities produced by :mod:`adjeff.modules.samplers` as input
variables.
"""

from .toa_to_unif import Toa2Unif
from .unif_to_toa import Unif2Toa

__all__ = ["Toa2Unif", "Unif2Toa"]
