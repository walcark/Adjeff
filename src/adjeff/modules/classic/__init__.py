"""Closed-form 5S models, assuming a uniform Lambertian surface (no GPU).

Both read the six radiative terms produced by
:mod:`adjeff.modules.samplers`.

Classes
-------
    Unif2Toa
        Forward model: ``rho_unif`` to ``rho_toa``.
    Toa2Unif
        Inverse model: ``rho_toa`` to ``rho_unif``.
"""

from .toa_to_unif import Toa2Unif
from .unif_to_toa import Unif2Toa

__all__ = ["Toa2Unif", "Unif2Toa"]
