"""Smart-G samplers of the 5S terms (CUDA GPU required).

The six terms enter::

    rho_toa = rho_atm
              + (tdir_up + tdif_up) * (tdir_down + tdif_down) * rho_unif
                / (1 - sph_alb * rho_unif)

Classes
-------
    TdirDownSampler, TdirUpSampler
        Direct transmittances, from the optical depth.
    TdifDownSampler, TdifUpSampler
        Diffuse transmittances, by Monte Carlo.
    SphAlbSampler
        Spherical albedo.
    RhoAtmSampler
        Atmospheric path reflectance.
    TdifUpBrdfSampler, SphAlbBrdfSampler
        ``tdif_up`` and ``sph_alb`` over an RTLS surface; they write the
        same variables as their Lambertian counterparts.
    RhoToaSymSampler
        ``rho_toa`` of a radially symmetric surface, on a few radii.
    RhoToaSampler
        ``rho_toa`` of an arbitrary surface, on a pixel sub-grid.
    RadiativePipeline
        The six standard samplers in a row.

Constants
---------
    RADIATIVE_VARS
        Names of the six 5S terms.

The atmospheric PSF of Wu et al. lives in :mod:`adjeff.reference`.
"""

from .radiatives import RadiativePipeline
from .rho_atm import RhoAtmSampler
from .rho_toa import RhoToaSampler
from .rho_toa_sym import RhoToaSymSampler
from .sph_alb import SphAlbSampler
from .sph_alb_brdf import SphAlbBrdfSampler
from .tdif_down import TdifDownSampler
from .tdif_up import TdifUpSampler
from .tdif_up_brdf import TdifUpBrdfSampler
from .tdir_down import TdirDownSampler
from .tdir_up import TdirUpSampler

#: The six quantities the 5S formula needs.  Each sampler declares its
#: own as an output; the list is published here because every caller
#: that reasons about "the radiative quantities" was rebuilding it.
RADIATIVE_VARS: tuple[str, ...] = (
    "tdir_down",
    "tdir_up",
    "tdif_down",
    "tdif_up",
    "rho_atm",
    "sph_alb",
)

__all__ = [
    "RadiativePipeline",
    "RADIATIVE_VARS",
    "RhoAtmSampler",
    "RhoToaSampler",
    "RhoToaSymSampler",
    "SphAlbBrdfSampler",
    "SphAlbSampler",
    "TdifDownSampler",
    "TdifUpBrdfSampler",
    "TdifUpSampler",
    "TdirDownSampler",
    "TdirUpSampler",
]
