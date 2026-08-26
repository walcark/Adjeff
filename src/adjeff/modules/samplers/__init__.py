"""Smart-G Monte-Carlo samplers for radiative transfer quantities.

All samplers extend :class:`~adjeff.modules.SweepSampler` and
require a CUDA-capable GPU.  They compute the six quantities needed
by the 5S radiative transfer formula::

    rho_toa = rho_atm
              + (tdir_up + tdif_up) * (tdir_down + tdif_down) * rho_unif
                / (1 - sph_alb * rho_unif)

**Transmittance samplers** (analytical from optical depth):

- :class:`TdirDownSampler` — direct downward transmittance ``tdir_down``.
- :class:`TdirUpSampler`   — direct upward transmittance ``tdir_up``.
- :class:`TdifDownSampler` — diffuse downward transmittance ``tdif_down``.
- :class:`TdifUpSampler`   — diffuse upward transmittance ``tdif_up``.

**Reflectance samplers** (full Monte-Carlo):

- :class:`SphAlbSampler`   — atmospheric spherical albedo ``sph_alb``.
- :class:`RhoAtmSampler`   — atmospheric path reflectance ``rho_atm``.

**TOA reflectance samplers** (require ``rho_s`` in the scene):

- :class:`RhoToaSymSampler` — radial sampling (symmetric PSF assumption).
- :class:`RhoToaSampler`    — full 2D grid sampling (arbitrary surface).

**Atmospheric PSF**: see :class:`adjeff.reference.WuPsfSampler`, which
implements a published method and therefore lives in
:mod:`adjeff.reference` rather than here.

**Convenience pipeline**:

- :class:`RadiativePipeline` — chains all six standard samplers.
"""

from .radiatives import RadiativePipeline
from .rho_atm import RhoAtmSampler
from .rho_toa import RhoToaSampler
from .rho_toa_sym import RhoToaSymSampler
from .sph_alb import SphAlbSampler
from .tdif_down import TdifDownSampler
from .tdif_up import TdifUpSampler
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
    "SphAlbSampler",
    "TdifDownSampler",
    "TdifUpSampler",
    "TdirDownSampler",
    "TdirUpSampler",
]
