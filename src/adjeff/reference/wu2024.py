"""The sampled PSF of Wu et al. (2024), reimplemented on Smart-G.

Reference
---------
Wu, Y. et al. (2024). Sensor-generic adjacency-effect correction for
remote sensing of coastal and inland waters.  *Remote Sensing of
Environment*, 315, 114433.

The method has no closed-form expression: the PSF is *sampled* by Monte
Carlo rather than fitted to a parametric shape.  Photons are launched
backward from the sensor and the energy they deposit on a surface
entity, as a function of its position, is the PSF.

What it deliberately leaves out is the earth-atmosphere coupling: a
photon that bounces off the surface, scatters in the atmosphere and
comes back down is not counted.  That assumption is what adjeff drops,
and comparing against this sampler is how the difference is measured.
The upstream implementation, T-Mart, exposes only a corrected image and
never the kernel, so the method is reimplemented here to make the
comparison possible at the PSF level.
"""

from __future__ import annotations

from typing import Any, ClassVar

import xarray as xr

import adjeff.atmosphere as atmo
from adjeff.core import ImageDict
from adjeff.modules.samplers._smartg import psf_atm
from adjeff.modules.sweep_sampler import SweepSampler
from adjeff.utils import CacheStore
from adjeff.utils._config import ConfigProtocol

from .._logging import get_logger

logger = get_logger(__name__)


def _psf_atm_from_scalars(
    vza: float,
    vaa: float,
    aot: float,
    rh: float,
    h: float,
    href: float,
    **statics: Any,
) -> xr.DataArray:
    """Call :func:`psf_atm` with the atmospheric state as 0-d arrays.

    Every parameter is a ``loop`` variable, so xsweep hands over native
    scalars, which is what an external engine usually wants.  Smart-G's
    atmosphere builder wants DataArrays, so they are wrapped back here
    rather than in the physics, which keeps the wrapping at the one place
    that knows why it is needed.
    """
    return psf_atm(
        vza=vza,
        vaa=vaa,
        aot=xr.DataArray(aot),
        rh=xr.DataArray(rh),
        h=xr.DataArray(h),
        href=xr.DataArray(href),
        **statics,
    )


class WuPsfSampler(SweepSampler):
    """Sample the atmospheric PSF the way Wu et al. (2024) do.

    A published method rather than one of adjeff's own: it lives in
    :mod:`adjeff.reference` because its purpose is to be compared
    against, not to be improved.

    Photons are launched backward from the sensor, propagated through
    the atmosphere until they hit a Smart-G ``Entity`` placed on the
    surface.  The energy deposited on the entity as a function of its
    position gives the atmospheric PSF directly, without coupling to
    surface adjacency effects.

    Every parameter is a ``loop`` variable: one Smart-G call per
    combination.  Nothing rides along as ``vec`` here, since the sensor
    geometry is rebuilt for each angle and the kernel is the output.

    Produced variable: ``psf_atm``.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Geometry — ``vza`` and ``sza`` must be single-element per call.
    spectral_config : SpectralConfig
        Bands to process.
    remove_rayleigh : bool
        Whether to suppress Rayleigh scattering.
    afgl_type : str
        AFGL atmosphere profile identifier.
    nr : int
        Number of radial sampling points.
    n_ph : int
        Number of photons per sensor.
    """

    _required_vars: ClassVar[list[str]] = []
    _output_vars: ClassVar[list[str]] = ["psf_atm"]
    contract: ClassVar[str] = (
        # Dim order is the one adapt_smartg_output produces: x then y.
        "loop(vza, vaa, aot, rh, h, href) -> psf_atm(x, y)"
    )
    point_fn: ClassVar[Any] = staticmethod(_psf_atm_from_scalars)

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        nr: int = 100,
        n_ph: int = int(1e6),
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
        rename: dict[str, str] | None = None,
    ) -> None:
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.remove_rayleigh = remove_rayleigh
        self.afgl_type = afgl_type
        self.nr = nr
        self.n_ph = n_ph
        super().__init__(cache=cache, batch_size=batch_size, dedup=dedup, rename=rename)

    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        return (self.atmo_config, self.geo_config)

    def _statics(self) -> dict[str, Any]:
        return {
            "species": self.atmo_config.species,
            "afgl_type": self.afgl_type,
            "remove_rayleigh": self.remove_rayleigh,
            "n_ph": self.n_ph,
        }

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Run the atmospheric PSF sampling for every band in the scene."""
        # One sweep per band: the physics reads the band's own scene to
        # size its sampling grid, so the two travel together.
        for band in scene.bands:
            arr = self._sweep(rho_s=scene[band], band=band)
            scene[band][self._slot("psf_atm")] = self._restore_coords(arr, scene[band])
            logger.info("Computed atmospheric PSF.", dims=arr.dims, band=band)
        return scene
