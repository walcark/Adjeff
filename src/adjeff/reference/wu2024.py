"""Sampled PSF of Wu et al. (2024), reimplemented on Smart-G.

Wu, Y. et al. (2024). Sensor-generic adjacency-effect correction for
remote sensing of coastal and inland waters. *Remote Sensing of
Environment*, 315, 114433.

The PSF is sampled by backward Monte Carlo rather than fitted, and
leaves out earth-atmosphere coupling: that omission is what adjeff
measures by comparing against it.  T-Mart, the upstream code, exposes
only corrected images, hence this reimplementation.

Classes
-------
    WuPsfSampler
        Sampler producing ``psf_atm`` for every band of a scene.
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
    """Call :func:`psf_atm`, wrapping the atmospheric scalars as 0-d arrays."""
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
    """Atmospheric PSF sampled as in Wu et al. (2024).

    Photons are launched backward from the sensor; the energy they
    deposit on a surface ``Entity``, by position, is the PSF.  One
    Smart-G run per combo of ``(vza, vaa, aot, rh, h, href)``.
    Produces ``psf_atm``.  Requires a CUDA GPU.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state and aerosol species.
    geo_config : GeoConfig
        Viewing geometry; ``vza`` and ``vaa`` are swept.
    remove_rayleigh : bool
        Suppress Rayleigh scattering.
    afgl_type : str, optional
        AFGL atmosphere profile, ``"afgl_exp_h8km"`` by default.
    n_ph : int, optional
        Photons per run, 1e6 by default.
    cache, batch_size, dedup, rename : optional
        See :class:`~adjeff.modules.SweepSampler`.
    """

    _required_vars: ClassVar[list[str]] = []
    _output_vars: ClassVar[list[str]] = ["psf_atm"]
    contract: ClassVar[str] = "loop(vza, vaa, aot, rh, h, href) -> psf_atm(x, y)"
    point_fn: ClassVar[Any] = staticmethod(_psf_atm_from_scalars)

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
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
        self.n_ph = n_ph
        super().__init__(
            cache=cache,
            batch_size=batch_size,
            dedup=dedup,
            rename=rename,
        )

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
        for band in scene.bands:
            arr = self._sweep(rho_s=scene[band], band=band)
            scene[band][self._slot("psf_atm")] = self._restore_coords(arr, scene[band])
            logger.debug("wu_psf.done", dims=list(arr.dims), band=str(band))
        return scene
