"""Module that computes rho_toa with Smart-G (symmetric radial sampling)."""

from __future__ import annotations

from typing import ClassVar

import xarray as xr
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict

from ..scene_module_sweep import SceneModuleSweep
from ._smartg import rho_toa_sym
from .rho_atm import SmartgSampler_Rho_atm

logger = get_logger(__name__)


class SmartgSampler_Rho_toa_sym(SceneModuleSweep):
    """Compute rho_toa by radial sampling under the symmetric PSF assumption.

    For each band, the module:

    1. Reads ``rho_s`` and bins it into a radial profile (``nr`` points from
       centre to image edge) — these radii are the sampling locations.
    2. Batches all AtmoConfig combinations (aot, rh, h, href) with the band
       wavelength, building a multi-profile atmosphere via ``multi_profiles``.
    3. Creates one position-specific sensor per radial point.
    4. Runs a single Smart-G simulation and writes ``rho_toa`` with dims
       ``(r, aot, rh, h, href)`` into the scene.

    vza and sza are scalar per call — sensor positions depend on vza so they
    cannot be vectorised within one run. Sweep over angles externally.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric parameters — may be full arrays (swept via multi_profiles).
    geo_config : GeoConfig
        Geometry — vza and sza must be single-element (scalar per call).
    remove_rayleigh : bool
        Whether to suppress Rayleigh scattering.
    afgl_type : str
        AFGL atmosphere profile identifier.
    nr : int
        Number of radial sampling points.
    n_ph : int
        Number of photons per sensor.
    """

    required_vars: ClassVar[list[str]] = ["rho_s"]
    output_vars: ClassVar[list[str]] = ["rho_toa"]
    scalar_dims: ClassVar[list[str]] = ["sza", "vza"]
    vector_dims: ClassVar[list[str]] = ["aot", "rh", "h", "href"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        nr: int = 100,
        n_ph: int = int(1e6),
        cache: utils.CacheStore | None = None,
    ) -> None:
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.remove_rayleigh = remove_rayleigh
        self.afgl_type = afgl_type
        self.nr = nr
        self.n_ph = n_ph
        super().__init__(cache=cache)

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        return (self.atmo_config, self.geo_config)

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Run the radial rho_toa computation for every band in the scene."""
        bundle: utils.ConfigBundle = self._make_bundle()

        scene = SmartgSampler_Rho_atm(
            atmo_config=self.atmo_config,
            geo_config=self.geo_config,
            spectral_config=atmo.SpectralConfig.from_bands(scene.bands),
            remove_rayleigh=self.remove_rayleigh,
            afgl_type=self.afgl_type,
            n_ph=int(3e7),
            cache=self._cache,
        )(scene)

        for band in scene.bands:
            logger.info("Start rho_toa computation.", band=band)
            rho_toa_arr: xr.DataArray = bundle.apply(
                rho_toa_sym,
                saa=self.geo_config.saa.item(),
                vaa=self.geo_config.vaa.item(),
                rho_s=scene[band],
                band=band,
                species=self.atmo_config.species,
                sat_height=self.geo_config.sat_height,
                afgl_type=self.afgl_type,
                remove_rayleigh=self.remove_rayleigh,
                nr=self.nr,
                n_ph=self.n_ph,
            )
            logger.info("Computed rho_toa.", dims=rho_toa_arr.dims, band=band)

            scene[band]["rho_toa"] = rho_toa_arr

        return scene
