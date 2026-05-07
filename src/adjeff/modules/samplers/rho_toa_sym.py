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
from .rho_atm import RhoAtmSampler

logger = get_logger(__name__)


class RhoToaSymSampler(SceneModuleSweep):
    """Compute rho_toa by radial sampling under the symmetric PSF assumption.

    Assumes the scene is radially symmetric around the image centre.
    For each band the module:

    1. Bins ``rho_s`` into ``nr`` radial samples from centre to edge.
    2. Places one Smart-G sensor per radial point.
    3. Runs a single simulation per ``(sza, vza)`` combination, sweeping
       all atmospheric states (``aot``, ``rh``, ``h``, ``href``) as a
       multi-profile atmosphere.
    4. Writes ``rho_toa`` with dims ``(r, aot, rh, h, href)`` into the
       scene.

    ``sza`` and ``vza`` are ``scalar_dims`` — sensor positions depend on
    ``vza`` and cannot be vectorised within one call.

    Produced variable: ``rho_toa``.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Geometry — ``vza`` and ``sza`` must be single-element per call.
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
        bundle, _ = self._make_bundle()

        scene = RhoAtmSampler(
            atmo_config=self.atmo_config,
            geo_config=self.geo_config,
            spectral_config=atmo.SpectralConfig.from_bands(scene.bands),
            remove_rayleigh=self.remove_rayleigh,
            afgl_type=self.afgl_type,
            n_ph=int(3e7),
            cache=self._cache,
        )(scene)

        for band in scene.bands:
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

            scene[band]["rho_toa"] = rho_toa_arr

        return scene
