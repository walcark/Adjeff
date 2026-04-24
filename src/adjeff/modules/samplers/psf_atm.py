"""Module that computes the atmospheric PSF with Smart-G."""

from __future__ import annotations

from typing import ClassVar

import xarray as xr
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict

from ..scene_module_sweep import SceneModuleSweep
from ._smartg import psf_atm

logger = get_logger(__name__)


class PsfAtmSampler(SceneModuleSweep):
    """Compute the atmospheric PSF of the 5S model.

    For each band, the module samples the PSF using a Smart-G Entity object.
    Photons are backward launched from the sensor, propagated through the
    atmosphere until they reach the Entity on the earth surface. When a photon
    hits the ground, its energy is registered by the Entity, and the photon
    path is terminated. This allows to avoid sampling coupling effects in the
    PSF model.

    vza and sza are scalar per call — sensor positions depend on vza so they
    cannot be vectorised within one run. Sweep over angles externally.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric parameters — may be full arrays (swept via multi_profiles).
    geo_config : GeoConfig
        Geometry — vza and sza must be single-element (scalar per call).
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

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["psf_atm"]
    scalar_dims: ClassVar[list[str]] = [
        "vza",
        "vaa",
        "aot",
        "rh",
        "h",
        "href",
    ]
    vector_dims: ClassVar[list[str]] = []

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
        """Run the atmospheric PSF sampling for every band in the scene."""
        bundle, _ = self._make_bundle()

        new_scene = ImageDict({b: xr.Dataset() for b in scene.bands})
        for band in scene.bands:
            psf_atm_arr: xr.DataArray = bundle.apply(
                psf_atm,
                rho_s=scene[band],
                band=band,
                species=self.atmo_config.species,
                afgl_type=self.afgl_type,
                remove_rayleigh=self.remove_rayleigh,
                n_ph=self.n_ph,
            )
            logger.info(
                "Computed atmospheric PSF.",
                dims=psf_atm_arr.dims,
                band=band,
            )

            new_scene[band]["psf_atm"] = psf_atm_arr
        return new_scene
