"""Module that computes spherical albedo with Smart-G."""

from typing import ClassVar

import xarray as xr
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict

from ..scene_module_sweep import SceneModuleSweep
from ._smartg import sph_alb

logger = get_logger(__name__)


class SmartgSampler_Sph_alb(SceneModuleSweep):
    """Sample the spherical albedo of the atmosphere with Smart-G.

    Computes ``sph_alb`` — the fraction of the upwelling flux reflected
    back downward by the atmosphere — for every combination of atmospheric
    state defined by the supplied configs.  Geometry-independent: no
    ``geo_config`` is required.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier,
        by default ``"afgl_exp_h8km"``.
    n_ph : int, optional
        Number of photons per Smart-G call, by default ``2e7``.
    cache : CacheStore or None, optional
        Result cache; ``None`` disables caching.
    chunks : dict[str, int] or None, optional
        Chunk sizes for vector dimensions.
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before sweeping.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["sph_alb"]
    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = ["wl", "aot", "rh", "h", "href"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(2e7),
        cache: utils.CacheStore | None = None,
        chunks: dict[str, int] | None = None,
        deduplicate_dims: list[str] | None = None,
    ) -> None:
        self.spectral_config = spectral_config
        self.atmo_config = atmo_config
        self.afgl_type = afgl_type
        self.remove_rayleigh = remove_rayleigh
        self.n_ph = n_ph
        super().__init__(
            cache=cache,
            chunks=chunks,
            deduplicate_dims=deduplicate_dims,
        )

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        return (self.spectral_config, self.atmo_config)

    def _compute(self, scene: ImageDict) -> ImageDict:
        for band in self.spectral_config.bands:
            if band not in scene.bands:
                scene[band] = xr.Dataset()

        bundle: utils.ConfigBundle = self._make_bundle()
        arr: xr.DataArray = bundle.apply(
            sph_alb,
            species=self.atmo_config.species,
            afgl_type=self.afgl_type,
            remove_rayleigh=self.remove_rayleigh,
            n_ph=self.n_ph,
        )
        logger.info("Computed sph_alb.", dims=arr.dims)

        for band in self.spectral_config.bands:
            scene[band]["sph_alb"] = arr.sel(wl=band.wl_nm)

        return scene
