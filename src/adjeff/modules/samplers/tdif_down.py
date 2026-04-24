"""Module that computes tdif_down with Smart-G."""

from typing import ClassVar

import xarray as xr
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict

from ..scene_module_sweep import SceneModuleSweep
from ._smartg import tdif_down

logger = get_logger(__name__)


class TdifDownSampler(SceneModuleSweep):
    """Sample downward diffuse transmittance with Smart-G.

    Computes ``tdif_down`` — the diffuse fraction of the solar flux
    reaching the surface — for every combination of illumination geometry
    and atmospheric state defined by the supplied configs.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Illumination geometry (``sza``, ``saa``).
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier,
        by default ``"afgl_exp_h8km"``.
    n_ph : int, optional
        Number of photons per Smart-G call, by default ``3e7``.
    cache : CacheStore or None, optional
        Result cache; ``None`` disables caching.
    sweep_chunks : dict[str, int] or None, optional
        Chunk sizes for Smart-G calls within this module.
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before sweeping.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["tdif_down"]
    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = ["wl", "aot", "rh", "h", "href", "sza"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(3e7),
        cache: utils.CacheStore | None = None,
        sweep_chunks: dict[str, int] | None = None,
        deduplicate_dims: list[str] | None = None,
    ) -> None:
        self.spectral_config = spectral_config
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.afgl_type = afgl_type
        self.remove_rayleigh = remove_rayleigh
        self.n_ph = n_ph
        super().__init__(
            cache=cache,
            sweep_chunks=sweep_chunks,
            deduplicate_dims=deduplicate_dims,
        )

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        return (self.spectral_config, self.atmo_config, self.geo_config)

    def _compute(self, scene: ImageDict) -> ImageDict:
        for band in self.spectral_config.bands:
            if band not in scene.bands:
                scene[band] = xr.Dataset()

        arr: xr.DataArray = self._apply_bundle(
            tdif_down,
            species=self.atmo_config.species,
            afgl_type=self.afgl_type,
            remove_rayleigh=self.remove_rayleigh,
            n_ph=self.n_ph,
            saa=self.geo_config.saa.values,
            sat_height=self.geo_config.sat_height,
        )

        for band in self.spectral_config.bands:
            scene[band]["tdif_down"] = arr.sel(wl=band.wl_nm)

        return scene
