"""Direct downward transmittance (``tdir_down``) sampler using Smart-G."""

from typing import ClassVar

import xarray as xr
from structlog import get_logger

import adjeff.atmosphere as atmo
import adjeff.utils as utils
from adjeff.core import ImageDict

from ..scene_module_sweep import SceneModuleSweep
from ._smartg import tdir_down

logger = get_logger(__name__)


class TdirDownSampler(SceneModuleSweep):
    """Sample direct downward transmittance analytically from optical depth.

    ``tdir_down`` is the fraction of the solar flux that reaches the
    surface without being scattered.  It is computed analytically from
    the total optical depth ``OD`` retrieved by Smart-G::

        tdir_down = exp(-OD / cos(sza))

    The Monte-Carlo photon count only affects the optical depth
    retrieval and can therefore be kept very small (default ``1e2``).
    This is the downward counterpart of :class:`TdirUpSampler`.

    Produced variable: ``tdir_down``.

    Notes
    -----
    Requires a CUDA-capable GPU (for the Smart-G optical depth
    retrieval), but very few photons suffice (default ``n_ph=1e2``).

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Illumination geometry (``sza``).
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh optical depth is set to zero.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier,
        by default ``"afgl_exp_h8km"``.
    n_ph : int, optional
        Number of photons for the optical depth retrieval,
        by default ``1e2``.
    cache : CacheStore or None, optional
        Result cache; ``None`` disables caching.
    sweep_chunks : dict[str, int] or None, optional
        Chunk sizes for Smart-G calls within this module.
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before sweeping.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["tdir_down"]
    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = ["wl", "aot", "rh", "h", "href", "sza"]

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(1e2),
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
        """Run the Smart-G sweep and write ``tdir_down`` into each band."""
        for band in self.spectral_config.bands:
            if band not in scene.bands:
                scene[band] = xr.Dataset()

        arr: xr.DataArray = self._apply_bundle(
            tdir_down,
            species=self.atmo_config.species,
            afgl_type=self.afgl_type,
            remove_rayleigh=self.remove_rayleigh,
            n_ph=self.n_ph,
        )

        for band in self.spectral_config.bands:
            scene[band]["tdir_down"] = arr.sel(wl=band.wl_nm)

        return scene
