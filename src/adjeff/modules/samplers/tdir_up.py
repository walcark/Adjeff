"""Direct upward transmittance (``tdir_up``) sampler using Smart-G."""

from typing import Any, Callable, ClassVar

import adjeff.atmosphere as atmo
from adjeff.utils import CacheStore
from adjeff.utils._config import ConfigProtocol

from ..sweep_sampler import SweepSampler
from ._smartg import tdir_up


class TdirUpSampler(SweepSampler):
    """Sample direct upward transmittance analytically from optical depth.

    ``tdir_up`` is the fraction of the surface-reflected flux that
    reaches the satellite without being scattered.  Computed
    analytically from the total optical depth ``OD`` retrieved by
    Smart-G::

        tdir_up = exp(-OD / cos(vza))

    This is the upward counterpart of :class:`TdirDownSampler`.

    Produced variable: ``tdir_up``.

    Notes
    -----
    Requires a CUDA-capable GPU (for the Smart-G optical depth
    retrieval), but very few photons suffice (default ``n_ph=1e9``).

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Viewing geometry (``vza``).
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh optical depth is set to zero.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier,
        by default ``"afgl_exp_h8km"``.
    n_ph : int, optional
        Number of photons for the optical depth retrieval,
        by default ``1e9``.
    cache : CacheStore or None, optional
        Result cache; ``None`` disables caching.
    batch_size : int, optional
        Atmospheric states per Smart-G call.
    dedup : bool, optional
        Collapse repeated states before calling.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["tdir_up"]
    contract: ClassVar[str] = "batch(aot, rh, h, href, vza) vec(wl) -> tdir_up(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(tdir_up)

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(1e9),
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
    ) -> None:
        self.spectral_config = spectral_config
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.afgl_type = afgl_type
        self.remove_rayleigh = remove_rayleigh
        self.n_ph = n_ph
        super().__init__(cache=cache, batch_size=batch_size, dedup=dedup)

    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        return (self.spectral_config, self.atmo_config, self.geo_config)

    def _statics(self) -> dict[str, Any]:
        return {
            "species": self.atmo_config.species,
            "afgl_type": self.afgl_type,
            "remove_rayleigh": self.remove_rayleigh,
            "n_ph": self.n_ph,
        }
