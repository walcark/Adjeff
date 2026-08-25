"""Direct downward transmittance (``tdir_down``) sampler using Smart-G."""

from typing import Any, Callable, ClassVar

import adjeff.atmosphere as atmo
from adjeff.utils import CacheStore
from adjeff.utils._config import ConfigProtocol

from ..sweep_sampler import SweepSampler
from ._smartg import tdir_down


class TdirDownSampler(SweepSampler):
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
    batch_size : int, optional
        Atmospheric states per Smart-G call.
    dedup : bool, optional
        Collapse repeated states before calling.
    """

    _required_vars: ClassVar[list[str]] = []
    _output_vars: ClassVar[list[str]] = ["tdir_down"]
    contract: ClassVar[str] = "batch(aot, rh, h, href, sza) vec(wl) -> tdir_down(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(tdir_down)

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(1e2),
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
        rename: dict[str, str] | None = None,
    ) -> None:
        self.spectral_config = spectral_config
        self.atmo_config = atmo_config
        self.geo_config = geo_config
        self.afgl_type = afgl_type
        self.remove_rayleigh = remove_rayleigh
        self.n_ph = n_ph
        super().__init__(
            cache=cache, batch_size=batch_size, dedup=dedup, rename=rename
        )

    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        return (self.spectral_config, self.atmo_config, self.geo_config)

    def _statics(self) -> dict[str, Any]:
        return {
            "species": self.atmo_config.species,
            "afgl_type": self.afgl_type,
            "remove_rayleigh": self.remove_rayleigh,
            "n_ph": self.n_ph,
        }
