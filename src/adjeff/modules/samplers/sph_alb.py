"""Atmospheric spherical albedo (``sph_alb``) sampler using Smart-G."""

from typing import Any, Callable, ClassVar

import adjeff.atmosphere as atmo
import adjeff.utils as utils

from ..sweep_sampler import SweepSampler
from ._smartg import sph_alb


class SphAlbSampler(SweepSampler):
    """Sample the atmospheric spherical albedo with Smart-G Monte-Carlo.

    ``sph_alb`` is the fraction of the upwelling flux that is reflected
    back downward by the atmosphere.  It appears in the 5S formula as
    the multiple-reflection coupling term::

        rho_toa = rho_atm + t_up * t_down * rho_unif / (1 - sph_alb * rho_unif)

    Geometry-independent: no ``geo_config`` is required.  All parameters
    are swept as ``vector_dims``.

    Produced variable: ``sph_alb``.

    Notes
    -----
    Requires a CUDA-capable GPU.

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
    batch_size : int, optional
        Atmospheric states per Smart-G call.
    dedup : bool, optional
        Collapse repeated states before calling.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["sph_alb"]
    contract: ClassVar[str] = (
        "batch(aot, rh, h, href) vec(wl) -> sph_alb(wl)"
    )
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(sph_alb)

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(2e7),
        cache: utils.CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
    ) -> None:
        self.spectral_config = spectral_config
        self.atmo_config = atmo_config
        self.afgl_type = afgl_type
        self.remove_rayleigh = remove_rayleigh
        self.n_ph = n_ph
        super().__init__(cache=cache, batch_size=batch_size, dedup=dedup)

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        return (self.spectral_config, self.atmo_config)

    def _statics(self) -> dict[str, Any]:
        return {
            "species": self.atmo_config.species,
            "afgl_type": self.afgl_type,
            "remove_rayleigh": self.remove_rayleigh,
            "n_ph": self.n_ph,
        }
