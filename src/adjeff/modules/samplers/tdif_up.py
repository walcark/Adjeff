"""Diffuse upward transmittance (``tdif_up``) sampler using Smart-G."""

from typing import Any, Callable, ClassVar

import adjeff.atmosphere as atmo
import adjeff.utils as utils

from ..sweep_sampler import SweepSampler
from ._smartg import tdif_up


class TdifUpSampler(SweepSampler):
    """Sample upward diffuse transmittance with Smart-G Monte-Carlo.

    ``tdif_up`` is the diffuse component of the surface-reflected flux
    reaching the satellite (scattered photons, as opposed to the direct
    beam ``tdir_up``).  It contributes to the total upward transmittance
    in the 5S formula::

        t_up = tdir_up + tdif_up

    Produced variable: ``tdif_up``.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters (``aot``, ``rh``, ``h``, ``href``).
    geo_config : GeoConfig
        Viewing geometry (``vza``, ``saa``).
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
    batch_size : int, optional
        Atmospheric states per Smart-G call.
    dedup : bool, optional
        Collapse repeated states before calling.
    """

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = ["tdif_up"]
    contract: ClassVar[str] = (
        "batch(aot, rh, h, href, vza) vec(wl) -> tdif_up(wl)"
    )
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(tdif_up)

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(3e7),
        cache: utils.CacheStore | None = None,
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

    def _get_configs(self) -> tuple[utils.ConfigProtocol, ...]:
        return (self.spectral_config, self.atmo_config, self.geo_config)

    def _statics(self) -> dict[str, Any]:
        return {
            "species": self.atmo_config.species,
            "afgl_type": self.afgl_type,
            "remove_rayleigh": self.remove_rayleigh,
            "n_ph": self.n_ph,
            "saa": float(self.geo_config.saa.values.flat[0]),
        }
