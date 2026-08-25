"""Diffuse downward transmittance (``tdif_down``) sampler using Smart-G."""

from typing import Any, Callable, ClassVar

import adjeff.atmosphere as atmo
from adjeff.utils import CacheStore
from adjeff.utils._config import ConfigProtocol

from ..sweep_sampler import SweepSampler
from ._smartg import tdif_down


class TdifDownSampler(SweepSampler):
    """Sample downward diffuse transmittance with Smart-G Monte-Carlo.

    ``tdif_down`` is the diffuse fraction of the solar flux reaching the
    surface (scattered photons, as opposed to the direct beam
    ``tdir_down``).  It contributes to the total downward transmittance
    in the 5S formula::

        t_down = tdir_down + tdif_down

    Produced variable: ``tdif_down``.

    Notes
    -----
    Requires a CUDA-capable GPU.

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
    batch_size : int, optional
        Atmospheric states per Smart-G call.
    dedup : bool, optional
        Collapse repeated states before calling.
    """

    _required_vars: ClassVar[list[str]] = []
    _output_vars: ClassVar[list[str]] = ["tdif_down"]
    contract: ClassVar[str] = "batch(aot, rh, h, href, sza) vec(wl) -> tdif_down(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(tdif_down)

    def __init__(
        self,
        atmo_config: atmo.AtmoConfig,
        geo_config: atmo.GeoConfig,
        spectral_config: atmo.SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph: int = int(3e7),
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
            "saa": float(self.geo_config.saa.values.flat[0]),
            "sat_height": self.geo_config.sat_height,
        }
