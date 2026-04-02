"""Convenience pipeline that computes all radiative quantities in sequence."""

from typing import Any

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.utils import CacheStore

from ..pipeline import Pipeline
from .rho_atm import SmartgSampler_Rho_atm
from .sph_alb import SmartgSampler_Sph_alb
from .tdif_down import SmartgSampler_Tdif_down
from .tdif_up import SmartgSampler_Tdif_up
from .tdir_down import SmartgSampler_Tdir_down
from .tdir_up import SmartgSampler_Tdir_up


class RadiativePipeline(Pipeline):
    """Pipeline that computes radiative parameters."""

    def __init__(
        self,
        atmo_config: AtmoConfig,
        geo_config: GeoConfig,
        spectral_config: SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph_sph_alb: int = int(2e7),
        n_ph_rho_atm: int = int(2e7),
        n_ph_tdif_up: int = int(3e7),
        n_ph_tdif_down: int = int(3e7),
        cache: CacheStore | None = None,
        chunks: dict[str, int] | None = None,
        deduplicate_dims: list[str] | None = None,
    ) -> None:
        common: dict[str, Any] = dict(
            atmo_config=atmo_config,
            geo_config=geo_config,
            spectral_config=spectral_config,
            remove_rayleigh=remove_rayleigh,
            afgl_type=afgl_type,
            cache=cache,
            chunks=chunks,
            deduplicate_dims=deduplicate_dims,
        )
        super().__init__(
            [
                SmartgSampler_Tdir_down(**common),
                SmartgSampler_Tdir_up(**common),
                SmartgSampler_Sph_alb(
                    atmo_config=atmo_config,
                    spectral_config=spectral_config,
                    remove_rayleigh=remove_rayleigh,
                    afgl_type=afgl_type,
                    n_ph=n_ph_sph_alb,
                    cache=cache,
                    chunks=chunks,
                    deduplicate_dims=deduplicate_dims,
                ),
                SmartgSampler_Tdif_up(**common, n_ph=n_ph_tdif_up),
                SmartgSampler_Tdif_down(**common, n_ph=n_ph_tdif_down),
                SmartgSampler_Rho_atm(**common, n_ph=n_ph_rho_atm),
            ]
        )
