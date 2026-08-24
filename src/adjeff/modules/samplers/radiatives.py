"""Convenience pipeline that computes all six radiative quantities."""

from typing import Any

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.utils import CacheStore

from ..pipeline import Pipeline
from .rho_atm import RhoAtmSampler
from .sph_alb import SphAlbSampler
from .tdif_down import TdifDownSampler
from .tdif_up import TdifUpSampler
from .tdir_down import TdirDownSampler
from .tdir_up import TdirUpSampler


class RadiativePipeline(Pipeline):
    """Pipeline that computes all six radiative quantities in sequence.

    Chains six :class:`~adjeff.modules.SweepSampler` samplers that
    produce the variables required by the 5S formula in order:

    ``tdir_down`` → ``tdir_up`` → ``sph_alb`` → ``tdif_up`` →
    ``tdif_down`` → ``rho_atm``

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters
    ----------
    atmo_config : AtmoConfig
        Atmospheric state parameters.
    geo_config : GeoConfig
        Viewing/illumination geometry.
    spectral_config : SpectralConfig
        Spectral bands and wavelengths to compute.
    remove_rayleigh : bool
        If ``True``, Rayleigh scattering is suppressed in all modules.
    afgl_type : str, optional
        AFGL standard atmosphere profile identifier,
        by default ``"afgl_exp_h8km"``.
    n_ph_sph_alb : int, optional
        Photons per call for the spherical albedo module,
        by default ``2e7``.
    n_ph_rho_atm : int, optional
        Photons per call for the atmospheric reflectance module,
        by default ``2e7``.
    n_ph_tdif_up : int, optional
        Photons per call for the upward diffuse transmittance module,
        by default ``3e7``.
    n_ph_tdif_down : int, optional
        Photons per call for the downward diffuse transmittance module,
        by default ``3e7``.
    cache : CacheStore or None, optional
        Shared result cache forwarded to all modules; ``None`` disables
        caching.
    batch_size : int, optional
        Atmospheric states per Smart-G call, forwarded to all modules.
    dedup : bool, optional
        Collapse repeated states before calling, forwarded to all modules.
    """

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
        batch_size: int = 64,
        dedup: bool = False,
    ) -> None:
        common: dict[str, Any] = dict(
            atmo_config=atmo_config,
            geo_config=geo_config,
            spectral_config=spectral_config,
            remove_rayleigh=remove_rayleigh,
            afgl_type=afgl_type,
            cache=cache,
            batch_size=batch_size,
            dedup=dedup,
        )
        super().__init__(
            [
                TdirDownSampler(**common),
                TdirUpSampler(**common),
                SphAlbSampler(
                    atmo_config=atmo_config,
                    spectral_config=spectral_config,
                    remove_rayleigh=remove_rayleigh,
                    afgl_type=afgl_type,
                    n_ph=n_ph_sph_alb,
                    cache=cache,
                    batch_size=batch_size,
                    dedup=dedup,
                ),
                TdifUpSampler(**common, n_ph=n_ph_tdif_up),
                TdifDownSampler(**common, n_ph=n_ph_tdif_down),
                RhoAtmSampler(**common, n_ph=n_ph_rho_atm),
            ]
        )
