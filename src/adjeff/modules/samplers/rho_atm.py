"""Atmospheric path reflectance (``rho_atm``) sampler using Smart-G."""

from typing import Any, Callable, ClassVar

import adjeff.atmosphere as atmo
from adjeff.core import ImageDict
from adjeff.utils import CacheStore

from ._atmo_sampler import AtmoSampler
from ._smartg import rho_atm


class RhoAtmSampler(AtmoSampler):
    """Sample atmospheric path reflectance with Smart-G Monte-Carlo.

    ``rho_atm`` is the TOA reflectance contribution of the atmosphere
    when the surface is perfectly absorbing (no surface term).  It
    appears as the additive offset in the 5S formula::

        rho_toa = rho_atm
                  + t_up * t_down * rho_unif / (1 - sph_alb * rho_unif)

    All atmospheric and geometric parameters are swept as
    ``vector_dims`` — a single Smart-G call handles all wavelengths,
    aerosol states, and viewing angles simultaneously.

    Produced variable: ``rho_atm``.

    Notes
    -----
    Requires a CUDA-capable GPU.

    Parameters are those of :class:`AtmoSampler`.
    """

    _output_vars: ClassVar[list[str]] = ["rho_atm"]
    contract: ClassVar[str] = "batch(aot, rh, h, href, vza, sza) vec(wl) -> rho_atm(wl)"
    point_fn: ClassVar[Callable[..., Any]] = staticmethod(rho_atm)
    default_n_ph: ClassVar[int] = 20000000
    geo_statics: ClassVar[tuple[str, ...]] = ("saa", "vaa", "sat_height")


def ensure_rho_atm(
    scene: ImageDict,
    *,
    atmo_config: atmo.AtmoConfig,
    geo_config: atmo.GeoConfig,
    remove_rayleigh: bool,
    afgl_type: str,
    n_ph: int,
    cache: CacheStore | None,
) -> ImageDict:
    """Return *scene* carrying ``rho_atm``, computing it only if absent.

    The path reflectance is a single Monte-Carlo number, and two draws of
    it differ.  A sampler that recomputes one the scene already holds
    therefore replaces the value every downstream module has agreed on:
    ``rho_toa`` would then be built with one draw and inverted with
    another, leaving a constant bias behind.  Worse, whether that happens
    depends on the cache, since a cached sampler never reaches this code
    at all.

    Parameters
    ----------
    scene : ImageDict
        Scene that may already carry ``rho_atm``.
    atmo_config, geo_config : AtmoConfig, GeoConfig
        Atmosphere and geometry used when it has to be computed.
    remove_rayleigh : bool
        Suppress Rayleigh scattering.
    afgl_type : str
        AFGL atmosphere profile identifier.
    n_ph : int
        Photon count for the computation.
    cache : CacheStore or None
        Cache forwarded to the sampler.

    Returns
    -------
    ImageDict
        The input scene when every band already holds ``rho_atm``,
        otherwise the enriched scene.
    """
    if all("rho_atm" in scene[band] for band in scene.bands):
        return scene
    return RhoAtmSampler(
        atmo_config=atmo_config,
        geo_config=geo_config,
        spectral_config=atmo.SpectralConfig.from_bands(scene.bands),
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        n_ph=n_ph,
        cache=cache,
    )(scene)
