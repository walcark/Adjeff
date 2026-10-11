"""Terms a sampler reads from the scene or computes when missing.

Functions
---------
    ensure_downward
        ``tdir_down`` and ``tdif_down`` of the scene, sampled if absent.
"""

from __future__ import annotations

import adjeff.atmosphere as atmo
from adjeff.core import ImageDict
from adjeff.utils import CacheStore

from ..sweep_sampler import SweepSampler
from .tdif_down import TdifDownSampler
from .tdir_down import TdirDownSampler
from .tdir_up import TdirUpSampler


def ensure_downward(
    scene: ImageDict,
    *,
    atmo_config: atmo.AtmoConfig,
    geo_config: atmo.GeoConfig,
    spectral_config: atmo.SpectralConfig,
    remove_rayleigh: bool,
    afgl_type: str,
    n_ph_tdif_down: int,
    cache: CacheStore | None,
    rename: dict[str, str] | None = None,
) -> ImageDict:
    """Return *scene* carrying ``tdir_down``, ``tdif_down`` and ``tdir_up``.

    Only the ones a band is missing are computed, so the numerator and
    the denominator of a BRDF ratio stay on one Monte-Carlo draw.

    Parameters
    ----------
    scene : ImageDict
        Scene that may already carry some of the three variables.
    atmo_config, geo_config, spectral_config : AtmoConfig, GeoConfig, SpectralConfig
        Configuration used for whichever variable has to be computed.
    remove_rayleigh : bool
        Suppress Rayleigh scattering.
    afgl_type : str
        AFGL atmosphere profile identifier.
    n_ph_tdif_down : int
        Photon count for ``tdif_down``, the only Monte-Carlo one of the
        three.  It sets the noise floor of any ratio built on it.
    cache : CacheStore or None
        Cache forwarded to whichever sampler runs.
    rename : dict[str, str] or None, optional
        Slot names the caller reads these variables under.

    Returns
    -------
    ImageDict
        The input scene, or a copy carrying what was missing.
    """

    def build(name: str) -> SweepSampler:
        """Return the sampler that produces *name*."""
        if name == "tdir_down":
            return TdirDownSampler(
                atmo_config=atmo_config,
                geo_config=geo_config,
                spectral_config=spectral_config,
                remove_rayleigh=remove_rayleigh,
                afgl_type=afgl_type,
                cache=cache,
                rename=rename,
            )
        if name == "tdir_up":
            return TdirUpSampler(
                atmo_config=atmo_config,
                geo_config=geo_config,
                spectral_config=spectral_config,
                remove_rayleigh=remove_rayleigh,
                afgl_type=afgl_type,
                cache=cache,
                rename=rename,
            )
        return TdifDownSampler(
            atmo_config=atmo_config,
            geo_config=geo_config,
            spectral_config=spectral_config,
            remove_rayleigh=remove_rayleigh,
            afgl_type=afgl_type,
            n_ph=n_ph_tdif_down,
            cache=cache,
            rename=rename,
        )

    slots = dict(rename or {})
    for name in ("tdir_down", "tdir_up", "tdif_down"):
        slot = slots.get(name, name)
        # An empty scene carries nothing, and `all` over no band is True.
        if not scene.bands or not all(slot in scene[band] for band in scene.bands):
            scene = build(name)(scene)
    return scene
