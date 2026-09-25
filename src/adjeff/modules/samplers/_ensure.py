"""Dependencies a sampler reuses from the scene rather than recomputing."""

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

    The BRDF samplers divide their raw Monte-Carlo result by these, so
    the draw they use must be the draw the rest of the scene agreed on.
    Recomputing one the scene already holds would leave the numerator
    and the denominator on two independent draws, and whether that
    happens would depend on the cache, since a cached sampler never
    reaches this code at all.  This is the reasoning of
    :func:`~adjeff.modules.samplers.rho_atm.ensure_rho_atm`, applied to
    three quantities instead of one.

    Only ``tdif_down`` is Monte-Carlo; the two direct transmittances are
    analytical, cost nothing and add no noise.  That is why a single
    photon count is enough here.

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
        Photon count for ``tdif_down``.  It sets the noise floor of any
        ratio built on it, so it should match the numerator's budget
        rather than be left to a hard-coded default.
    cache : CacheStore or None
        Cache forwarded to whichever sampler runs.
    rename : dict[str, str] or None, optional
        Slot names the caller reads these variables under.  Both the
        presence test and the sampler that fills a gap use them, so a
        renamed dependency is looked for, and written, where the caller
        expects it.

    Returns
    -------
    ImageDict
        The input scene when every band already holds the three, or the
        enriched scene.
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
