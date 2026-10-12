"""The forward chain, from a surface reflectance to the scene.

Functions
---------
    SceneT
        See :mod:`adjeff.api`.
    run_forward_pipeline
        See :mod:`adjeff.api`.
    run_radiatives_from_scene
        See :mod:`adjeff.api`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import (
    ImageDict,  # noqa: E402
)
from adjeff.modules import Pipeline
from adjeff.modules.classic import Toa2Unif
from adjeff.modules.samplers import RadiativePipeline, RhoToaSymSampler
from adjeff.utils import CacheStore

from ._config import load_config

#: One scene or a batch of them.  Constrained rather than bound, so that
#: a function taking a list is known to return a list.
SceneT = TypeVar("SceneT", ImageDict, list[ImageDict])


def _run_each(
    scene: ImageDict | list[ImageDict], run: Callable[[ImageDict], ImageDict]
) -> ImageDict | list[ImageDict]:
    """Apply *run* to one scene, or to each scene of a batch."""
    if isinstance(scene, list):
        return [run(s) for s in scene]
    return run(scene)


def _map_scenes(scene: SceneT, run: Callable[[ImageDict], ImageDict]) -> SceneT:
    """Apply *run* to *scene*, returning one scene or a list as given."""
    return cast(SceneT, _run_each(scene, run))


def run_forward_pipeline(
    scene: SceneT,
    atmo_config: AtmoConfig,
    geo_config: GeoConfig,
    spectral_config: SpectralConfig,
    cache: CacheStore | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    nr: int = 500,
    n_ph: int = int(1e5),
    batch_size: int = 64,
    rtls: tuple[float, float, float] | None = None,
    stream_dims: dict[str, int] | None = None,
) -> SceneT:
    """Add the radiative terms, ``rho_toa`` and ``rho_unif`` to *scene*.

    Chains :class:`~adjeff.modules.samplers.RadiativePipeline`,
    :class:`~adjeff.modules.samplers.RhoToaSymSampler` and
    :class:`~adjeff.modules.classic.Toa2Unif`; a list of scenes reuses
    the same modules.  Typically called as
    ``run_forward_pipeline(scene, **make_full_config(...))``.

    Parameters
    ----------
    scene : ImageDict or list[ImageDict]
        Scene(s) holding ``rho_s``.
    nr, n_ph : int, optional
        Radii and photons per sensor of ``rho_toa``.
    batch_size : int, optional
        Atmospheric states per Smart-G call; changes the cost only.
    rtls : tuple[float, float, float] or None, optional
        RTLS weights ``(k0, k1p, k2p)`` of the radiative terms; ``None``
        keeps them Lambertian.  ``rho_toa`` is unaffected.
    stream_dims : dict[str, int] or None, optional
        Run the chain on chunks of these dims, e.g. ``{"aot": 3}``.

    The other parameters are those of the module docstring.
    """
    radiative = RadiativePipeline(
        atmo_config=atmo_config,
        geo_config=geo_config,
        spectral_config=spectral_config,
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        cache=cache,
        batch_size=batch_size,
        rtls=rtls,
    )
    rho_toa = RhoToaSymSampler(
        atmo_config=atmo_config,
        geo_config=geo_config,
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        cache=cache,
        nr=nr,
        n_ph=n_ph,
    )
    pipeline = Pipeline(
        [radiative, rho_toa, Toa2Unif()],  # type: ignore[list-item]
        stream_dims=stream_dims,
    )

    out = _map_scenes(scene, pipeline)
    # `saa` and `vaa` are statics of the samplers, so they never reach the
    # output on their own.  Recorded here so the scene describes the
    # geometry it was simulated at, which is what `load_config` reads back.
    return _map_scenes(
        out,
        lambda one: _with_azimuths(one, geo_config),
    )


def run_radiatives_from_scene(
    scene: SceneT,
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    cache: CacheStore | None = None,
    dedup: bool = False,
) -> SceneT:
    """Add the six radiative terms to *scene*, configured from its own fields.

    The atmosphere and geometry are read from each band by
    :func:`load_config`, and a pipeline is run per band, since the
    viewing angles differ between bands.  *scene* may be a list.

    The other parameters are those of the module docstring.
    """

    def _run(s: ImageDict) -> ImageDict:
        s = s.shallow_copy()
        for band in s.bands:
            scene_band = ImageDict({band: s[band]})
            config = load_config(scene_band, band, n_bins=n_bins, species=species)
            radiative = RadiativePipeline(
                atmo_config=config["atmo_config"],
                geo_config=config["geo_config"],
                spectral_config=config["spectral_config"],
                remove_rayleigh=remove_rayleigh,
                afgl_type=afgl_type,
                cache=cache,
                dedup=dedup,
            )
            scene_band = radiative(scene_band)
            s[band] = scene_band[band]
        return s

    return _map_scenes(scene, _run)


def _with_azimuths(scene: ImageDict, geo_config: GeoConfig) -> ImageDict:
    """Return *scene* carrying the azimuths its samplers took as constants.

    Parameters
    ----------
    scene : ImageDict
        A scene the radiative pipeline has just written into.
    geo_config : GeoConfig
        The geometry it was run at.

    Returns
    -------
    ImageDict
        The same scene, with ``saa`` and ``vaa`` as scalar coordinates.
    """
    out = scene.shallow_copy()
    azimuths = {
        name: float(np.asarray(getattr(geo_config, name)).ravel()[0])
        for name in ("saa", "vaa")
    }
    for band in out.bands:
        out[band] = out[band].assign_coords(azimuths)
    return out
