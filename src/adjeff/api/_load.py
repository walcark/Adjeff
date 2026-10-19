"""Products into scenes.

Functions
---------
    load_scene
        See :mod:`adjeff.api`.
    load_maja
        See :mod:`adjeff.api`.
"""

from __future__ import annotations

from pathlib import Path

from adjeff.core import (
    ImageDict,  # noqa: E402
    SensorBand,
)
from adjeff.modules.loaders import MajaLoader, ProductLoader
from adjeff.utils import CacheStore

from ._config import _DEFAULT_SPECIES, _SPECIES_ATTR
from ._pipeline import run_radiatives_from_scene


def load_scene(
    loader: ProductLoader,
    *,
    compute_radiatives: bool = False,
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    cache: CacheStore | None = None,
    dedup: bool = False,
    batch_size: int = 64,
) -> ImageDict:
    """Return the scene *loader* produces, with its aerosol species.

    The species, from *species*, else ``loader.species()``, else
    sulphate, are stored in ``attrs["adjeff:species"]`` of every band
    for :func:`load_config`.  With *compute_radiatives*, the radiative
    terms are added by :func:`run_radiatives_from_scene`.

    The other parameters are those of the module docstring.
    """
    scene = loader.forward()

    _species: dict[str, float] = (
        species
        or (loader.species() if hasattr(loader, "species") else None)
        or _DEFAULT_SPECIES
    )
    for band in scene.bands:
        scene[band].attrs[_SPECIES_ATTR] = _species

    if compute_radiatives:
        scene = run_radiatives_from_scene(
            scene,
            n_bins=n_bins,
            species=_species,
            remove_rayleigh=remove_rayleigh,
            afgl_type=afgl_type,
            cache=cache,
            dedup=dedup,
            batch_size=batch_size,
        )

    return scene


def load_maja(
    product_path: Path,
    bands: list[SensorBand],
    res: float | list[float],
    mnt_path: Path | None = None,
    href: float = 2.0,
    as_map: bool = False,
    cache: CacheStore | None = None,
    compute_radiatives: bool = False,
    n_bins: int | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    dedup: bool = False,
    batch_size: int = 64,
) -> ImageDict:
    """Return :func:`load_scene` of a :class:`~adjeff.modules.loaders.MajaLoader`.

    *product_path*, *res*, *mnt_path* (required), *href* and *as_map*
    are those of the loader; the rest are those of :func:`load_scene`.
    """
    if mnt_path is None:
        from adjeff.exceptions import ConfigurationError

        raise ConfigurationError(
            "load_maja requires mnt_path (path to the DEM folder). "
            "Example: mnt_path=Path('/data/dtm')."
        )
    loader = MajaLoader(
        product_path=product_path,
        bands=bands,
        res=res,
        mnt_path=mnt_path,
        href=href,
        as_map=as_map,
        cache=cache,
    )
    return load_scene(
        loader,
        compute_radiatives=compute_radiatives,
        n_bins=n_bins,
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        cache=cache,
        dedup=dedup,
        batch_size=batch_size,
    )
