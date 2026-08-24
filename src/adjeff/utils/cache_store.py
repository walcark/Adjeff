"""Disk cache for expensive module computations."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import xarray as xr

if TYPE_CHECKING:
    from adjeff.core import ImageDict, SensorBand

logger = structlog.get_logger(__name__)


class CacheStore:
    """Zarr-backed content-hash cache for a SceneModule output.

    Cache entries are stored as Zarr stores keyed by a content hash of
    the module configuration and the input provenance.

    Writes are atomic: data is written to a temporary directory then
    renamed to the final path to prevent partial-write corruption.

    Parameters
    ----------
    cache_dir:
        Root directory for cache storage.  Pass ``None`` to disable caching
        (all operations become no-ops).
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    @property
    def enabled(self) -> bool:
        """Return True when a cache directory is configured."""
        return self._cache_dir is not None

    @property
    def cache_dir(self) -> Path | None:
        """Return the root cache directory, or None if caching is disabled."""
        return self._cache_dir

    def save_vars(
        self,
        key: str,
        scene: "ImageDict",
        variables: list[str],
    ) -> None:
        """Save *variables* DataArrays for each band to Zarr under *key*.

        Only *variables* are saved, allowing the caller to persist only
        the data produced by the module.

        Writes are atomic: data is written to a temporary directory then
        renamed to the final path to prevent partial-write corruption.

        Parameters
        ----------
        key : str
            Content hash identifying this cache entry.
        scene : ImageDict
            ImageDict whose band Datasets are the data source.
        variables : list[str]
            Variable names to persist.

        """
        if not self.enabled:
            return
        assert self._cache_dir is not None

        for band in scene.bands:
            ds = scene[band]
            subset = ds[variables]
            dest = self._cache_dir / key / f"{band}.zarr"
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Chunk size 1 along every non-spatial dimension so that
            # selecting a single atmospheric combo (aot, rh, …) reads only
            # the required slice rather than the full array.
            spatial = {"x", "y"}
            chunks = {
                str(d): 1 if str(d) not in spatial else -1 for d in subset.dims
            }

            with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
                tmp_path = Path(tmp) / "data.zarr"
                subset.chunk(chunks).to_zarr(tmp_path, mode="w")
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(tmp_path), str(dest))

            logger.debug(
                "Scene was saved to cache.",
                key=key[:8],
                band=band,
                vars=variables,
                path=str(dest),
            )

    def load_vars(
        self,
        key: str,
        bands: list[SensorBand],
        variables: list[str],
    ) -> dict[SensorBand, dict[str, xr.DataArray]] | None:
        """Return cached DataArrays or None on cache miss.

        Band ids must be provided because the cache doesn't know the bands
        for which the object was saved. Returns None if any var or any band
        is missing.

        Parameters
        ----------
        key : str
            Content hash to look up.
        bands : list[str]
            Band identifiers to load.
        variables : list[str]
            Variable names to retrieve.

        Returns
        -------
        dict or None
            ``{band: {var: DataArray}}`` on hit, ``None`` on miss.
            ``_adjeff_provenance`` attributes are restored alongside DataArrays
            to preserve the provenance chain.

        """
        if not self.enabled or self._cache_dir is None:
            return None

        result: dict[SensorBand, dict[str, xr.DataArray]] = {}
        for band in bands:
            path = self._cache_dir / key / f"{band}.zarr"
            if not path.exists():
                logger.debug("cache miss", key=key[:8], band=band)
                return None
            try:
                ds = xr.open_zarr(path)
            except Exception:
                logger.warning(
                    "failed to load from cache",
                    key=key[:8],
                    band=band,
                    path=str(path),
                )
                return None

            # A truncated entry (interrupted write, output_vars changed
            # since it was written) must read as a miss.  Returning the
            # variables that happen to be there would hand the caller a
            # scene silently short of an output, flagged as a cache hit.
            absent = [var for var in variables if var not in ds]
            if absent:
                logger.debug(
                    "cache miss", key=key[:8], band=band, missing=absent
                )
                return None
            result[band] = {var: ds[var] for var in variables}

        logger.debug(
            "Cache was hit.", key=key[:8], bands=bands, vars=variables
        )
        return result if result else None

    def clear(self) -> None:
        """Remove all cache entries."""
        if self._cache_dir is not None and self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)

    def clear_function(self, module_name: str) -> None:
        """Remove cache entries for a specific module.

        Parameters
        ----------
        module_name:
            Class name of the SceneModule (e.g. ``"RhoAtmSampler"``).
        """
        if self._cache_dir is None:
            return
        target = self._cache_dir / module_name
        if target.exists():
            shutil.rmtree(target)
