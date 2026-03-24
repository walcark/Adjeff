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


# TODO: if might be interesting to enable cache to return the bands
# TODO: where all variables were found, and adapte _compute in Module
# TOOD: to work only on those bands.


class CacheStore:
    """Zarr-backed content-hash cache for a SceneModule output.

    Cache entries are stored as Zarr stored. The key is produced by hashing
    the module configuration + the input provenance.

    The writing is first performed in a temporary file, then moved to the
    actual cache location. This allows to avoid file corruption when writing
    to disk.

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

        This method only saves *variables*, and thus allows to solely save the
        data produced by the module.

        Write are atomics: data is writen to a temporary directory then renamed
        to the final path to prevent partial-write computation.

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

        for band_id in scene.band_ids:
            ds = scene[band_id]
            subset = ds[variables]
            dest = self._cache_dir / key / f"{band_id}.zarr"
            dest.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
                tmp_path = Path(tmp) / "data.zarr"
                subset.to_zarr(tmp_path, mode="w")
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(tmp_path), str(dest))

            logger.debug(
                "Scene was saved to cache.",
                key=key[:8],
                band=band_id,
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
        band_ids : list[str]
            Band identifiers to load.
        variables : list[str]
            Variable names to retrieve.

        Returns
        -------
        dict or None
            ``{band_id: {var: DataArray}}`` on hit, ``None`` on miss.
            ``_atcor_provenance`` attributes are restored alongside DataArrays
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
                result[band] = {var: ds[var] for var in variables if var in ds}
            except Exception:
                logger.warning(
                    "failed to load from cache",
                    key=key[:8],
                    band=band,
                    path=str(path),
                )
                return None

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
            Class name of the SceneModule (e.g. ``"SmartGSimulation"``).
        """
        if self._cache_dir is None:
            return
        target = self._cache_dir / module_name
        if target.exists():
            shutil.rmtree(target)
