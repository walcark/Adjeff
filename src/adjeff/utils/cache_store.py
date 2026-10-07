"""Disk cache of scene module outputs.

Classes
-------
    CacheStore
        Zarr entries keyed by a content hash, written atomically.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from .._logging import get_logger

if TYPE_CHECKING:
    from adjeff.core import ImageDict, SensorBand

logger = get_logger(__name__)


class CacheStore:
    """Zarr cache of scene module outputs, one store per key and band.

    Parameters
    ----------
    cache_dir : str, Path or None, optional
        Root directory; ``None`` disables caching.
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
        """Write *variables* of every band under *key*, atomically.

        Chunked one value per non-spatial dim, so that reading one combo
        reads one slice.
        """
        if not self.enabled:
            return
        assert self._cache_dir is not None

        for band in scene.bands:
            ds = scene[band]
            # Drop the encoding read back from disk: to_zarr would prefer
            # its chunks to the ones set below, and refuse the mismatch.
            subset = ds[variables].drop_encoding()
            dest = self._cache_dir / key / f"{band}.zarr"
            dest.parent.mkdir(parents=True, exist_ok=True)
            spatial = {"x", "y"}
            chunks = {str(d): 1 if str(d) not in spatial else -1 for d in subset.dims}

            with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
                tmp_path = Path(tmp) / "data.zarr"
                subset.chunk(chunks).to_zarr(tmp_path, mode="w")
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(tmp_path), str(dest))

            logger.debug(
                "cache.write",
                key=key[:8],
                band=str(band),
                vars=variables,
                path=str(dest),
            )

    def load_vars(
        self,
        key: str,
        bands: list[SensorBand],
        variables: list[str],
    ) -> dict[SensorBand, dict[str, xr.DataArray]] | None:
        """Return ``{band: {var: DataArray}}`` lazily, or ``None`` on a miss.

        A missing band, an unreadable store or a missing variable is a miss.
        """
        if not self.enabled or self._cache_dir is None:
            return None

        result: dict[SensorBand, dict[str, xr.DataArray]] = {}
        for band in bands:
            path = self._cache_dir / key / f"{band}.zarr"
            if not path.exists():
                logger.debug("cache.miss", key=key[:8], band=str(band))
                return None
            try:
                ds = xr.open_zarr(path)
            except Exception:
                logger.warning(
                    "cache.unreadable",
                    key=key[:8],
                    band=band,
                    path=str(path),
                )
                return None

            absent = [var for var in variables if var not in ds]
            if absent:
                logger.debug("cache.miss", key=key[:8], band=str(band), missing=absent)
                return None
            result[band] = {var: ds[var] for var in variables}

        logger.debug("cache.hit", key=key[:8], bands=len(bands), vars=variables)
        return result if result else None

    def clear(self) -> None:
        """Remove all cache entries."""
        if self._cache_dir is not None and self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
