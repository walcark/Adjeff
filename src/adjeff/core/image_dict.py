"""Data wrapper for multiple images on multiple bands."""

from __future__ import annotations

import structlog
import xarray as xr

from .bands import SensorBand

logger = structlog.get_logger(__name__)


class ImageDict:
    """Wrapper around dict[str, xr.Dataset], one dataset per band.

    This wrapper allows to store multiple bands that have multiple resolutions.
    The Datasets are progressively enriche by SceneModules. The Datasets may
    carry extra parameter dimensions (aot, wl, etc.) for atmospheric config
    with multiple values.
    """

    def __init__(self, band_datasets: dict[SensorBand, xr.Dataset]) -> None:
        self._data: dict[SensorBand, xr.Dataset] = dict(band_datasets)

    @property
    def band_ids(self) -> list[SensorBand]:
        """Sorted list of band identifiers (B02 < B03 < etc.)."""
        return sorted(self._data.keys(), key=lambda b: b.value)

    def __getitem__(self, band: SensorBand) -> xr.Dataset:
        """Return a wavelength band."""
        return self._data[band]

    def __setitem__(self, band: SensorBand, ds: xr.Dataset) -> None:
        """Store a Dataset under the *band* key."""
        self._data[band] = ds

    def __contains__(self, band: object) -> bool:
        """Check if *band* is store in the ImageDict."""
        return band in self._data

    def __repr__(self) -> str:
        """Return a string representation of the ImageDict."""
        parts = []
        for bid in self.band_ids:
            var_names = list(self._data[bid].data_vars)
            parts.append(f"  {bid!r}: {var_names}")
        inner = "\n".join(parts)
        return f"ImageDict(\n{inner}\n)" if parts else "ImageDict({})"
