"""Multi-band scene container.

Classes
-------
    ImageDict
        One ``xr.Dataset`` per sensor band, each on its own grid.
"""

from __future__ import annotations

from typing import Hashable

import xarray as xr

from adjeff.exceptions import MissingVariableError

from .._logging import get_logger
from .bands import SensorBand

logger = get_logger(__name__)


class ImageDict:
    """One ``xr.Dataset`` per sensor band.

    Bands may differ in resolution, and datasets may carry extra dimensions
    (``aot``, ``wl``, ...). Scene modules add variables to them along a
    pipeline.
    """

    def __init__(self, band_datasets: dict[SensorBand, xr.Dataset]) -> None:
        self._data: dict[SensorBand, xr.Dataset] = dict(band_datasets)

    @property
    def bands(self) -> list[SensorBand]:
        """Sorted list of band identifiers (B02 < B03 < etc.)."""
        return sorted(self._data.keys(), key=lambda b: b.value)

    def variables(self, band: SensorBand) -> list[Hashable]:
        """Return the DataArray variable names present in *band*'s Dataset."""
        return list(self._data[band].data_vars)

    def require_vars(self, vars: list[str]) -> None:
        """Raise an exception if any var is absent from any band Dataset.

        Raises
        ------
        MissingVariableError
            If *var* is missing from at least one band Dataset.
        """
        for var in vars:
            missing_bands = [
                bid for bid, ds in self._data.items() if var not in ds.data_vars
            ]
            if missing_bands:
                raise MissingVariableError(
                    f"Var {var!r} is missing from band(s): {missing_bands}"
                )

    def shallow_copy(self) -> "ImageDict":
        """Return a copy whose new variables do not reach the original.

        DataArrays are shared, not copied, and dask graphs are kept.
        """
        return ImageDict({band: ds.copy(deep=False) for band, ds in self._data.items()})

    def __getitem__(self, band: SensorBand) -> xr.Dataset:
        """Return the Dataset for *band*."""
        return self._data[band]

    def __setitem__(self, band: SensorBand, ds: xr.Dataset) -> None:
        """Store a Dataset under the *band* key."""
        self._data[band] = ds

    def __contains__(self, band: object) -> bool:
        """Check if *band* is stored in the ImageDict."""
        return band in self._data

    def __repr__(self) -> str:
        """Return a string representation of the ImageDict."""
        parts = []
        for band in self.bands:
            var_names = list(self._data[band].data_vars)
            parts.append(f"  {band!r}: {var_names}")
        inner = "\n".join(parts)
        return f"ImageDict(\n{inner}\n)" if parts else "ImageDict({})"
