"""Data wrapper for multiple images on multiple bands."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Hashable

import numpy as np
import structlog
import xarray as xr

from adjeff.exceptions import MissingVariableError

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

    def write_to_directory(
        self,
        directory: str | Path,
        var: str,
    ) -> list[Path]:
        """Write *var* arrays to *directory* as .npy files.

        All non-spatial dimensions are encoded in the filename. For instance
        ``rho_s__aot=0.4__rh=50.0__B02.npy`` if ``aot`` and ``rh`` are present
        as dimensions. If there are no extra (non-spatial) dims, the name will
        have the form ``{var}__{band}.npy``

        Parameters
        ----------
        directory : str | Path
            Target directory to write the variable images.
        var : str
            Name of the internal variable to store.

        Returns
        -------
        list[Path]
            A list of written Path objects.
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []

        for band in self.bands:
            ds = self._data[band]
            if var not in ds.data_vars:
                raise MissingVariableError(
                    f"Variable {var!r} is missing from band {band!r}"
                )

            da: xr.DataArray = ds[var]

            # Identify non-spatial dimensions (everything except "x" and "y")
            extra_dims = [d for d in da.dims if d not in ("x", "y")]

            if not extra_dims:
                # No extra dims — write a single file
                filename = f"{var}__{band}.npy"
                path = out_dir / filename
                np.save(path, da.values)
                written.append(path)
            else:
                # Iterate over the Cartesian product of extra dim coordinates
                coord_values = [da.coords[d].values for d in extra_dims]
                for combo in product(*coord_values):
                    # Build selector and filename suffix
                    selector = dict(zip(extra_dims, combo))
                    suffix_parts = "__".join(
                        f"{dim}={val}" for dim, val in zip(extra_dims, combo)
                    )
                    filename = f"{var}__{suffix_parts}__{band}.npy"
                    path = out_dir / filename
                    slice_da = da.sel(selector)
                    np.save(path, slice_da.values)
                    written.append(path)

            logger.debug(
                "Saved DataArray to .npy file",
                var=var,
                band=band,
                path=str(directory),
            )

        return written

    @property
    def bands(self) -> list[SensorBand]:
        """Sorted list of band identifiers (B02 < B03 < etc.)."""
        return sorted(self._data.keys(), key=lambda b: b.value)

    def variables(self, band: SensorBand) -> list[Hashable]:
        """Return the DataArray variable names present in *band*'s Dataset."""
        return list(self._data[band].data_vars)

    def has_var(self, var: str) -> bool:
        """Return True if *all* band Datasets contain *var*."""
        return all(var in ds.data_vars for ds in self._data.values())

    def require_vars(self, vars: list[str]) -> None:
        """Raise an exception if any var is absent from any band Dataset."""
        for var in vars:
            missing_bands = [
                bid
                for bid, ds in self._data.items()
                if var not in ds.data_vars
            ]
            if missing_bands:
                raise MissingVariableError(
                    f"Var {var!r} is missing from band(s): {missing_bands}"
                )

    def shallow_copy(self) -> "ImageDict":
        """Return a new ImageDict sharing the same DataArrays by reference.

        Each band Dataset is shallow-copied so that new variable assignments
        on the copy do not affect the original. Existing DataArrays are shared
        in memory and dask graphs are preserved (no compute triggered).
        """
        return ImageDict(
            {band: ds.copy(deep=False) for band, ds in self._data.items()}
        )

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
        for band in self.bands:
            var_names = list(self._data[band].data_vars)
            parts.append(f"  {band!r}: {var_names}")
        inner = "\n".join(parts)
        return f"ImageDict(\n{inner}\n)" if parts else "ImageDict({})"
