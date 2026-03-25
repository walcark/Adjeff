"""Data wrapper for multiple PSF models on multiple bands."""

from __future__ import annotations

from typing import Self

import xarray as xr

from adjeff.exceptions import ConfigurationError

from ._psf import PSFGrid, PSFModule
from .bands import SensorBand
from .image_dict import ImageDict


class PSFDict:
    """Mapping of sensor bands to PSF kernels Datasets.

    All PSFModules must be of the same type (one model type per PSFDict).
    """

    def __init__(self, psfs: list[PSFModule]) -> None:
        """Initialize PSFDict from a list of PSFModules of the same type."""
        psf_types = {type(psf) for psf in psfs}
        if len(psf_types) > 1:
            raise ConfigurationError(
                f"All PSFModules must be of the same type, got: {psf_types}"
            )

        bands = [psf.band for psf in psfs]
        duplicate_bands = {b for b in bands if bands.count(b) > 1}
        if duplicate_bands:
            raise ConfigurationError(
                f"Duplicate bands in PSFDict: {duplicate_bands}"
            )

        self._data: dict[SensorBand, xr.Dataset] = {}
        for psf in psfs:
            da = psf.to_dataarray()
            self._data[psf.band] = xr.Dataset({"kernel": da})

    @classmethod
    def training_input(
        cls,
        scene: ImageDict,
        psf_type: type[PSFModule],
        psf_init_params: dict[str, float],
        psf_grids: dict[SensorBand, PSFGrid],
    ) -> Self:
        """Initialise a PSFDict for a training.

        One PSF type is instantiated for each band in the scene, using the
        same initialization parameters across bands.
        """
        missing_bands = [b for b in scene.bands if b not in psf_grids]
        if missing_bands:
            raise ConfigurationError(
                f"Missing bands grid specification for bands: {missing_bands}"
            )

        psfs = [
            psf_type(grid=psf_grids[band], band=band, **psf_init_params)  # type: ignore
            for band in scene.bands
        ]

        return cls(psfs=psfs)

    @property
    def bands(self) -> list[SensorBand]:
        """Sorted list of band identifiers present in this PSFDict."""
        return sorted(self._data.keys(), key=lambda b: b.value)

    def __getitem__(self, band: SensorBand) -> xr.Dataset:
        """Return the Dataset for *band*."""
        return self._data[band]

    def __setitem__(self, band: SensorBand, ds: xr.Dataset) -> None:
        """Store a Dataset under the *band* key."""
        self._data[band] = ds

    def __contains__(self, band: object) -> bool:
        """Check if *band* is stored in the PSFDict."""
        return band in self._data

    def __repr__(self) -> str:
        """Return a string representation of the PSFDict."""
        var_sets = {
            band: sorted(str(dv) for dv in ds.data_vars.keys())
            for band, ds in self._data.items()
        }
        vars_summary = "; ".join(
            f"{bid}: [{', '.join(vs)}]" for bid, vs in var_sets.items()
        )
        band_summary = ", ".join(str(b) for b in self.bands)
        return f"PSFDict(bands=[{band_summary}], vars={{{vars_summary}}})"

    def to_dataarray(self, band: SensorBand) -> xr.DataArray:
        """Return the kernel for the band of interest."""
        return self._data[band]["kernel"]

    def kernel(self, band: SensorBand) -> xr.DataArray:
        """Alias for to_dataarray()."""
        return self.to_dataarray(band)
