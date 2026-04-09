"""Data wrapper for multiple PSF models on multiple bands."""

from __future__ import annotations

from typing import Any, Self

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
    def from_kernels(
        cls,
        kernels: dict["SensorBand", xr.DataArray],
        params: dict["SensorBand", dict[str, xr.DataArray]] | None = None,
    ) -> "PSFDict":
        """Build a PSFDict directly from pre-built kernel DataArrays.

        Unlike the standard constructor (which takes PSFModules), this
        accepts DataArrays that may carry extra dimensions produced by
        stacking per-combo optimisation results (e.g. ``aot``, ``rh``).

        Per-combo parameter values can be supplied via *params* and are
        stored as ``param_<name>`` variables alongside ``kernel`` in each
        band Dataset.  Retrieve them afterwards with :meth:`params`.

        Parameters
        ----------
        kernels : dict[SensorBand, xr.DataArray]
            Mapping from band to kernel DataArray (dims at minimum
            ``y_psf`` and ``x_psf``; any extra dims are preserved).
        params : dict[SensorBand, dict[str, xr.DataArray]] or None
            Optional per-band parameter DataArrays produced by the
            optimiser (e.g. ``{B02: {"sigma": DataArray(aot, rh)}}``.

        Returns
        -------
        PSFDict
        """
        obj: PSFDict = cls.__new__(cls)
        obj._data = {}
        for band, da in kernels.items():
            ds_vars: dict[str, xr.DataArray] = {"kernel": da}
            if params and band in params:
                for pname, pda in params[band].items():
                    ds_vars[f"param_{pname}"] = pda
            obj._data[band] = xr.Dataset(ds_vars)
        return obj

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

    def params(
        self, band: SensorBand
    ) -> dict[str, xr.DataArray | float] | None:
        """Return the PSF parameter values for *band*.

        Two storage strategies are supported:

        - **Multi-combo** (post-optimisation): parameters were stored as
          ``param_<name>`` variables in the Dataset by :meth:`from_kernels`.
          Returns ``{name: DataArray}`` where each DataArray carries the
          combo dimensions (e.g. ``aot``, ``rh``).

        - **Single-combo** (direct from PSFModule): parameters are read
          from ``kernel.attrs["adjeff:params"]``.
          Returns ``{name: float}``.

        Returns ``None`` when no parameter information is available
        (e.g. :class:`~adjeff.core.non_analytical_psf.NonAnalyticalPSF`).

        Parameters
        ----------
        band : SensorBand
            Band of interest.

        Returns
        -------
        dict[str, xr.DataArray | float] or None
        """
        ds = self._data[band]
        param_vars: dict[str, xr.DataArray | float] = {
            k[6:]: ds[k]
            for k in ds.data_vars
            if isinstance(k, str) and k.startswith("param_")
        }
        if param_vars:
            return param_vars
        return ds["kernel"].attrs.get("adjeff:params")

    def to_dataarray(self, band: SensorBand) -> xr.DataArray:
        """Return the kernel for the band of interest."""
        return self._data[band]["kernel"]

    def kernel(self, band: SensorBand) -> xr.DataArray:
        """Alias for to_dataarray()."""
        return self.to_dataarray(band)

    def _cache_dict(self) -> dict[str, Any]:
        """Return a joblib-hashable representation of all kernel data.

        xr.DataArray values are converted to nested lists so that joblib
        can hash them.  Used by SceneModule._config_dict().
        """
        result: dict[str, Any] = {}
        for band, ds in self._data.items():
            result[str(band)] = {
                var: da.values for var, da in ds.data_vars.items()
            }
        return result
