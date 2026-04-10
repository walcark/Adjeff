"""Data wrapper for multiple PSF models on multiple bands."""

from __future__ import annotations

from typing import Any, cast

import xarray as xr

from ._psf import PSFGrid, PSFModule
from .bands import SensorBand


class PSFDict:
    """Mapping of sensor bands to PSF kernels Datasets.

    A PSFDict is either *trainable* (holds live :class:`PSFModule` objects
    with gradient-tracked parameters) or *frozen* (holds pre-computed
    :class:`xr.DataArray` kernels).  Use :func:`init_psf_dict` to create a
    trainable instance and :meth:`to_frozen` to export after training.

    All PSFModules must be of the same type (one model type per PSFDict).
    """

    _modules: dict[SensorBand, PSFModule] | None
    _data: dict[SensorBand, xr.Dataset]

    @classmethod
    def from_modules(
        cls,
        modules: dict[SensorBand, PSFModule],
    ) -> "PSFDict":
        """Create a trainable PSFDict holding live PSFModule objects.

        The returned instance is in *trainable* mode: it holds the original
        :class:`PSFModule` objects (with gradient-tracked parameters) and
        carries no pre-computed DataArrays.  Pass it to
        :class:`~adjeff.modules.models.psf_conv_module.PSFConvModule` to
        optimise the parameters, then call :meth:`to_frozen` to export.

        Parameters
        ----------
        modules : dict[SensorBand, PSFModule]
            Mapping from band to an instantiated PSFModule.

        Returns
        -------
        PSFDict
            Trainable PSFDict.
        """
        obj: PSFDict = cls.__new__(cls)
        obj._modules = dict(modules)
        obj._data = {}
        return obj

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
        obj._modules = None
        obj._data = {}
        for band, da in kernels.items():
            ds_vars: dict[str, xr.DataArray] = {"kernel": da}
            if params and band in params:
                for pname, pda in params[band].items():
                    ds_vars[f"param_{pname}"] = pda
            obj._data[band] = xr.Dataset(ds_vars)
        return obj

    # ------------------------------------------------------------------
    # Trainable mode helpers
    # ------------------------------------------------------------------

    @property
    def is_trainable(self) -> bool:
        """Return True when this PSFDict holds live PSFModule objects."""
        return getattr(self, "_modules", None) is not None

    def get_module(self, band: SensorBand) -> PSFModule:
        """Return the PSFModule for *band* (trainable mode only).

        Parameters
        ----------
        band : SensorBand
            Band of interest.

        Returns
        -------
        PSFModule
            Live PSFModule with gradient-tracked parameters.

        Raises
        ------
        RuntimeError
            If this PSFDict is not in trainable mode.
        """
        if not self.is_trainable or self._modules is None:
            raise RuntimeError(
                "get_module() requires a trainable PSFDict. "
                "Use init_psf_dict() to create one."
            )
        return self._modules[band]

    def to_frozen(self) -> "PSFDict":
        """Export current parameters to a frozen PSFDict.

        Calls :meth:`~adjeff.core._psf.PSFModule.to_dataarray` on each
        live module to capture the current (possibly trained) kernel.

        Returns
        -------
        PSFDict
            Frozen PSFDict with DataArray kernels reflecting the current
            parameter values.
        """
        if not self.is_trainable or self._modules is None:
            return self
        return PSFDict.from_kernels(
            {band: psf.to_dataarray() for band, psf in self._modules.items()}
        )

    @property
    def bands(self) -> list[SensorBand]:
        """Sorted list of band identifiers present in this PSFDict."""
        source = (
            self._modules
            if self.is_trainable and self._modules
            else self._data
        )
        return sorted(source.keys(), key=lambda b: b.value)

    def __getitem__(self, band: SensorBand) -> xr.Dataset:
        """Return the Dataset for *band*."""
        return self._data[band]

    def __setitem__(self, band: SensorBand, ds: xr.Dataset) -> None:
        """Store a Dataset under the *band* key."""
        self._data[band] = ds

    def __contains__(self, band: object) -> bool:
        """Check if *band* is stored in the PSFDict."""
        if self.is_trainable and self._modules is not None:
            return band in self._modules
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
        """Return the frozen kernel DataArray for *band*.

        Raises
        ------
        RuntimeError
            If this PSFDict is in trainable mode (call :meth:`to_frozen`
            first, or use :meth:`get_module` to access the live PSFModule).
        """
        if self.is_trainable:
            raise RuntimeError(
                "to_dataarray() is not available in trainable mode. "
                "Call to_frozen() first."
            )
        return self._data[band]["kernel"]

    def kernel(self, band: SensorBand) -> xr.DataArray:
        """Alias for to_dataarray()."""
        return self.to_dataarray(band)

    def _cache_dict(self) -> dict[str, Any]:
        """Return a joblib-hashable representation of all kernel data.

        xr.DataArray values are converted to nested lists so that joblib
        can hash them. Used by SceneModule._config_dict().
        """
        result: dict[str, Any] = {}
        for band, ds in self._data.items():
            result[str(band)] = {
                var: da.values for var, da in ds.data_vars.items()
            }
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def init_psf_dict(
    grids: dict[SensorBand, PSFGrid],
    model: type[PSFModule],
    init_parameters: dict[str, float] | dict[SensorBand, dict[str, float]],
) -> PSFDict:
    """Create a trainable :class:`PSFDict` with fresh PSFModule instances.

    Parameters
    ----------
    grids : dict[SensorBand, PSFGrid]
        Spatial grid for each band.
    model : type[PSFModule]
        PSF model class to instantiate (e.g. :class:`~adjeff.core.KingPSF`).
    init_parameters : dict[str, float] or dict[SensorBand, dict[str, float]]
        Initial parameter values.  Pass ``{"sigma": 0.1}`` to use the same
        values for every band, or ``{S2Band.B02: {"sigma": 0.1}, ...}`` to
        set per-band values.

    Returns
    -------
    PSFDict
        Trainable PSFDict.  Pass it to
        :class:`~adjeff.modules.models.psf_conv_module.PSFConvModule` for
        optimisation, then call :meth:`~PSFDict.to_frozen` to export.
    """
    modules: dict[SensorBand, PSFModule] = {}
    per_band = bool(init_parameters) and isinstance(
        next(iter(init_parameters)), SensorBand
    )
    for band, grid in grids.items():
        if per_band:
            params: dict[str, float] = cast(
                dict[SensorBand, dict[str, float]], init_parameters
            )[band]
        else:
            params = cast(dict[str, float], init_parameters)
        modules[band] = model(grid=grid, band=band, **params)  # type: ignore[call-arg]
    return PSFDict.from_modules(modules)
