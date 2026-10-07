"""Base of the configuration models.

Every field is stored as a DataArray, so that a scalar, a sweep and a
spatial map are passed the same way.

Classes
-------
    _Config
        Frozen pydantic model of DataArray fields.
    ConfigProtocol
        What the samplers read from a config.

Functions
---------
    to_arr
        Validator turning a scalar, list or DataArray into a DataArray.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

Parameter = float | int | np.ndarray | list[float | int] | xr.DataArray
Module = Callable[["_Config"], xr.DataArray]


class ConfigProtocol(Protocol):
    """Contract for Configuration classes (Atmo, Geo, ...)."""

    @property
    def _arrays(self) -> dict[str, xr.DataArray]:
        """Return DataArrays attributes of the configuration."""
        ...

    @property
    def _non_arrays(self) -> dict[str, Any]:
        """Return non-DataArray fields of the configuration."""
        ...


def to_arr(
    field_name: str,
    ge: Optional[float] = None,
    le: Optional[float] = None,
) -> Callable[[Parameter], xr.DataArray]:
    """Return a validator turning a field into a DataArray within ``[ge, le]``.

    A scalar or a 1-D array becomes a DataArray on dim *field_name*; a
    DataArray must name its dims.

    Raises
    ------
    ValueError
        On implicit dims, an unnamed array of more than 1 dim, or a
        value out of bounds.
    """

    def _validate(v: Parameter) -> xr.DataArray:
        if isinstance(v, xr.DataArray):
            da = v
            default_dims = [d for d in da.dims if str(d).startswith("dim_")]
            if default_dims:
                raise ValueError(
                    f"'{field_name}': DataArray has implicit dimensions "
                    f"{default_dims} Please provide explicit dimension names."
                )
            # Label a bare 1-D dim with its values, so that .sel works.
            if da.ndim == 1 and da.dims[0] not in da.coords:
                da = da.assign_coords({str(da.dims[0]): da.values})
        elif isinstance(v, (float, int)):
            arr = np.atleast_1d(v)
            da = xr.DataArray(arr, dims=[field_name], coords={field_name: arr})
        else:
            arr = np.asarray(v)
            if arr.ndim == 1:
                da = xr.DataArray(arr, dims=[field_name], coords={field_name: arr})
            else:
                raise ValueError(
                    f"'{field_name}': {arr.ndim}D array without explicit "
                    "`dims`, use an xr.DataArray instead."
                )
        if ge is not None and float(da.min()) < ge:
            raise ValueError(f"'{field_name}': minimal value {float(da.min())} < {ge}.")
        if le is not None and float(da.max()) > le:
            raise ValueError(f"'{field_name}': maximal value {float(da.max())} > {le}.")
        return da

    return _validate


class _Config(BaseModel):
    """Frozen pydantic model whose fields are DataArrays, or not."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @property
    def _arrays(self) -> dict[str, xr.DataArray]:
        """Return only the xr.DataArray fields of this config."""
        return {
            k: v
            for k in type(self).model_fields
            if isinstance(v := getattr(self, k), xr.DataArray)
        }

    @property
    def _non_arrays(self) -> dict[str, Any]:
        """Return non-DataArray fields of this config."""
        arrays = self._arrays
        return {k: getattr(self, k) for k in type(self).model_fields if k not in arrays}

    @property
    def _stable_hash_repr(self) -> dict[str, object]:
        """Fields as plain Python values, for a cache key stable across builds."""
        result: dict[str, object] = {
            k: v.values.tolist() for k, v in self._arrays.items()
        }
        result.update(self._non_arrays)
        return result

    @property
    def dataset(self) -> xr.Dataset:
        """Return the atmosphere parameters as a dataset."""
        return xr.Dataset(self._arrays)
