"""Define the base class for atmospheric and geometric parameters.

The class ensures that both scalar parameters, spatial distribution
of parameters (2D images) or just sensibility study of parameters
can be passed in a similar way to any module that uses those classes.
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
    """Before validator for the xr.DataArray in _Config.

    Ensure that even float, int or array-like inputs are converted in
    a valid xr.DataArray.

    Parameters
    ----------
    field_name : str
        The name of the field to register.
    ge : float
        The minimal value of the xr.DataArray.
    le : float
        The maximal value of the xr.DataArray.

    Raises
    ------
    ValueError
        If an input array with more than 2 dimensions is used.
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
            # Assign coords for 1D dims that have none, so label-based
            # selection (sel, isel by label) works out of the box.
            if da.ndim == 1 and da.dims[0] not in da.coords:
                da = da.assign_coords({str(da.dims[0]): da.values})
        elif isinstance(v, (float, int)):
            arr = np.atleast_1d(v)
            da = xr.DataArray(arr, dims=[field_name], coords={field_name: arr})
        else:
            arr = np.asarray(v)
            if arr.ndim == 1:
                da = xr.DataArray(
                    arr, dims=[field_name], coords={field_name: arr}
                )
            else:
                raise ValueError(
                    f"'{field_name}': {arr.ndim}D array without explicit "
                    "`dims`, use an xr.DataArray instead."
                )
        if ge is not None and float(da.min()) < ge:
            raise ValueError(
                f"'{field_name}': minimal value {float(da.min())} < {ge}."
            )
        if le is not None and float(da.max()) > le:
            raise ValueError(
                f"'{field_name}': maximal value {float(da.max())} > {le}."
            )
        return da

    return _validate


class _Config(BaseModel):
    """Base Pydantic model for the atmosphere / Geometric parameters.

    The parameters are defined as xr.DataArray, this enable to define them
    as both scalars, 2D maps or just varying parameters for sensibility
    studies.
    """

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
        return {
            k: getattr(self, k)
            for k in type(self).model_fields
            if k not in arrays
        }

    @property
    def _stable_hash_repr(self) -> dict[str, object]:
        """Return a primitive-only repr suitable for stable cache keying.

        Converts DataArray fields to plain Python lists so that
        ``joblib.hash`` produces the same result regardless of the Python
        environment (e.g. CPU vs GPU builds).
        """
        result: dict[str, object] = {
            k: v.values.tolist() for k, v in self._arrays.items()
        }
        result.update(self._non_arrays)
        return result

    @property
    def dataset(self) -> xr.Dataset:
        """Return the atmosphere parameters as a dataset."""
        return xr.Dataset(self._arrays)
