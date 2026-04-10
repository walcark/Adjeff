"""Define the base class for atmospheric and geometric parameters.

The class ensures that both scalar parameters, spatial distribution
of parameters (2D images) or just sensibility study of parameters
can be passed in a similar way to any module that uses those classes.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Protocol, Self

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

    def unique(self, dims: list[str]) -> tuple[Self, xr.DataArray]:
        """Deduplicate the fields living on dims, keep other intact."""
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

    def unique(self, dims: list[str]) -> tuple[Self, xr.DataArray]:
        """Deduplicate the fields living on `dims`, and keep other intact.

        Parameters
        ----------
        dims : list[str]
            Dimensions on which to compute unique.

        Returns
        -------
        config_unique : _Config
            Dataset with fields on `dims` reduced to a single `index` dimension
            and fields out of `dims` kept unchanged.
        inverse_map : DataArray(dims)
            Keeps to index of each pixels to rebuild a DataArray computed on
            index.
        """
        arrays = self._arrays
        others = {
            k: getattr(self, k)
            for k in type(self).model_fields
            if k not in arrays
        }

        # Split DataArrays between those living on `dims` and the rest
        varying: dict[str, xr.DataArray] = {
            k: v for k, v in arrays.items() if any(d in v.dims for d in dims)
        }
        kept: dict[str, xr.DataArray] = {
            k: v
            for k, v in arrays.items()
            if not any(d in v.dims for d in dims)
        }

        # Strip coords on target dims before creating the Dataset: different
        # fields can have value-based coords on the same dim, which would cause
        # xarray to silently align/reindex on label rather than position when
        # building the Dataset.
        varying_stripped = {
            k: v.drop_vars([d for d in dims if d in v.coords], errors="ignore")
            for k, v in varying.items()
        }

        # Broadcast before stack to handle partial dimensions
        ds = xr.Dataset(varying_stripped)
        broadcasted = dict(
            zip(varying, xr.broadcast(*[ds[k] for k in varying]))
        )
        stacked = xr.Dataset(broadcasted).stack(index=dims)
        values = np.stack([stacked[k].values for k in varying], axis=1)
        unique_rows, inverse_indices = np.unique(
            values,
            axis=0,
            return_inverse=True,
        )

        # Create the unique atmosphere configuration
        unique_params = {
            name: xr.DataArray(unique_rows[:, i], dims=["index"])
            for i, name in enumerate(varying)
        }
        unique_atm: Self = type(self)(**unique_params, **kept, **others)

        # Create the inverse map — positional only, no coords to avoid
        # duplicate-label issues when the original dim has repeated values.
        ref = next(iter(broadcasted.values()))
        inverse_map = xr.DataArray(
            inverse_indices.reshape([ref.sizes[d] for d in dims]),
            dims=dims,
        )

        return unique_atm, inverse_map

    def iter(self, n_batch: int, dim: str) -> Iterator[Self]:
        """Yield sub-configs by slicing *dim* into chunks of size *n_batch*.

        The chunk size is computed so that the total number of elements
        across all dimensions does not exceed *n_batch* per iteration.
        If *dim* is not present in any array the whole configuration is
        yielded as a single chunk.

        Parameters
        ----------
        n_batch : int
            Target maximum number of elements per chunk (across all dims).
        dim : str
            Dimension name along which to slice.

        Yields
        ------
        _Config
            A sub-configuration of the same type with *dim* sliced.
        """
        arrays = self._arrays
        dim_sizes: dict[str, int] = {}
        for v in arrays.values():
            for d, s in v.sizes.items():
                dim_sizes.setdefault(str(d), s)

        if dim not in dim_sizes:
            yield self
            return

        other_size = max(
            1, int(np.prod([s for d, s in dim_sizes.items() if d != dim]))
        )
        chunk_size = max(1, n_batch // other_size)

        for start in range(0, dim_sizes[dim], chunk_size):
            slc = slice(start, start + chunk_size)
            yield type(self)(
                **{
                    k: v.isel({dim: slc}) if dim in v.dims else v
                    for k, v in arrays.items()
                },
                **{
                    k: getattr(self, k)
                    for k in type(self).model_fields
                    if k not in arrays
                },
            )

    def run(self, fn: Module, n_batch: int, dim: str) -> xr.DataArray:
        """Apply *fn* on each chunked sub-configuration and concatenate.

        Parameters
        ----------
        fn : Module
            Callable that takes a ``_Config`` instance and returns a
            ``xr.DataArray``.
        n_batch : int
            Target maximum number of elements per chunk, forwarded to
            :meth:`iter`.
        dim : str
            Dimension to slice along, forwarded to :meth:`iter`.  The
            outputs are concatenated along this same dimension.

        Returns
        -------
        xr.DataArray
            Concatenation of all per-batch outputs along *dim*.
        """
        return xr.concat(
            [fn(batch) for batch in self.iter(n_batch, dim)],
            dim=dim,
        )
