"""SweepBundle: Cartesian sweep over scalar parameters with chunked vectors."""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Iterator
from typing import Any, Callable

import structlog
import xarray as xr

from adjeff.utils._config import ConfigProtocol

logger = structlog.get_logger(__name__)


def _aggregate(
    configs: list[ConfigProtocol],
    names: list[str],
) -> tuple[dict[str, xr.DataArray], dict[str, Any]]:
    """Collect DataArrays and non-array fields from configs by name.

    First occurrence wins for each name.

    Raises
    ------
    ValueError
        If any name in *names* is absent from all configs.
    """
    das: dict[str, xr.DataArray] = {}
    other: dict[str, Any] = {}
    for cfg in configs:
        for k, v in cfg._arrays.items():
            if k not in das:
                das[k] = v
        for k, v in cfg._non_arrays.items():
            if k not in other:
                other[k] = v
    missing = set(names) - set(das)
    if missing:
        raise ValueError(f"Names {sorted(missing)!r} not found in any config.")
    return das, other


def _func_params(func: Callable[..., Any]) -> frozenset[str]:
    """Return explicit parameter names of *func* (excludes *args/**kwargs)."""
    _var = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    return frozenset(
        k
        for k, p in inspect.signature(func).parameters.items()
        if p.kind not in _var
    )


class SweepBundle:
    """Cartesian sweep over 1-D scalar parameters with chunked vector arrays.

    Scalars are iterated over as a Cartesian product; vectors are passed
    whole (or in chunks) to the function at each scalar step.

    All scalar DataArrays must be 1-D.  Use
    :meth:`~adjeff.sweep.UniqueIndex.build` to reduce multi-dimensional
    arrays to a 1-D ``"index"`` dimension before constructing a SweepBundle.

    Parameters
    ----------
    scalars
        Arrays iterated point-by-point (Cartesian product).  Must all be 1-D.
    vectors
        Arrays passed whole (or chunked) to the function.
    sweep_chunks
        Maximum slice size per vector dimension, e.g. ``{"wl": 4}``.
    """

    def __init__(
        self,
        scalars: dict[str, xr.DataArray],
        vectors: dict[str, xr.DataArray],
        sweep_chunks: dict[str, int] | None = None,
    ) -> None:
        for name, da in scalars.items():
            if da.ndim > 1:
                raise ValueError(
                    f"Scalar '{name}' has {da.ndim} dimensions. "
                    "Reduce to 1-D first (e.g. via UniqueIndex.build())."
                )
            if da.ndim == 1 and str(da.dims[0]) not in da.coords:
                da = da.assign_coords({str(da.dims[0]): da.values})
                scalars = {**scalars, name: da}

        self._scalars = scalars
        self._vectors = vectors
        self._chunks = sweep_chunks or {}

    @classmethod
    def from_configs(
        cls,
        configs: list[ConfigProtocol],
        scalar_names: list[str],
        vector_names: list[str],
        sweep_chunks: dict[str, int] | None = None,
    ) -> "SweepBundle":
        """Build a SweepBundle by extracting DataArrays from config objects.

        Parameters
        ----------
        configs
            Source config objects (AtmoConfig, GeoConfig, SpectralConfig, …).
        scalar_names
            Fields to iterate over as scalars (Cartesian product).
        vector_names
            Fields to pass as full / chunked arrays.
        sweep_chunks
            Optional chunk sizes for vector dimensions.
        """
        das, _ = _aggregate(configs, scalar_names + vector_names)
        return cls(
            scalars={k: das[k] for k in scalar_names if k in das},
            vectors={k: das[k] for k in vector_names if k in das},
            sweep_chunks=sweep_chunks,
        )

    def apply(
        self,
        func: Callable[..., xr.DataArray],
        **kwargs: Any,
    ) -> xr.DataArray:
        """Apply *func* over the full parameter space.

        Return the combined result.

        The outer loop iterates over the Cartesian product of scalar
        coordinates; the inner loop iterates over vector chunk slices.
        *func* receives only the keyword arguments it explicitly declares —
        extra bundle names are silently filtered out.

        Parameters
        ----------
        func
            Called at each (scalar_combo, vector_chunk).  Must return an
            unnamed :class:`xr.DataArray`.
        **kwargs
            Extra fixed arguments forwarded to every call of *func*.

        Returns
        -------
        xr.DataArray
            Combined result with scalar coordinates restored.
        """
        params = _func_params(func)
        all_results: list[xr.DataArray] = []

        if self._scalars:
            scalar_names = list(self._scalars)
            bcasted = dict(
                zip(scalar_names, xr.broadcast(*self._scalars.values()))
            )
            ref = next(iter(bcasted.values()))
            dim_order = [str(d) for d in ref.dims]
            coord_vals = {d: ref.coords[d].values for d in dim_order}
            idx_ranges = [range(ref.sizes[d]) for d in dim_order]
            n_steps = ref.size
        else:
            scalar_names, bcasted, ref = [], {}, None
            dim_order, coord_vals, idx_ranges, n_steps = [], {}, [], 1

        step = 0
        for idx in itertools.product(*idx_ranges) if idx_ranges else [()]:
            step += 1
            step_coords = {
                d: float(coord_vals[d][j]) for d, j in zip(dim_order, idx)
            }
            scalar_vals = {
                name: bcasted[name].isel(
                    {
                        d: int(j)
                        for d, j in zip(dim_order, idx)
                        if d in bcasted[name].dims
                    }
                )
                for name in scalar_names
            }

            logger.debug(
                "sweep step",
                step=f"{step}/{n_steps}",
                coords={k: round(float(v), 4) for k, v in step_coords.items()},
            )

            for chunk in self._iter_chunks():
                all_named = {**scalar_vals, **chunk}
                out = func(
                    **{
                        k: v
                        for k, v in {**all_named, **kwargs}.items()
                        if k in params
                    }
                )
                for d, v in step_coords.items():
                    if d not in out.dims:
                        out = out.expand_dims({d: [v]})
                all_results.append(out)

        combined = xr.combine_by_coords(all_results)
        if not isinstance(combined, xr.DataArray):
            raise TypeError(
                "combine_by_coords returned a Dataset. "
                "Make sure func always returns an unnamed DataArray."
            )
        return combined

    def _iter_chunks(self) -> Iterator[dict[str, xr.DataArray]]:
        """Yield one dict of vector arrays per chunk combination."""
        if not self._vectors or not self._chunks:
            yield dict(self._vectors)
            return

        chunk_specs: list[tuple[str, list[slice]]] = []
        for dim, size in self._chunks.items():
            dim_size = next(
                (
                    da.sizes[dim]
                    for da in self._vectors.values()
                    if dim in da.dims
                ),
                None,
            )
            if dim_size is None:
                continue
            chunk_specs.append(
                (
                    dim,
                    [
                        slice(s, min(s + size, dim_size))
                        for s in range(0, dim_size, size)
                    ],
                )
            )

        if not chunk_specs:
            yield dict(self._vectors)
            return

        for combo in itertools.product(*[slices for _, slices in chunk_specs]):
            chunk = dict(self._vectors)
            for (dim, _), slc in zip(chunk_specs, combo):
                chunk = {
                    k: v.isel({dim: slc}) if dim in v.dims else v
                    for k, v in chunk.items()
                }
            yield chunk
