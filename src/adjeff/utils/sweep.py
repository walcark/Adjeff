"""Apply_ufunc-based parameter sweep with optional deduplication."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Callable, Protocol, Self, cast

import numpy as np
import structlog
import xarray as xr

from .xrutils import _normalize_da

logger = structlog.get_logger(__name__)


class ConfigProtocol(Protocol):
    """Contract for Configuration classes (Atmo, Geo, ...)."""

    @property
    def _arrays(self) -> dict[str, xr.DataArray]:
        """Return DataArrays attributes of the configuration."""
        ...

    def unique(self, dims: list[str]) -> tuple[Self, xr.DataArray]:
        """Deduplicate the fields living on dims, keep other intact."""
        ...


class DimSweeper:
    """Orchestrate a parametric sweep over _Config dimensions via apply_ufunc.

    Handles three orthogonal concerns:

    - **scalar_dims**: dimensions that the underlying engine (e.g. Smart-G)
      requires as Python floats. ``apply_ufunc(vectorize=True)`` iterates over
      them automatically and reconstructs the output grid.

    - **vector_dims**: dimensions the engine accepts as full 1-D numpy arrays
      in a single call. Optional chunking limits peak GPU memory.

    - **deduplication**: when config parameters vary over spatial dimensions
      (e.g. ``aot(x, y)``), the full Cartesian sweep over every pixel is
      prohibitively expensive. Passing ``deduplicate_dims`` triggers
      ``_Config.unique()`` to collapse those spatial dimensions into a compact
      ``index`` dimension before sweeping, then reconstructs the original shape
      afterwards.

    Parameters
    ----------
    scalar_dims:
        Dims passed as float scalars to the target function.
    vector_dims:
        Dims passed as np.ndarray 1-D to the target function.
    chunks:
        Maximum chunk size per vector dim, e.g. ``{"h": 1000}``.
        Chunked dims must appear in the function output (they are added to
        ``output_core_dims``). Dims absent from the dict are passed whole.
        ``None`` disables chunking.
    deduplicate_dims:
        Spatial dimensions to deduplicate before sweeping, e.g. ``["x", "y"]``.
        ``None`` disables deduplication.  When set, config params living on
        these dims are reduced to a single ``index`` dimension via
        ``_Config.unique()``, and the result is reconstructed to the original
        shape after the sweep.

    Notes
    -----
    Config parameters with more than one dimension are only valid when
    ``deduplicate_dims`` is provided.  A ``ValueError`` is raised otherwise.
    """

    def __init__(
        self,
        scalar_dims: list[str],
        vector_dims: list[str],
        chunks: dict[str, int] | None = None,
        deduplicate_dims: list[str] | None = None,
    ) -> None:
        self.scalar_dims = scalar_dims
        self.vector_dims = vector_dims
        self._chunks = chunks or {}
        self._deduplicate_dims = deduplicate_dims

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self, *configs: ConfigProtocol
    ) -> tuple[dict[str, xr.DataArray], xr.DataArray | None]:
        """Extract per-dim DataArrays from one or more _Config instances.

        When ``deduplicate_dims`` is set, calls ``_Config.unique()`` on the
        first config that carries those dims, reducing spatial parameters to a
        compact ``index`` dimension.

        Returns
        -------
        dim_arrays:
            Mapping from dim name to DataArray, ready to be passed to
            ``apply()``.
        inverse_map:
            DataArray mapping the original spatial dims back to ``index``.
            ``None`` when deduplication is disabled.

        Raises
        ------
        ValueError
            If a config parameter has more than one dimension and
            ``deduplicate_dims`` is not set.
        """
        inverse_map: xr.DataArray | None = None

        if self._deduplicate_dims is not None:
            configs, inverse_map = self._deduplicate(configs)

        dim_arrays = self._extract(configs)

        logger.debug(
            "collected dim arrays",
            scalar_dims=self.scalar_dims,
            vector_dims=self.vector_dims,
            dims={k: list(v.shape) for k, v in dim_arrays.items()},
            deduplicated=inverse_map is not None,
        )

        return dim_arrays, inverse_map

    def apply(
        self,
        func: Callable[..., np.ndarray],
        required_arrays: list[xr.DataArray],
        dim_arrays: dict[str, xr.DataArray],
        inverse_map: xr.DataArray | None,
    ) -> xr.DataArray:
        """Call ``func`` via ``apply_ufunc``, sweeping over ``scalar_dims``.

        When ``chunks`` is set, vector dims are split into sub-arrays before
        each ``apply_ufunc`` call and results are recombined via
        ``xr.combine_by_coords``.

        Parameters
        ----------
        func:
            Target function (e.g. ``_core_band``).  Receives arguments in
            order: ``required_arrays`` as np.ndarray, ``vector_dims`` as
            np.ndarray 1-D, ``scalar_dims`` as float.
        required_arrays:
            Spatial input DataArrays (e.g. ``[band_ds["rho_s"]]``).
            Empty list for modules with no spatial input.
        dim_arrays:
            Dict produced by ``collect()``.
        inverse_map:
            DataArray produced by ``collect()``.  When not ``None``, the
            result indexed along ``index`` is reconstructed to the original
            spatial shape via ``isel(index=inverse_map)``.

        Returns
        -------
        xr.DataArray
            Result with scalar dims reconstructed by ``apply_ufunc`` and, if
            deduplication was used, spatial dims restored via ``inverse_map``.
        """
        chunk_results = [
            self._apply_ufunc(func, required_arrays, chunk, dim_arrays)
            for chunk in self._iter_vector_chunks(dim_arrays)
        ]

        result: xr.DataArray = cast(
            xr.DataArray,
            xr.combine_by_coords(chunk_results)
            if len(chunk_results) > 1
            else chunk_results[0],
        )

        logger.debug(
            "apply_ufunc done",
            result_dims=list(result.dims),
            result_shape=list(result.shape),
        )

        if inverse_map is not None:
            result = result.isel(index=inverse_map)
            logger.debug(
                "spatial reconstruction done",
                result_dims=list(result.dims),
                result_shape=list(result.shape),
            )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _iter_vector_chunks(
        self, dim_arrays: dict[str, xr.DataArray]
    ) -> Iterator[dict[str, xr.DataArray]]:
        """Yield combinations of vector dim slices according to chunks.

        Dims absent from ``self._chunks`` are yielded whole.
        """
        slices_per_dim: list[list[xr.DataArray]] = []
        for d in self.vector_dims:
            da = dim_arrays[d]
            size = self._chunks.get(d)
            if size is not None:
                slices_per_dim.append(
                    [
                        da.isel({d: slice(i, i + size)})
                        for i in range(0, len(da), size)
                    ]
                )
            else:
                slices_per_dim.append([da])

        for combo in itertools.product(*slices_per_dim):
            yield dict(zip(self.vector_dims, combo))

    def _apply_ufunc(
        self,
        func: Callable[..., np.ndarray],
        required_arrays: list[xr.DataArray],
        vector_chunk: dict[str, xr.DataArray],
        dim_arrays: dict[str, xr.DataArray],
    ) -> xr.DataArray:
        """Single apply_ufunc call for one combination of vector dim chunks."""
        args = [
            *required_arrays,
            *[vector_chunk[d] for d in self.vector_dims],
            *[dim_arrays[d] for d in self.scalar_dims],
        ]
        input_core_dims: list[list[str]] = [
            *[[str(d) for d in da.dims] for da in required_arrays],
            *[[d] for d in self.vector_dims],
            *[[] for _ in self.scalar_dims],
        ]
        out_core = (
            list(required_arrays[0].dims) if required_arrays else []
        ) + [d for d in self.vector_dims if d in self._chunks]

        return cast(
            xr.DataArray,
            xr.apply_ufunc(
                func,
                *args,
                input_core_dims=input_core_dims,
                output_core_dims=[out_core],
                vectorize=True,
            ),
        )

    def _extract(
        self, configs: tuple[ConfigProtocol, ...]
    ) -> dict[str, xr.DataArray]:
        """Extract and validate DataArrays for all declared dims.

        Raises
        ------
        ValueError
            If a parameter DataArray has more than one dimension and
            ``deduplicate_dims`` is not set.
        """
        all_dims = set(self.scalar_dims) | set(self.vector_dims)
        result: dict[str, xr.DataArray] = {}

        for cfg in configs:
            for name, da in cfg._arrays.items():
                if name not in all_dims or name in result:
                    continue
                result[name] = _normalize_da(name, da, self._deduplicate_dims)

        return result

    def _deduplicate(
        self, configs: tuple[ConfigProtocol, ...]
    ) -> tuple[tuple[ConfigProtocol, ...], xr.DataArray]:
        """Deduplicate multiple-parameters configurations."""
        assert self._deduplicate_dims is not None

        deduped: list[ConfigProtocol] = []
        inverse_map: xr.DataArray | None = None

        for cfg in configs:
            arrays = cfg._arrays
            has_spatial = any(
                any(d in da.dims for d in self._deduplicate_dims)
                for da in arrays.values()
            )
            if has_spatial:
                cfg_unique, inv = cfg.unique(self._deduplicate_dims)
                deduped.append(cfg_unique)
                if inverse_map is None:
                    inverse_map = inv
            else:
                deduped.append(cfg)

        if inverse_map is None:
            raise ValueError(
                f"None of the provided configs have parameters living on "
                f"deduplicate_dims={self._deduplicate_dims}."
            )

        n_original = int(
            np.prod([inverse_map.sizes[d] for d in self._deduplicate_dims])
        )
        n_unique = int(inverse_map.max()) + 1
        logger.debug(
            "deduplication complete",
            deduplicate_dims=self._deduplicate_dims,
            n_original_pixels=n_original,
            n_unique_combinations=n_unique,
            reduction_factor=round(n_original / n_unique, 2),
        )

        return tuple(deduped), inverse_map
