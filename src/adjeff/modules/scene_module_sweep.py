"""SceneModuleSweep: SceneModule with SweepBundle-based parameter sweep."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import xarray as xr

from adjeff.sweep import SweepBundle, UniqueIndex
from adjeff.sweep.bundle import _aggregate
from adjeff.utils import CacheStore, ConfigProtocol

from .scene_module import SceneModule

if TYPE_CHECKING:
    from adjeff.core import ImageDict


class SceneModuleSweep(SceneModule):
    """SceneModule with a SweepBundle factory for parameter-space iteration.

    Subclasses declare which config fields to iterate over via class
    variables, then call :meth:`_apply_bundle` in their ``_compute``
    implementation.

    Class variables to declare in subclasses
    -----------------------------------------
    scalar_dims : list[str]
        Field names iterated point-by-point as a Cartesian product.
    vector_dims : list[str]
        Field names passed as full (or chunked) arrays to the core function.

    Parameters
    ----------
    cache
        Result cache.
    sweep_chunks
        Chunk size per vector dimension for Smart-G call granularity,
        e.g. ``{"wl": 4}``.  ``None`` disables chunking.
    deduplicate_dims
        Spatial dimensions to deduplicate before sweeping, e.g. ``["x", "y"]``.
        Reduces redundant Smart-G calls when atmospheric
        parameters are 2-D maps.
    """

    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = []

    def __init__(
        self,
        cache: CacheStore | None = None,
        sweep_chunks: dict[str, int] | None = None,
        deduplicate_dims: list[str] | None = None,
    ) -> None:
        super().__init__(cache)
        self._sweep_chunks = sweep_chunks
        self._deduplicate_dims = deduplicate_dims

    @abstractmethod
    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        """Return the config instances to aggregate into the bundle."""

    @abstractmethod
    def _compute(self, scene: "ImageDict") -> "ImageDict":
        """Run the core transform using :meth:`_apply_bundle`."""

    def _make_bundle(self) -> tuple[SweepBundle, UniqueIndex | None]:
        """Build a SweepBundle from current configs.

        Applies optional deduplication via UniqueIndex when
        deduplicate_dims is set.
        """
        all_names = self.scalar_dims + self.vector_dims
        das, _ = _aggregate(list(self._get_configs()), all_names)

        dedup: UniqueIndex | None = None
        if self._deduplicate_dims:
            dedup, das = UniqueIndex.build(das, self._deduplicate_dims)

        for name in self.scalar_dims:
            if name in das and das[name].ndim > 1:
                raise ValueError(
                    f"Scalar '{name}' has shape {das[name].shape} after "
                    "deduplication. Add its dimensions to deduplicate_dims."
                )

        return SweepBundle(
            scalars={k: das[k] for k in self.scalar_dims if k in das},
            vectors={k: das[k] for k in self.vector_dims if k in das},
            sweep_chunks=self._sweep_chunks,
        ), dedup

    def _apply_bundle(
        self,
        func: Callable[..., xr.DataArray],
        **kwargs: Any,
    ) -> xr.DataArray:
        """Build bundle, apply *func*, expand deduplication if needed."""
        bundle, dedup = self._make_bundle()
        arr = bundle.apply(func, **kwargs)
        if dedup is not None:
            arr = dedup.expand(arr)
        return arr
