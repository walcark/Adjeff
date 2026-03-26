"""SceneModuleSweep: SceneModule with apply_ufunc-based parameter sweep."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from adjeff.modules._scene_module import SceneModule
from adjeff.utils import CacheStore
from adjeff.utils.sweep import ConfigProtocol, DimSweeper

if TYPE_CHECKING:
    from adjeff.core import ImageDict


class SceneModuleSweep(SceneModule):
    """SceneModule with parametric sweep over _Config dimensions.

    Extends :class:`SceneModule` for modules whose computation depends on
    atmospheric or geometric parameters that must be swept (e.g. Smart-G
    radiative transfer calls).  The sweep and optional deduplication are
    delegated to a :class:`~adjeff.utils.sweep.DimSweeper`; subclasses only
    expose physics via ``_core_band``.

    Class attributes to declare in subclasses
    ------------------------------------------
    scalar_dims : list[str]
        Dimensions the underlying engine requires as Python float scalars.
        ``apply_ufunc`` iterates over them automatically and reconstructs the
        output grid.
    vector_dims : list[str]
        Dimensions the underlying engine accepts as full 1-D numpy arrays in a
        single call.

    Guaranteed argument order for ``_core_band``
    --------------------------------------------
    1. ``required_vars`` arrays → np.ndarray (spatial dims, if any)
    2. ``vector_dims``          → np.ndarray 1-D (in ``vector_dims`` order)
    3. ``scalar_dims``          → float (in ``scalar_dims`` order)

    Parameters
    ----------
    cache:
        Result cache, see :class:`~adjeff.modules._scene_module.SceneModule`.
    chunks:
        Maximum chunk size per vector dim, e.g. ``{"h": 1000}``.
        Dims absent from the dict are passed whole. ``None`` disables chunking.
    deduplicate_dims:
        Spatial dimensions to deduplicate before sweeping, e.g. ``["x", "y"]``.
        When set, config parameters living on these dims are collapsed to a
        compact ``index`` dimension via ``_Config.unique()`` before the sweep,
        and the result is reconstructed to the original shape afterwards.
        ``None`` disables deduplication.  Config parameters with more than one
        dimension require this to be set.
    """

    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = []

    def __init__(
        self,
        cache: CacheStore | None = None,
        chunks: dict[str, int] | None = None,
        deduplicate_dims: list[str] | None = None,
    ) -> None:
        super().__init__(cache)
        self._sweeper = DimSweeper(
            scalar_dims=self.scalar_dims,
            vector_dims=self.vector_dims,
            chunks=chunks,
            deduplicate_dims=deduplicate_dims,
        )

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        """Return the tuple of _Config instances (Atmo, GeoConfig, ...)."""

    @abstractmethod
    def _core_band(self, *args: np.ndarray | float) -> np.ndarray:
        """Physics computation for one parameter combination.

        Receives arguments in order:
        - ``required_vars`` as np.ndarray (spatial dims, if present)
        - ``vector_dims``   as np.ndarray 1-D
        - ``scalar_dims``   as float (apply_ufunc has already iterated)

        Returns a np.ndarray with the same spatial dims as the input, or a
        scalar for modules with no ``required_vars``.
        """

    # ------------------------------------------------------------------
    # _compute (sealed)
    # ------------------------------------------------------------------

    def _compute(self, scene: "ImageDict") -> "ImageDict":
        """Drive the parameter sweep via DimSweeper for each band."""
        dim_arrays, inverse_map = self._sweeper.collect(*self._get_configs())

        for band in scene.bands:
            band_ds = scene[band]
            required = [band_ds[v] for v in self.required_vars]

            result = self._sweeper.apply(
                self._core_band, required, dim_arrays, inverse_map
            )
            band_ds[self.output_vars[0]] = result

            self._log.debug(
                "sweep done",
                band=band,
                deduplicated=inverse_map is not None,
            )

        return scene
