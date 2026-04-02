"""SceneModuleSweep: SceneModule with ConfigBundle-based parameter sweep."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from adjeff.utils import CacheStore, ConfigBundle, ConfigProtocol

from .scene_module import SceneModule

if TYPE_CHECKING:
    from adjeff.core import ImageDict


class SceneModuleSweep(SceneModule):
    """SceneModule exposing a ConfigBundle factory to subclasses.

    Extends :class:`SceneModule` with a configured :meth:`_make_bundle`
    helper.  The sweep infrastructure (aggregation, joint deduplication,
    flattening) is encapsulated in :class:`~adjeff.utils.ConfigBundle`;
    ``_compute`` remains abstract — each subclass drives the sweep and
    assigns results back to the :class:`ImageDict` as it sees fit.

    Class attributes to declare in subclasses
    -----------------------------------------
    scalar_dims : list[str]
        Attribute names passed as float scalars (iterated via xarray
        broadcasting).
    vector_dims : list[str]
        Attribute names passed as full 1D arrays in a single call.

    Parameters
    ----------
    cache:
        Result cache, see :class:`~adjeff.modules._scene_module.SceneModule`.
    chunks:
        Chunk size per vector attribute, e.g. ``{"wl": 50}``.
        ``None`` disables chunking.
    deduplicate_dims:
        Spatial dimensions to deduplicate jointly before sweeping,
        e.g. ``["x", "y"]``.  ``None`` disables deduplication.
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
        self._chunks = chunks
        self._deduplicate_dims = deduplicate_dims

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        """Return the config instances to aggregate into the bundle."""

    @abstractmethod
    def _compute(self, scene: "ImageDict") -> "ImageDict":
        """Run the core transform using :meth:`_make_bundle`."""

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _make_bundle(self) -> ConfigBundle:
        """Construct a ConfigBundle from the current configs."""
        return ConfigBundle(
            list(self._get_configs()),
            scalars=self.scalar_dims,
            vectors=self.vector_dims,
            deduplicate_dims=self._deduplicate_dims,
            chunks=self._chunks,
        )
