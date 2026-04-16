"""Base class for scene source modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import xarray as xr

from adjeff.utils import CacheStore

from .scene_module import SceneModule

if TYPE_CHECKING:
    from adjeff.core import ImageDict
    from adjeff.core.bands import SensorBand


class SceneSource(SceneModule):
    """A SceneModule that produces an ImageDict without requiring any input.

    Unlike :class:`SceneModule` (which transforms an existing scene), a
    ``SceneSource`` creates a scene from scratch — typically by loading data
    from disk or an external product.  Passing a scene to :meth:`forward` is
    optional; when omitted an empty :class:`~adjeff.core.ImageDict` with the
    declared bands is created automatically.

    Subclasses must still implement :meth:`_compute`.

    Parameters
    ----------
    bands : list[SensorBand]
        Bands that this source will populate.
    cache : CacheStore | None
        Optional on-disk cache.
    """

    required_vars: ClassVar[list[str]] = []

    def __init__(
        self,
        bands: list["SensorBand"],
        cache: CacheStore | None = None,
    ) -> None:
        self._bands = bands
        super().__init__(cache=cache)

    @property
    def bands(self) -> list["SensorBand"]:
        """Bands this source will produce."""
        return self._bands

    def __call__(
        self,
        scene: "ImageDict | None" = None,
    ) -> "ImageDict":
        """Create or enrich a scene."""
        return self.forward(scene)

    def forward(
        self,
        scene: "ImageDict | None" = None,
    ) -> "ImageDict":
        """Run the source, optionally enriching an existing *scene*.

        When *scene* is ``None``, an empty
        :class:`~adjeff.core.ImageDict` is built from the declared
        :attr:`bands` before delegating to :meth:`SceneModule.forward`.
        """
        from adjeff.core import ImageDict

        if scene is None:
            scene = ImageDict({b: xr.Dataset() for b in self._bands})
        return SceneModule.forward(self, scene)
