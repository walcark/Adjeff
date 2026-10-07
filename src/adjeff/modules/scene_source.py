"""Base class of the modules that create a scene.

Classes
-------
    SceneSource
        SceneModule with no required input, called without a scene.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import xarray as xr

from adjeff.utils import CacheStore

from .scene_module import SceneModule

if TYPE_CHECKING:
    from adjeff.core import ImageDict
    from adjeff.core.bands import SensorBand


class SceneSource(SceneModule):
    """SceneModule creating a scene, e.g. by loading a product.

    Called without a scene, it starts from an empty one holding *bands*.

    Parameters
    ----------
    bands : list[SensorBand]
        Bands the source produces.
    cache, rename : optional
        See :class:`SceneModule`.
    """

    _required_vars: ClassVar[list[str]] = []

    def __init__(
        self,
        bands: list["SensorBand"],
        cache: CacheStore | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        self._bands = bands
        super().__init__(cache=cache, rename=rename)

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
