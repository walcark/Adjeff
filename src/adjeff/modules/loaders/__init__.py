"""Loaders for earth observation products into :class:`~adjeff.core.ImageDict`.

The base class is :class:`ProductLoader`, a
:class:`~adjeff.modules.SceneSource` that always populates ``rho_s``.
Additional variables are declared via mixins:

- :class:`GeometryMixin` — adds ``vza``, ``vaa``, ``sza``, ``saa``.
- :class:`AtmosphereMixin` — adds ``aot``, ``rh``, ``href``.
- :class:`ElevationMixin` — adds ``h`` (surface elevation from a DEM).

Concrete implementation: :class:`MajaLoader` (MAJA L2A processor
output for Sentinel-2).
"""

from .maja_loader import MajaLoader
from .product_loader import (
    AtmosphereMixin,
    ElevationMixin,
    GeometryMixin,
    ProductLoader,
)

__all__ = [
    "AtmosphereMixin",
    "ElevationMixin",
    "GeometryMixin",
    "MajaLoader",
    "ProductLoader",
]
