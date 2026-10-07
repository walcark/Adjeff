"""Earth observation products read into an ImageDict.

Classes
-------
    ProductLoader
        Base loader, producing ``rho_s``.
    GeometryMixin
        Adds ``vza``, ``vaa``, ``sza``, ``saa``.
    AtmosphereMixin
        Adds ``aot``, ``rh``, ``href``.
    ElevationMixin
        Adds ``h``, from a DEM.
    MajaLoader
        Sentinel-2 MAJA L2A products.
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
