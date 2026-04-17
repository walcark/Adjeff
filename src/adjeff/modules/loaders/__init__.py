"""Define loaders for earth observation products."""

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
