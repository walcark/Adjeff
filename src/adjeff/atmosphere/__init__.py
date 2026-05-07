"""Atmospheric configuration and Smart-G atmosphere factory for adjeff.

**Configuration models**

- :class:`AtmoConfig` — aerosol and molecular parameters (AOT, RH, species).
- :class:`GeoConfig` — sun/sensor geometry (SZA, VZA, SAA, VAA).
- :class:`SpectralConfig` — spectral bands and wavelengths.

**Surface**

- :class:`SurfaceFactory` — builds Smart-G ``LambSurface`` and
  ``Environment`` objects from an adjeff scene.

**Factory**

- :func:`create_atmosphere` — assembles a multi-profile Smart-G
  atmosphere from :class:`AtmoConfig`, :class:`GeoConfig` and
  :class:`SpectralConfig` parameters.
"""

from .atmo_config import AtmoConfig
from .atmo_factory import create_atmosphere
from .geo_config import GeoConfig
from .spectral_config import SpectralConfig
from .surface import SurfaceFactory

__all__ = [
    "AtmoConfig",
    "create_atmosphere",
    "GeoConfig",
    "SpectralConfig",
    "SurfaceFactory",
]
