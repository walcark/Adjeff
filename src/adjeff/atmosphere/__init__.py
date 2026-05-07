"""Define all classes and operations related to the Atmosphere.

The atmosphere in the ``adjeff`` context is composed of :
- aerosols and molecules — :class:`AtmoConfig`
- a sun / sensor geometry — :class:`GeoConfig`
- light with spectral bands — :class:`SpectralConfig`
- the earth surface, an object instantiated through :class:`SurfaceFactory`
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
