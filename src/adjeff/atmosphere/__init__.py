"""Smart-G atmosphere and surface inputs, built from adjeff configurations.

Classes
-------
    AtmoConfig
        Aerosol and molecular parameters (AOT, RH, ground and aerosol
        heights, species).
    GeoConfig
        Sun and sensor geometry.
    SpectralConfig
        Wavelengths and the sensor bands they resolve to.
    SurfaceFactory
        Smart-G surface and environment of an adjeff scene.

Functions
---------
    create_atmosphere
        Multi-profile Smart-G atmosphere, one profile per parameter set.
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
