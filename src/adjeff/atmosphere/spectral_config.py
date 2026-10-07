"""Wavelengths and the sensor bands they stand for.

Classes
-------
    SpectralConfig
        Wavelengths, each resolved to the nearest band of a sensor.
"""

from __future__ import annotations

from typing import Annotated, Any

import xarray as xr
from pydantic import PrivateAttr
from pydantic.functional_validators import BeforeValidator as Before

from adjeff.core.bands import SensorBand
from adjeff.exceptions import ConfigurationError
from adjeff.utils._config import _Config, to_arr


class SpectralConfig(_Config):
    """Wavelengths and the sensor bands they stand for.

    Build it from wavelengths and a band type, or from bands with
    :meth:`from_bands`.

    Parameters
    ----------
    wl : xr.DataArray
        Central wavelengths [nm], dim ``"wl"``.
    band_type : type[SensorBand]
        Sensor whose nearest band each wavelength resolves to.
    """

    wl: Annotated[xr.DataArray, Before(to_arr("wl", ge=0.0))]
    band_type: type[SensorBand]
    _bands: list[SensorBand] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Create bands from wavelength and sensor band type."""
        self._bands = [
            min(
                [b for b in self.band_type],
                key=lambda b: abs(b.wl_nm - wl_nm),
            )
            for wl_nm in self.wl
        ]

    @classmethod
    def from_bands(cls, bands: list[SensorBand]) -> SpectralConfig:
        """Build a config whose ``wl`` holds the wavelengths of *bands*.

        Raises
        ------
        ConfigurationError
            If *bands* mix several sensor types.
        """
        band_type: type[SensorBand] = type(bands[0])
        if not all(isinstance(b, band_type) for b in bands):
            raise ConfigurationError("All bands should have the same type.")

        wl_values = [b.wl_nm for b in bands]
        wl = xr.DataArray(wl_values, dims=["wl"], coords={"wl": wl_values})
        return cls(wl=wl, band_type=band_type)

    @property
    def bands(self) -> list[SensorBand]:
        """Return the list of SensorBand."""
        return self._bands
