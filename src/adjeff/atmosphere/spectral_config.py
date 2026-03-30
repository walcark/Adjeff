"""Define a configuration for spectral parameters."""

from __future__ import annotations

from typing import Annotated

import xarray as xr
from pydantic.functional_validators import BeforeValidator as Before

from adjeff.core.bands import SensorBand
from adjeff.exceptions import ConfigurationError

from ._config import _Config, to_arr


class SpectralConfig(_Config):
    """Pydantic model for spectral parameters.

    Only contains wavelength and sensor bands. Can either be instanciated
    through wavelengths specification (and sensor type) or directly from
    SensorBand specification (classmethod ``from_bands()``).

    Parameters
    ----------
    wl : xr.DataArray
        Central wavelengths [nm], dim ``"wl"``.
    bands : list[SensorBand]
        Source bands, kept for inverse mapping via :meth:`find_band`.
    """

    wl: Annotated[xr.DataArray, Before(to_arr("wl", ge=0.0))]
    band_type: type[SensorBand]

    @classmethod
    def from_bands(cls, bands: list[SensorBand]) -> SpectralConfig:
        """Construct SpectralParams from a list of SensorBand instances.

        Parameters
        ----------
        bands:
            Bands to sweep over. The ``wl`` coordinate is derived from
            ``band.wl_nm`` for each band.
        """
        band_type: type[SensorBand] = type(bands[0])
        if not all(isinstance(b, band_type) for b in bands):
            raise ConfigurationError("All bands should have the same type.")

        wl_values = [b.wl_nm for b in bands]
        wl = xr.DataArray(wl_values, dims=["wl"], coords={"wl": wl_values})
        return cls(wl=wl, band_type=band_type)

    @property
    def bands(self) -> list[SensorBand]:
        """Return the list of SensorBand objects."""
        return [
            min(
                [b for b in self.band_type],
                key=lambda b: abs(b.wl_nm - wl_nm),
            )
            for wl_nm in self.wl
        ]
