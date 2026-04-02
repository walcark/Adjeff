"""Define a configuration for spectral parameters."""

from __future__ import annotations

from typing import Annotated, Any

import xarray as xr
from pydantic import PrivateAttr
from pydantic.functional_validators import BeforeValidator as Before

from adjeff.core.bands import SensorBand
from adjeff.exceptions import ConfigurationError
from adjeff.utils import _Config, to_arr


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
        """Construct a SpectralConfig from a list of SensorBand instances.

        Parameters
        ----------
        bands : list[SensorBand]
            Bands to sweep over. The ``wl`` coordinate is derived from
            ``band.wl_nm`` for each band. All bands must be of the same
            type.

        Returns
        -------
        SpectralConfig
            Config with ``wl`` populated from the band wavelengths.

        Raises
        ------
        ConfigurationError
            If *bands* contains instances of more than one ``SensorBand``
            subclass.
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
