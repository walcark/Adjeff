"""Define a configuration for geometric parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import xarray as xr
from pydantic import Field
from pydantic.functional_validators import BeforeValidator as Before

if TYPE_CHECKING:
    from smartg.smartg import Sensor

from ._config import _Config, to_arr


class GeoConfig(_Config):
    """Pydantic model for the geometric parameters.

    Parameters
    ----------
    sza : xr.DataArray
        Sun zenith angle [°].
    saa : xr.DataArray
        Sun azimuth angle [°].
    vza : xr.DataArray
        Viewing zenith angle [°].
    vaa : xr.DataArray
        Viewing azimuth angle [°].
    sat_height : float
        Satellite elevation [km].
    """

    sza: Annotated[xr.DataArray, Before(to_arr("sza", ge=0.0, le=90.0))]
    vza: Annotated[xr.DataArray, Before(to_arr("vza", ge=0.0, le=90.0))]
    saa: Annotated[xr.DataArray, Before(to_arr("saa", ge=0.0, le=360.0))]
    vaa: Annotated[xr.DataArray, Before(to_arr("vaa", ge=0.0, le=360.0))]
    sat_height: float = Field(default=700.0, ge=0.0)

    @property
    def sun_le(self) -> dict[str, Any]:
        """Return the Sun local-estimate."""
        return {
            "th_deg": self.sza.data,
            "phi_deg": self.saa.data,
            "zip": True,
        }

    @property
    def sat_le(self) -> dict[str, Any]:
        """Return the Satellite local-estimate."""
        return {
            "th_deg": self.vza.data,
            "phi_deg": self.saa.data,
            "zip": True,
        }

    @property
    def sun_sensor(self) -> Sensor:
        """Return the Sun Smart-G Sensor object."""
        from smartg.smartg import Sensor

        return Sensor(
            POSZ=self.sat_height,
            THDEG=180.0 - self.sza.data,
            PHDEG=self.saa.data,
            LOC="ATMOS",
        )

    @property
    def sat_sensor(self) -> Sensor:
        """Return the Satellite Smart-G Sensor object."""
        from smartg.smartg import Sensor

        return Sensor(
            POSZ=self.sat_height,
            THDEG=180.0 - self.vza.data,
            PHDEG=self.vaa.data,
            LOC="ATMOS",
        )

    @property
    def satellite_relative_position(self) -> tuple[float, float]:
        """Return satellite position relative to the observation point."""
        # Compute the observation angles cosines
        tan_vza: float = np.tan(np.radians(self.vza.data))
        cos_vaa: float = np.cos(np.radians(180 - self.vaa.data))
        sin_vaa: float = np.sin(np.radians(180 - self.vaa.data))
        # Compute relative positions
        x: float = (self.sat_height * tan_vza) * cos_vaa
        y: float = (self.sat_height * tan_vza) * sin_vaa

        return (np.round(x, 4), np.round(y, 4))
