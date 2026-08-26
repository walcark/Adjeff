"""Define a configuration for geometric parameters."""

from __future__ import annotations

from typing import Annotated

import xarray as xr
from pydantic import Field
from pydantic.functional_validators import BeforeValidator as Before

from adjeff.utils._config import _Config, to_arr


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
