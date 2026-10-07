"""Sun and sensor geometry.

Classes
-------
    GeoConfig
        Zenith and azimuth angles of sun and sensor, and satellite height.
"""

from __future__ import annotations

from typing import Annotated

import xarray as xr
from pydantic import Field
from pydantic.functional_validators import BeforeValidator as Before

from adjeff.utils._config import _Config, to_arr


class GeoConfig(_Config):
    """Sun and sensor geometry.

    Parameters
    ----------
    sza, vza : xr.DataArray
        Sun and viewing zenith angles [°], in ``[0, 90]``.
    saa, vaa : xr.DataArray
        Sun and viewing azimuth angles [°], in ``[0, 360]``.
    sat_height : float, optional
        Satellite altitude [km], 700 by default.
    """

    sza: Annotated[xr.DataArray, Before(to_arr("sza", ge=0.0, le=90.0))]
    vza: Annotated[xr.DataArray, Before(to_arr("vza", ge=0.0, le=90.0))]
    saa: Annotated[xr.DataArray, Before(to_arr("saa", ge=0.0, le=360.0))]
    vaa: Annotated[xr.DataArray, Before(to_arr("vaa", ge=0.0, le=360.0))]
    sat_height: float = Field(default=700.0, ge=0.0)
