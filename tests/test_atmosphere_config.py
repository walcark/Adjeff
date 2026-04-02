from typing import Any

import pytest
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig
from conftest import requires_cuda

_VALID_ATMO: dict[str, Any] = dict(
    aot=xr.DataArray(0.2),
    h=xr.DataArray(0.5),
    rh=xr.DataArray(50.0),
    href=xr.DataArray(2.0),
    species={"sulphate": 1.0},
)


def test_atmo_config_wrong_input():
    """Ensure that ge/le are respected for each atmospheric parameter."""
    with pytest.raises(ValueError, match="'aot'"):
        AtmoConfig(**{**_VALID_ATMO, "aot": xr.DataArray(-0.1)})

    with pytest.raises(ValueError, match="'h'"):
        AtmoConfig(**{**_VALID_ATMO, "h": xr.DataArray(10.1)})

    with pytest.raises(ValueError, match="'href'"):
        AtmoConfig(**{**_VALID_ATMO, "href": xr.DataArray(0.0)})

    with pytest.raises(ValueError, match="'rh'"):
        AtmoConfig(**{**_VALID_ATMO, "rh": xr.DataArray(101.0)})
    
    with pytest.raises(ValueError):
        AtmoConfig(**{**_VALID_ATMO, "species": {"sulphate": 0.999}})


_VALID_GEO: dict[str, Any] = dict(
    sza=xr.DataArray(30.0),
    saa=xr.DataArray(45.0),
    vza=xr.DataArray(15.0),
    vaa=xr.DataArray(180.0),
    sat_height=700.0
)


def test_geo_config_wrong_input():
    """Ensure that ge/le are respected for each geometric parameter."""
    with pytest.raises(ValueError, match="'sza'"):
        GeoConfig(**{**_VALID_GEO, "sza": xr.DataArray(91.0)})

    with pytest.raises(ValueError, match="'saa'"):
        GeoConfig(**{**_VALID_GEO, "saa": xr.DataArray(-1.0)})

    with pytest.raises(ValueError, match="'vza'"):
        GeoConfig(**{**_VALID_GEO, "vza": xr.DataArray(361.0)})

    with pytest.raises(ValueError, match="'vaa'"):
        GeoConfig(**{**_VALID_GEO, "vaa": xr.DataArray(-1.0)})

    with pytest.raises(ValueError):
        GeoConfig(**{**_VALID_GEO, "sat_height": -1.0})


def test_geo_config_sun_le():
    """sun_le should expose sza as th_deg and saa as phi_deg."""
    geo = GeoConfig(**_VALID_GEO)
    assert geo.sun_le == {"th_deg": 30.0, "phi_deg": 45.0, "zip": True}


def test_geo_config_sat_le():
    """sat_le should expose vza as th_deg and saa as phi_deg."""
    geo = GeoConfig(**_VALID_GEO)
    assert geo.sat_le == {"th_deg": 15.0, "phi_deg": 45.0, "zip": True}


@requires_cuda
def test_geo_config_sun_sensor():
    """sun_sensor should return a Sensor instance."""
    from smartg.smartg import Sensor
    assert isinstance(GeoConfig(**_VALID_GEO).sun_sensor, Sensor)


@requires_cuda
def test_geo_config_sat_sensor():
    """sat_sensor should return a Sensor instance."""
    from smartg.smartg import Sensor
    assert isinstance(GeoConfig(**_VALID_GEO).sat_sensor, Sensor)


@pytest.mark.parametrize("vza,vaa,expected_x,expected_y", [
    (0.0,   0.0,   0.0,    0.0),   # nadir: no horizontal offset
    (45.0, 180.0,  700.0,  0.0),   # south: positive x
    (45.0,   0.0, -700.0,  0.0),   # north: negative x
    (45.0,  90.0,  0.0,  700.0),   # east: positive y
    (45.0, 270.0,  0.0, -700.0),   # west: negative y
])
def test_geo_config_satellite_relative_position(
    vza: float, 
    vaa: float, 
    expected_x: float, 
    expected_y: float,
) -> None:
    """Physical cases for satellite_relative_position at sat_height=700km."""
    geo = GeoConfig(**{**_VALID_GEO, "vza": xr.DataArray(vza), "vaa": xr.DataArray(vaa)})
    x, y = geo.satellite_relative_position
    assert x == pytest.approx(expected_x, abs=1e-4)
    assert y == pytest.approx(expected_y, abs=1e-4)
