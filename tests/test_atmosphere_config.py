from typing import Any

import numpy as np
import pytest
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig
from adjeff.exceptions import ConfigurationError
from conftest import requires_cuda

_VALID_ATMO: dict[str, Any] = dict(
    aot=xr.DataArray(0.2),
    h=xr.DataArray(0.5),
    rh=xr.DataArray(50.0),
    href=xr.DataArray(2.0),
    species={"sulphate": 1.0},
)


@pytest.mark.parametrize("field,bad_value,match", [
    pytest.param("aot",  xr.DataArray(-0.1),  "'aot'",  id="aot_negative"),
    pytest.param("h",    xr.DataArray(10.1),  "'h'",    id="h_above_max"),
    pytest.param("href", xr.DataArray(0.0),   "'href'", id="href_zero"),
    pytest.param("rh",   xr.DataArray(101.0), "'rh'",   id="rh_above_100"),
])
def test_atmo_config_field_out_of_bounds(field, bad_value, match):
    """Each atmospheric field must raise ValueError when out of bounds."""
    with pytest.raises(ValueError, match=match):
        AtmoConfig(**{**_VALID_ATMO, field: bad_value})


def test_atmo_config_species_sum_raises():
    """Species proportions not summing to 1.0 must raise ValueError."""
    with pytest.raises(ValueError):
        AtmoConfig(**{**_VALID_ATMO, "species": {"sulphate": 0.999}})


_VALID_GEO: dict[str, Any] = dict(
    sza=xr.DataArray(30.0),
    saa=xr.DataArray(45.0),
    vza=xr.DataArray(15.0),
    vaa=xr.DataArray(180.0),
    sat_height=700.0
)


@pytest.mark.parametrize("field,bad_value,match", [
    pytest.param("sza", xr.DataArray(91.0),  "'sza'", id="sza_above_90"),
    pytest.param("saa", xr.DataArray(-1.0),  "'saa'", id="saa_negative"),
    pytest.param("vza", xr.DataArray(361.0), "'vza'", id="vza_above_360"),
    pytest.param("vaa", xr.DataArray(-1.0),  "'vaa'", id="vaa_negative"),
])
def test_geo_config_field_out_of_bounds(field, bad_value, match):
    """Each geometric field must raise ValueError when out of bounds."""
    with pytest.raises(ValueError, match=match):
        GeoConfig(**{**_VALID_GEO, field: bad_value})


def test_geo_config_sat_height_negative_raises():
    """Negative sat_height must raise ValueError."""
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


# ---------------------------------------------------------------------------
# atmo_factory — parse_params
# ---------------------------------------------------------------------------


def test_parse_params_wrong_ndim_raises():
    """parse_params raises ConfigurationError when a parameter has ndim != 1."""
    from adjeff.atmosphere.atmo_factory import parse_params

    da_2d = xr.DataArray(np.ones((2, 2)), dims=["x", "y"])
    with pytest.raises(ConfigurationError, match="exactly one dimension"):
        parse_params(
            {
                "wl": da_2d,
                "aot": xr.DataArray([0.1, 0.2], dims=["index"]),
                "rh": xr.DataArray([50.0, 70.0], dims=["index"]),
                "h": xr.DataArray([0.0, 0.0], dims=["index"]),
                "href": xr.DataArray([2.0, 2.0], dims=["index"]),
            }
        )


def test_parse_params_size_mismatch_raises():
    """parse_params raises ConfigurationError when parameter sizes differ."""
    from adjeff.atmosphere.atmo_factory import parse_params

    with pytest.raises(ConfigurationError):
        parse_params(
            {
                "wl": xr.DataArray([440.0, 560.0], dims=["index"]),
                "aot": xr.DataArray([0.1, 0.2, 0.3], dims=["index"]),
                "rh": xr.DataArray([50.0, 70.0], dims=["index"]),
                "h": xr.DataArray([0.0, 0.0], dims=["index"]),
                "href": xr.DataArray([2.0, 2.0], dims=["index"]),
            }
        )
