"""Tests for the atmospheric and geometric configuration models."""

from typing import Any

import numpy as np
import pytest
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SurfaceFactory
from adjeff.core import S2Band, random_image_dict
from adjeff.exceptions import ConfigurationError

_VALID_ATMO: dict[str, Any] = dict(
    aot=xr.DataArray(0.2),
    h=xr.DataArray(0.5),
    rh=xr.DataArray(50.0),
    href=xr.DataArray(2.0),
    species={"sulphate": 1.0},
)


_VALID_GEO: dict[str, Any] = dict(
    sza=xr.DataArray(30.0),
    saa=xr.DataArray(45.0),
    vza=xr.DataArray(15.0),
    vaa=xr.DataArray(180.0),
    sat_height=700.0,
)


class TestValidation:
    """What a config refuses at construction."""

    @pytest.mark.parametrize(
        "field,bad_value,match",
        [
            pytest.param("aot", xr.DataArray(-0.1), "'aot'", id="aot_negative"),
            pytest.param("h", xr.DataArray(10.1), "'h'", id="h_above_max"),
            pytest.param("href", xr.DataArray(0.0), "'href'", id="href_zero"),
            pytest.param("rh", xr.DataArray(101.0), "'rh'", id="rh_above_100"),
        ],
    )
    def test_atmo_config_field_out_of_bounds(self, field, bad_value, match):
        with pytest.raises(ValueError, match=match):
            AtmoConfig(**{**_VALID_ATMO, field: bad_value})

    def test_atmo_config_species_sum_raises(self):
        """Species proportions not summing to 1.0 must raise ValueError."""
        with pytest.raises(ValueError):
            AtmoConfig(**{**_VALID_ATMO, "species": {"sulphate": 0.999}})

    @pytest.mark.parametrize(
        "field,bad_value,match",
        [
            pytest.param("sza", xr.DataArray(91.0), "'sza'", id="sza_above_90"),
            pytest.param("saa", xr.DataArray(-1.0), "'saa'", id="saa_negative"),
            pytest.param("vza", xr.DataArray(361.0), "'vza'", id="vza_above_360"),
            pytest.param("vaa", xr.DataArray(-1.0), "'vaa'", id="vaa_negative"),
        ],
    )
    def test_geo_config_field_out_of_bounds(self, field, bad_value, match):
        with pytest.raises(ValueError, match=match):
            GeoConfig(**{**_VALID_GEO, field: bad_value})

    def test_geo_config_sat_height_negative_raises(self):
        """Negative sat_height must raise ValueError."""
        with pytest.raises(ValueError):
            GeoConfig(**{**_VALID_GEO, "sat_height": -1.0})

    def test_parse_params_wrong_ndim_raises(self):
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

    def test_parse_params_size_mismatch_raises(self):
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


# Importing smartg needs SMARTG_DIR_AUXDATA, which is what the
# `integration` marker guards; no GPU is used here.
@pytest.mark.integration
class TestAlbedoMapLevels:
    """How many reflectance levels an arbitrary scene may be quantised to."""

    @staticmethod
    def _band():
        """Return a one-band scene carrying an arbitrary `rho_s`."""
        scene = random_image_dict(
            bands=[S2Band.B03], variables=["rho_s"], res_km=1.0, n=9, seed=0
        )
        return scene[S2Band.B03]

    def test_the_default_is_what_smartg_can_hold(self):
        from smartg.smartg import MAX_NREF

        env = SurfaceFactory().custom_environment(self._band())
        assert len(env.alb.list) == MAX_NREF

    def test_more_levels_than_smartg_holds_is_refused(self):
        from smartg.smartg import MAX_NREF

        with pytest.raises(ConfigurationError, match="n_alb must be"):
            SurfaceFactory().custom_environment(self._band(), n_alb=MAX_NREF + 1)
