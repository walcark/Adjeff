"""Test _Config base class methods via AtmoConfig."""

from typing import Any

import xarray as xr

from adjeff.api import load_config
from adjeff.atmosphere import AtmoConfig
from adjeff.core import S2Band, gaussian_image_dict

_SCALAR: dict[str, Any] = dict(
    aot=xr.DataArray(0.2),
    h=xr.DataArray(0.5),
    rh=xr.DataArray(50.0),
    href=xr.DataArray(2.0),
    species={"sulphate": 1.0},
)


def _pix_conf(aot: list[float], h: list[float] | None = None) -> AtmoConfig:
    """Instanciate AtmoConfig with a 'pixel' dimension on aot (and optionally h)."""
    n = len(aot)
    return AtmoConfig(
        **{
            **_SCALAR,
            "aot": xr.DataArray(aot, dims=["pixel"]),
            "h": xr.DataArray(h if h is not None else [0.5] * n, dims=["pixel"]),
        }
    )


class TestDatasetView:
    """A config seen as a dataset."""

    def test_dataset_returns_dataset(self):
        cfg = AtmoConfig(**_SCALAR)
        assert isinstance(cfg.dataset, xr.Dataset)

    def test_dataset_keys(self):
        cfg = AtmoConfig(**_SCALAR)
        assert set(cfg.dataset.data_vars) == {"aot", "h", "rh", "href"}

    def test_dataset_values(self):
        cfg = AtmoConfig(**_SCALAR)
        xr.testing.assert_equal(cfg.dataset["aot"], cfg.aot)
        xr.testing.assert_equal(cfg.dataset["rh"], cfg.rh)


class TestAggregatedConfig:
    """A config built from reduced fields, as `fit_psf` builds one."""

    @staticmethod
    def _scene():
        """Return a scene carrying the eight fields `load_config` reads."""
        scene = gaussian_image_dict(
            sigma=2.0,
            res_km=1.0,
            rho_min=0.0,
            rho_max=0.5,
            bands=[S2Band.B03],
            n=9,
        )
        for name, value in (
            ("aot", 0.3),
            ("rh", 50.0),
            ("h", 0.0),
            ("href", 2.0),
            ("sza", 40.0),
            ("vza", 0.0),
            ("saa", 0.0),
            ("vaa", 0.0),
        ):
            scene[S2Band.B03][name] = xr.DataArray(value)
        return scene

    def test_a_mean_keeps_the_dimension_the_samplers_allocate_from(self):
        cfg = load_config(self._scene(), S2Band.B03, aggregate=True)
        for name in ("aot", "rh", "h", "href"):
            assert getattr(cfg["atmo_config"], name).dims == (name,)
        for name in ("sza", "vza", "saa", "vaa"):
            assert getattr(cfg["geo_config"], name).dims == (name,)
