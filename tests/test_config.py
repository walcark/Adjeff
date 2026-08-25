"""Test _Config base class methods via AtmoConfig."""

from typing import Any

import xarray as xr

from adjeff.atmosphere import AtmoConfig

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


def test_dataset_returns_dataset():
    """Ensure that the dataset property returns a xr.Dataset instance."""
    cfg = AtmoConfig(**_SCALAR)
    assert isinstance(cfg.dataset, xr.Dataset)


def test_dataset_keys():
    """Ensure thatthe dataset property exposes exactly the AtmoConfig fields."""
    cfg = AtmoConfig(**_SCALAR)
    assert set(cfg.dataset.data_vars) == {"aot", "h", "rh", "href"}


def test_dataset_values():
    """Ensure that the dataset values match the config fields."""
    cfg = AtmoConfig(**_SCALAR)
    xr.testing.assert_equal(cfg.dataset["aot"], cfg.aot)
    xr.testing.assert_equal(cfg.dataset["rh"], cfg.rh)
