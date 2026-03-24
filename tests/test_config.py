"""Test _Config base class methods via AtmoConfig."""

from typing import Any

import numpy as np
import pytest
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


def test_iter_missing_dim_yields_self():
    """Ensure that iterating on an absent dimension yield self."""
    cfg = AtmoConfig(**_SCALAR)
    batches = list(cfg.iter(n_batch=10, dim="pixel"))
    assert len(batches) == 1
    assert batches[0] is cfg


def test_iter_preserves_type():
    """Ensure that iterating on a dimension preserves the type."""
    cfg = _pix_conf([0.1, 0.2, 0.3, 0.4])
    for batch in cfg.iter(n_batch=2, dim="pixel"):
        assert isinstance(batch, AtmoConfig)


def test_iter_covers_all_pixels():
    """Ensure that concatenate batches reconstruct the original aot."""
    cfg = _pix_conf([0.1, 0.2, 0.3, 0.4])
    batches = list(cfg.iter(n_batch=2, dim="pixel"))
    reconstructed = xr.concat([b.aot for b in batches], dim="pixel")
    xr.testing.assert_equal(reconstructed, cfg.aot)


def test_iter_scalars_broadcast_to_all_batches():
    """Ensure that scalar fields (no 'pixel' dim) are unchanged in every batch."""
    cfg = _pix_conf([0.1, 0.2, 0.3, 0.4])
    for batch in cfg.iter(n_batch=2, dim="pixel"):
        xr.testing.assert_equal(batch.rh, cfg.rh)


@pytest.mark.parametrize("n_batch,expected_n_batches", [
    (1, 4),   # chunk_size=1 → one batch per pixel
    (2, 2),   # chunk_size=2 → two batches
    (4, 1),   # chunk_size=4 → single batch
    (100, 1), # chunk_size > total → single batch
])
def test_iter_batch_count(n_batch: int, expected_n_batches: int) -> None:
    """Ensure that iter produces the expected number of batches."""
    cfg = _pix_conf([0.1, 0.2, 0.3, 0.4])
    assert len(list(cfg.iter(n_batch=n_batch, dim="pixel"))) == expected_n_batches


def test_unique_returns_correct_type():
    """Ensure that unique return the correction type AtmoConfig."""
    cfg = _pix_conf([0.1, 0.2, 0.1])
    unique_cfg, _ = cfg.unique(dims=["pixel"])
    assert isinstance(unique_cfg, AtmoConfig)


def test_unique_all_distinct():
    """Ensure that unique with all-distinct rows keeps all of them."""
    cfg = _pix_conf([0.1, 0.2, 0.3], h=[0.1, 0.2, 0.3])
    unique_cfg, inverse_map = cfg.unique(dims=["pixel"])
    assert unique_cfg.aot.sizes["index"] == 3
    assert inverse_map.sizes["pixel"] == 3


def test_unique_compresses_repetitions():
    """Ensure that unique merges duplicate (aot, h) pairs."""
    cfg = _pix_conf([0.1, 0.2, 0.1], h=[0.5, 0.5, 0.5])
    unique_cfg, inverse_map = cfg.unique(dims=["pixel"])
    assert unique_cfg.aot.sizes["index"] == 2
    assert inverse_map.sizes["pixel"] == 3


def test_unique_inverse_map_reconstructs_aot():
    """Ensure that inverse_map allows exact reconstruction of the original aot."""
    aot_vals = [0.1, 0.2, 0.1, 0.2, 0.3]
    cfg = _pix_conf(aot_vals)
    unique_cfg, inverse_map = cfg.unique(dims=["pixel"])
    reconstructed = unique_cfg.aot.isel(index=inverse_map)
    np.testing.assert_array_almost_equal(reconstructed.values, aot_vals)


def test_unique_scalar_fields_unchanged():
    """Ensure that Fields without the target dim pass through unchanged."""
    cfg = _pix_conf([0.1, 0.2, 0.1])
    unique_cfg, _ = cfg.unique(dims=["pixel"])
    xr.testing.assert_equal(unique_cfg.rh, cfg.rh)


def test_run_applies_fn_and_concatenates():
    """Ensure run applies fn to each batch and concatenate along dim."""
    cfg = _pix_conf([0.1, 0.2, 0.3, 0.4])

    def double_aot(batch) -> xr.DataArray:
        return batch.aot * 2.0

    result = cfg.run(double_aot, n_batch=2, dim="pixel")
    xr.testing.assert_allclose(result, cfg.aot * 2.0)


def test_run_result_covers_full_dim():
    """Ensure run result has the same pixel dimension as the input."""
    cfg = _pix_conf([0.1, 0.2, 0.3, 0.4])

    def identity(batch) -> xr.DataArray:
        return batch.aot

    result = cfg.run(identity, n_batch=1, dim="pixel")
    assert result.sizes["pixel"] == 4
