import numpy as np
import pytest
import torch
import xarray as xr

import adjeff  # noqa: F401 — registers the accessor


@pytest.fixture
def flat_ds() -> xr.Dataset:
    """Return a 10x10 constant Dataset on a centered grid."""
    x = np.linspace(-4.5, 4.5, 10, dtype=np.float32)
    y = np.linspace(-4.5, 4.5, 10, dtype=np.float32)
    data = np.full((10, 10), 2.0, dtype=np.float32)
    return xr.Dataset({"img": xr.DataArray(data, dims=["y", "x"], coords={"x": x, "y": y})})


@pytest.fixture
def annotated_ds(flat_ds: xr.Dataset) -> xr.Dataset:
    """Add adjeff attrs to the Dataset variable."""
    flat_ds["img"].attrs.update(
        {
            "adjeff:kind": "analytical",
            "adjeff:model": "gaussian",
            "adjeff:params": {"sigma": 1.5},
        }
    )
    return flat_ds


def test_to_tensor_dtype(flat_ds):
    """Ensure that to_tensor returns a float32 torch.Tensor."""
    t = flat_ds.adjeff.to_tensor("img")
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32


def test_to_tensor_values(flat_ds):
    """ensure that to_tensor values match the original array."""
    t = flat_ds.adjeff.to_tensor("img")
    expected = torch.from_numpy(flat_ds["img"].values.astype(np.float32))
    assert torch.allclose(t, expected)


def test_from_tensor_roundtrip(flat_ds):
    """Ensure that from_tensor write a DataArray with matching shape and coords."""
    t = flat_ds.adjeff.to_tensor("img")
    flat_ds.adjeff.from_tensor(t * 2, like="img", target="img2")
    assert "img2" in flat_ds
    assert flat_ds["img2"].shape == flat_ds["img"].shape
    assert list(flat_ds["img2"].dims) == list(flat_ds["img"].dims)


def test_apply_mask_nans_masked_pixels(flat_ds):
    """Ensure that masked pixels become NaN across all variables."""
    mask = xr.DataArray(
        np.zeros((10, 10), dtype=bool), dims=["y", "x"]
    )
    mask[3, 4] = True
    flat_ds.adjeff.apply_mask(mask)
    assert np.isnan(flat_ds["img"].values[3, 4])


def test_apply_mask_unmasked_pixels_unchanged(flat_ds):
    """Ensure that unmasked pixels keep their original value."""
    mask = xr.DataArray(np.zeros((10, 10), dtype=bool), dims=["y", "x"])
    mask[0, 0] = True
    flat_ds.adjeff.apply_mask(mask)
    assert flat_ds["img"].values[5, 5] == pytest.approx(2.0)


def test_kind(annotated_ds):
    """Ensure that kind() returns the adjeff:kind attribute."""
    assert annotated_ds.adjeff.kind("img") == "analytical"


def test_kind_missing(flat_ds):
    """Ensure that kind() returns None when the attribute is absent."""
    assert flat_ds.adjeff.kind("img") is None


def test_is_analytical(annotated_ds):
    """Ensure that is_analytical() return True for kind 'analytical'."""
    assert annotated_ds.adjeff.is_analytical("img") is True


def test_model(annotated_ds):
    """Ensure that model() returns the adjeff:model attribute."""
    assert annotated_ds.adjeff.model("img") == "gaussian"


def test_params(annotated_ds):
    """Ensure that params() returns the adjeff:params dict."""
    assert annotated_ds.adjeff.params("img") == {"sigma": 1.5}


def test_radial_returns_dataarray(flat_ds):
    """Ensure that radial() returns a 1-D DataArray with a 'r' dimension."""
    result = flat_ds.adjeff.radial("img")
    assert isinstance(result, xr.DataArray)
    assert "r" in result.dims


def test_radial_constant_field_is_flat(flat_ds):
    """Ensure that a constant field produces a flat radial profile."""
    result = flat_ds.adjeff.radial("img")
    assert not np.any(np.isnan(result.values))
    assert np.allclose(result.values, 2.0, atol=1e-5)


def test_radial_upsample(flat_ds):
    """Ensure that requesting too much bins upsamples via interpolation."""
    result = flat_ds.adjeff.radial("img", n_bins=100)
    assert result.sizes["r"] == 100


def test_radial_cdf_normalized(flat_ds):
    """Ensure that radial_cdf with normalize=True ends at 1.0."""
    cdf = flat_ds.adjeff.radial_cdf("img")
    assert cdf.values[-1] == pytest.approx(1.0, abs=1e-5)


def test_radial_cdf_monotone(flat_ds):
    """Ensure that radial_cdf is non-decreasing."""
    cdf = flat_ds.adjeff.radial_cdf("img")
    diffs = np.diff(cdf.values)
    assert np.all(diffs >= -1e-6)


def test_radial_std_nonnegative(flat_ds):
    """Ensure that radial_std values are >= 0 (ignoring NaN)."""
    result = flat_ds.adjeff.radial_std("img")
    values = result.values[~np.isnan(result.values)]
    assert np.all(values >= 0.0)


def test_radial_std_constant_field_is_zero(flat_ds):
    """Ensure that a constant field has zero azimuthal dispersion."""
    result = flat_ds.adjeff.radial_std("img")
    values = result.values[~np.isnan(result.values)]
    assert np.allclose(values, 0.0, atol=1e-5)
