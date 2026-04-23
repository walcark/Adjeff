import numpy as np
import pytest
import xarray as xr

import adjeff  # noqa: F401 — registers the accessor


@pytest.fixture
def flat_da() -> xr.DataArray:
    """Return a 10x10 constant DataArray on a centered grid."""
    x = np.linspace(-4.5, 4.5, 10, dtype=np.float32)
    y = np.linspace(-4.5, 4.5, 10, dtype=np.float32)
    data = np.full((10, 10), 2.0, dtype=np.float32)
    return xr.DataArray(data, dims=["y", "x"], coords={"x": x, "y": y})


@pytest.fixture
def disk_da() -> xr.DataArray:
    """Return a 51x51 DataArray with a high-reflectance disk at center."""
    x = np.linspace(-25.0, 25.0, 51, dtype=np.float32)
    y = np.linspace(-25.0, 25.0, 51, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    data = np.where(xx**2 + yy**2 < 10.0**2, 0.8, 0.1).astype(np.float32)
    return xr.DataArray(data, dims=["y", "x"], coords={"x": x, "y": y})


@pytest.fixture
def annotated_da(flat_da: xr.DataArray) -> xr.DataArray:
    """Return flat_da with adjeff metadata attrs."""
    flat_da.attrs.update(
        {
            "adjeff:kind": "analytical",
            "adjeff:model": "gaussian",
            "adjeff:params": {"sigma": 1.5},
        }
    )
    return flat_da


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------


def test_kind(annotated_da):
    """kind() returns the adjeff:kind attribute."""
    assert annotated_da.adjeff.kind() == "analytical"


def test_kind_missing(flat_da):
    """kind() returns None when the attribute is absent."""
    assert flat_da.adjeff.kind() is None


def test_is_analytical_true(annotated_da):
    """is_analytical() returns True when kind is 'analytical'."""
    assert annotated_da.adjeff.is_analytical() is True


def test_is_analytical_false(flat_da):
    """is_analytical() returns False when kind attr is absent."""
    assert flat_da.adjeff.is_analytical() is False


def test_model(annotated_da):
    """model() returns the adjeff:model attribute."""
    assert annotated_da.adjeff.model() == "gaussian"


def test_model_missing(flat_da):
    """model() returns None when the attribute is absent."""
    assert flat_da.adjeff.model() is None


def test_params(annotated_da):
    """params() returns the adjeff:params dict."""
    assert annotated_da.adjeff.params() == {"sigma": 1.5}


def test_params_missing(flat_da):
    """params() returns None when the attribute is absent."""
    assert flat_da.adjeff.params() is None


# ------------------------------------------------------------------
# Spatial properties
# ------------------------------------------------------------------


def test_res(flat_da):
    """res returns the pixel spacing derived from x coordinates."""
    assert flat_da.adjeff.res == pytest.approx(1.0, rel=1e-5)


def test_n(flat_da):
    """n returns the number of pixels on the x dimension."""
    assert flat_da.adjeff.n == 10


# ------------------------------------------------------------------
# Radial profile
# ------------------------------------------------------------------


def test_radial_returns_dataarray(flat_da):
    """radial() returns a 1-D DataArray with dim 'r'."""
    result = flat_da.adjeff.radial()
    assert isinstance(result, xr.DataArray)
    assert "r" in result.dims


def test_radial_constant_field_is_flat(flat_da):
    """A constant field produces a flat radial profile."""
    result = flat_da.adjeff.radial()
    assert not np.any(np.isnan(result.values))
    assert np.allclose(result.values, 2.0, atol=1e-5)


def test_radial_upsample(flat_da):
    """Requesting more bins than natural upsamples via interpolation."""
    result = flat_da.adjeff.radial(n_bins=100)
    assert result.sizes["r"] == 100


def test_radial_cdf_normalized(flat_da):
    """radial_cdf with normalize=True ends at 1.0."""
    cdf = flat_da.adjeff.radial("cdf")
    assert cdf.values[-1] == pytest.approx(1.0, abs=1e-5)


def test_radial_cdf_monotone(flat_da):
    """radial_cdf is non-decreasing."""
    cdf = flat_da.adjeff.radial("cdf")
    diffs = np.diff(cdf.values)
    assert np.all(diffs >= -1e-6)


def test_radial_std_nonnegative(flat_da):
    """radial_std values are >= 0 (ignoring NaN)."""
    result = flat_da.adjeff.radial("std")
    values = result.values[~np.isnan(result.values)]
    assert np.all(values >= 0.0)


def test_radial_std_constant_field_is_zero(flat_da):
    """A constant field has zero azimuthal dispersion."""
    result = flat_da.adjeff.radial("std")
    values = result.values[~np.isnan(result.values)]
    assert np.allclose(values, 0.0, atol=1e-5)


# ------------------------------------------------------------------
# Adaptive radial sampling
# ------------------------------------------------------------------


def test_radial_adaptive_count(disk_da):
    """radial_adaptive returns exactly n samples before gap-filling."""
    result = disk_da.adjeff.radial("adaptive", n=20)
    assert isinstance(result, xr.DataArray)
    assert "r" in result.dims
    assert result.sizes["r"] >= 20


def test_radial_adaptive_concentrates_near_edge(disk_da):
    """Adaptive sampling places more points near the disk edge (r~10)."""
    result = disk_da.adjeff.radial("adaptive", n=30)
    r = result.coords["r"].values
    # Density near the disk edge (8–12) should be higher than far from it
    near_edge = np.sum((r >= 8) & (r <= 12))
    far = np.sum(r > 20)
    assert near_edge > far


def test_radial_adaptive_max_gap(disk_da):
    """max_gap ensures no two consecutive samples exceed the given distance."""
    max_gap = 3.0
    result = disk_da.adjeff.radial("adaptive", n=10, max_gap=max_gap)
    r = result.coords["r"].values
    gaps = np.diff(r)
    assert np.all(gaps <= max_gap + 1e-6)


def test_radial_adaptive_from_profile(disk_da):
    """radial_adaptive on a 1-D 'r' DataArray skips recomputing azimuthal mean."""
    profile = disk_da.adjeff.radial()
    result = profile.adjeff.radial("adaptive", n=15)
    assert "r" in result.dims
    assert result.sizes["r"] >= 15


# ------------------------------------------------------------------
# Profile → 2D field reconstruction
# ------------------------------------------------------------------


def test_to_field_shape(disk_da):
    """to_field reconstructs a 2D field with the target grid shape."""
    profile = disk_da.adjeff.radial()
    target_ds = disk_da.to_dataset(name="rho_s")
    result = profile.adjeff.to_field(target_ds)
    assert result.sizes["y"] == disk_da.sizes["y"]
    assert result.sizes["x"] == disk_da.sizes["x"]


def test_to_field_center_value(disk_da):
    """to_field value at center (r=0) matches the profile value at r=0."""
    profile = disk_da.adjeff.radial()
    target_ds = disk_da.to_dataset(name="rho_s")
    result = profile.adjeff.to_field(target_ds)
    center_y = disk_da.sizes["y"] // 2
    center_x = disk_da.sizes["x"] // 2
    # Center pixel should be close to the high-reflectance disk value
    assert result.values[center_y, center_x] == pytest.approx(0.8, abs=0.05)


def test_to_field_broadcasts_extra_dims(disk_da):
    """to_field broadcasts correctly when the profile has extra dimensions."""
    profile = disk_da.adjeff.radial()
    # Add an extra dimension (e.g. aot) by stacking
    profile_extra = xr.concat(
        [profile * 1.0, profile * 0.5], dim=xr.DataArray([0.1, 0.3], dims=["aot"])
    )
    target_ds = disk_da.to_dataset(name="rho_s")
    result = profile_extra.adjeff.to_field(target_ds)
    assert result.sizes["aot"] == 2
    assert result.sizes["y"] == disk_da.sizes["y"]
    assert result.sizes["x"] == disk_da.sizes["x"]
