import math

import numpy as np
import pytest
import torch
import xarray as xr

from adjeff.utils.radial import bin_radial, natural_npix, radial_distances


# Fixtures


@pytest.fixture
def simple_ds():
    """5x5 Dataset with x/y coords centered on zero."""
    x = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    data = np.ones((5, 5), dtype=np.float32)
    return xr.Dataset({"img": xr.DataArray(data, dims=["y", "x"], coords={"x": x, "y": y})})


@pytest.fixture
def psf_ds():
    """5x5 Dataset with x_psf/y_psf dims (PSF convention)."""
    x = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    data = np.ones((5, 5), dtype=np.float32)
    return xr.Dataset(
        {"psf": xr.DataArray(data, dims=["y_psf", "x_psf"], coords={"x_psf": x, "y_psf": y})}
    )


# radial_distances


def test_radial_distances_shape(simple_ds):
    """Output arrays must be flat with length equal to the number of pixels."""
    rr, vv = radial_distances(simple_ds, "img", center=None)
    assert rr.shape == (25,)
    assert vv.shape == (25,)


def test_radial_distances_dtype(simple_ds):
    """Output arrays must be float32."""
    rr, vv = radial_distances(simple_ds, "img", center=None)
    assert rr.dtype == np.float32
    assert vv.dtype == np.float32


def test_radial_distances_origin_pixel_is_zero(simple_ds):
    """The pixel at (cx, cy) must have radial distance zero."""
    rr, _ = radial_distances(simple_ds, "img", center=(0.0, 0.0))
    assert np.isclose(rr.min(), 0.0)


def test_radial_distances_custom_center(simple_ds):
    """Shifting the center must shift all distances accordingly."""
    rr_default, _ = radial_distances(simple_ds, "img", center=None)
    rr_shifted, _ = radial_distances(simple_ds, "img", center=(1.0, 1.0))
    assert not np.allclose(rr_default, rr_shifted)


def test_radial_distances_psf_dims(psf_ds):
    """PSF dims must default center to (0, 0) and return the right shape."""
    rr, vv = radial_distances(psf_ds, "psf", center=None)
    assert rr.shape == (25,)
    assert np.isclose(rr.min(), 0.0)


# natural_npix


def test_natural_npix_square():
    """Check formula for a 10x10 variable: int(9/sqrt(2)) - 1 = 5."""
    x = np.linspace(0, 1, 10)
    data = np.ones((10, 10), dtype=np.float32)
    ds = xr.Dataset({"v": xr.DataArray(data, dims=["y", "x"], coords={"x": x, "y": x})})
    expected = max(int(9 / math.sqrt(2)) - 1, 2)
    assert natural_npix(ds, "v") == expected


def test_natural_npix_minimum():
    """Very small arrays must return at least 2."""
    x = np.array([0.0, 1.0])
    data = np.ones((2, 2), dtype=np.float32)
    ds = xr.Dataset({"v": xr.DataArray(data, dims=["y", "x"], coords={"x": x, "y": x})})
    assert natural_npix(ds, "v") >= 2


def test_natural_npix_uses_shortest_side():
    """For a non-square array the shortest side drives the bin count."""
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 10)
    data = np.ones((10, 20), dtype=np.float32)
    ds = xr.Dataset({"v": xr.DataArray(data, dims=["y", "x"], coords={"x": x, "y": y})})
    expected = max(int(9 / math.sqrt(2)) - 1, 2)
    assert natural_npix(ds, "v") == expected


# bin_radial


def test_bin_radial_counts_sum_to_npixels():
    """Total count across all bins must equal the number of input pixels."""
    rr = torch.linspace(0.0, 5.0, 25)
    vv = torch.ones(25)
    _, _, counts, _, _ = bin_radial(rr, vv, npix=5)
    assert counts.sum().item() == 25


def test_bin_radial_r_centers_origin():
    """First bin centre must be exactly 0."""
    rr = torch.linspace(0.0, 4.0, 20)
    vv = torch.ones(20)
    _, _, _, _, r_centers = bin_radial(rr, vv, npix=4)
    assert r_centers[0].item() == 0.0


def test_bin_radial_sum_vals_uniform():
    """For uniform values, sum_vals[b] must equal counts[b] * value."""
    value = 3.0
    rr = torch.linspace(0.0, 4.0, 20)
    vv = torch.full((20,), value)
    _, _, counts, sum_vals, _ = bin_radial(rr, vv, npix=4)
    mask = counts > 0
    assert torch.allclose(sum_vals[mask], counts[mask] * value)


def test_bin_radial_output_shapes():
    """All returned tensors must have the expected shapes."""
    n, npix = 30, 6
    rr = torch.rand(n)
    vv = torch.rand(n)
    bins, inds, counts, sum_vals, r_centers = bin_radial(rr, vv, npix=npix)
    assert bins.shape == (npix + 1,)
    assert inds.shape == (n,)
    assert counts.shape == (npix,)
    assert sum_vals.shape == (npix,)
    assert r_centers.shape == (npix,)
