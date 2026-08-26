"""Tests for the radial binning helpers."""

import math

import numpy as np
import pytest
import torch
import xarray as xr

from adjeff.utils.radial import (
    RadialBinning,
    annulus_areas,
    cumulate,
    edges_from_centres,
    natural_npix,
    radial_distances,
)

# Fixtures


@pytest.fixture
def simple_ds():
    """5x5 Dataset with x/y coords centered on zero."""
    x = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    data = np.ones((5, 5), dtype=np.float32)
    return xr.Dataset(
        {"img": xr.DataArray(data, dims=["y", "x"], coords={"x": x, "y": y})}
    )


@pytest.fixture
def psf_ds():
    """5x5 Dataset with x_psf/y_psf dims (PSF convention)."""
    x = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
    data = np.ones((5, 5), dtype=np.float32)
    return xr.Dataset(
        {
            "psf": xr.DataArray(
                data, dims=["y_psf", "x_psf"], coords={"x_psf": x, "y_psf": y}
            )
        }
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


# RadialBinning


def test_binning_counts_sum_to_npixels():
    """Total count across all bins must equal the number of input pixels."""
    rr = torch.linspace(0.0, 5.0, 25)
    assert RadialBinning(rr, 5).counts.sum().item() == 25


def test_binning_centres_start_at_the_origin():
    """First bin centre must be exactly 0, unlike its true midpoint."""
    binning = RadialBinning(torch.linspace(0.0, 4.0, 20), 4)
    assert binning.centres[0].item() == 0.0
    assert binning.midpoints[0].item() > 0.0


def test_binning_sum_is_uniform_when_the_values_are():
    """For uniform values, sum[b] must equal counts[b] * value."""
    value = 3.0
    binning = RadialBinning(torch.linspace(0.0, 4.0, 20), 4)
    total = binning.sum(torch.full((20,), value))
    filled = binning.filled
    assert torch.allclose(total[filled], binning.counts[filled] * value)


def test_binning_output_shapes():
    """All the arrays the binning exposes must have the expected shapes."""
    n, n_bins = 30, 6
    binning = RadialBinning(torch.rand(n), n_bins)
    assert binning.edges.shape == (n_bins + 1,)
    assert binning.index.shape == (n,)
    assert binning.counts.shape == (n_bins,)
    assert binning.centres.shape == (n_bins,)
    assert binning.area.shape == (n_bins,)
    assert binning.sum(torch.rand(n)).shape == (n_bins,)


def test_binning_mean_fills_empty_bins():
    """A bin no pixel landed in gets the fill value, not a division by zero."""
    rr = torch.tensor([0.0, 0.05, 1.0])
    binning = RadialBinning(rr, 10)
    mean = binning.mean(torch.ones(3))
    assert torch.isnan(mean[binning.counts == 0]).all()
    assert torch.allclose(mean[binning.filled], torch.ones(int(binning.filled.sum())))
    assert (binning.mean(torch.ones(3), fill=0.0)[~binning.filled] == 0.0).all()


def test_binning_std_matches_numpy():
    """The per-bin standard deviation must be the population one."""
    rr = torch.linspace(0.0, 4.0, 400)
    vv = torch.rand(400)
    binning = RadialBinning(rr, 4)
    std = binning.std(vv)
    for b in range(4):
        picked = vv[binning.index == b]
        assert std[b].item() == pytest.approx(
            float(picked.std(unbiased=False)), rel=1e-4
        )


def test_binning_on_a_square_grid_covers_its_diagonal():
    """A grid binning must reach the largest radius its pixels span."""
    binning = RadialBinning.on_square_grid(64, 10.0)
    assert binning.counts.sum().item() == 64 * 64
    assert float(binning.edges[-1]) == pytest.approx(32 * 10.0 * np.sqrt(2), rel=1e-5)


def test_binning_cdf_of_a_flat_field_grows_as_the_area():
    """Integrating a constant must give the disc area, normalised."""
    binning = RadialBinning.on_square_grid(64, 1.0)
    cdf = binning.cdf(torch.ones(64 * 64))
    assert cdf.shape == (binning.n_bins + 1,)
    assert cdf[0].item() == 0.0
    assert cdf[-1].item() == pytest.approx(1.0)
    half = binning.radius_at(cdf, 0.5)
    edge = float(binning.edges[-1])
    assert half == pytest.approx(edge / np.sqrt(2), rel=0.02)


def test_annulus_areas_tile_the_disc():
    """The areas must telescope to the disc their edges span."""
    edges = torch.linspace(0.0, 10.0, 11)
    area = annulus_areas(edges)
    assert float(area.sum()) == pytest.approx(np.pi * 10.0**2, rel=1e-5)
    assert (area > 0).all()


def test_edges_from_centres_never_goes_negative():
    """A reconstructed inner edge is clamped at zero, radii being positive."""
    edges = edges_from_centres(torch.linspace(0.0, 10.0, 11))
    assert float(edges[0]) == 0.0
    assert float(edges[-1]) == pytest.approx(10.5)
    assert (edges[1:] > edges[:-1]).all()


def test_cumulate_starts_at_zero_and_closes_at_one():
    """The cumulated energy is read at the edges, one point more than bins."""
    edges = torch.linspace(0.0, 10.0, 11)
    cdf = cumulate(torch.ones(10), edges)
    assert cdf.shape == (11,)
    assert float(cdf[0]) == 0.0
    assert float(cdf[-1]) == pytest.approx(1.0)
    # A constant field cumulates as the area, hence as the radius squared.
    expected = (edges / edges[-1]) ** 2
    assert torch.allclose(cdf, expected, atol=1e-6)
