"""Tests for analytical and non-analytical PSF modules."""

import numpy as np
import pytest
import torch
import xarray as xr

from adjeff.core._psf import PSFGrid
from adjeff.core.analytical_psf import (
    GeneralizedGaussianPSF,
    GaussPSF,
    KingPSF,
    MoffatGeneralizedPSF,
    VoigtPSF,
)
from adjeff.core.bands import S2Band, SensorBand
from adjeff.core.non_analytical_psf import NonAnalyticalPSF


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid() -> PSFGrid:
    """Minimal valid PSFGrid: 11×11 pixels, 0.01 km resolution."""
    return PSFGrid(res=0.01, n=11)


@pytest.fixture
def band() -> SensorBand:
    return S2Band.B04


# ---------------------------------------------------------------------------
# PSFGrid
# ---------------------------------------------------------------------------


def test_grid_invalid_res():
    """PSFGrid must reject non-positive resolution."""
    with pytest.raises(Exception):
        PSFGrid(res=0.0, n=11)


def test_grid_invalid_n_even():
    """PSFGrid must reject even n."""
    with pytest.raises(Exception):
        PSFGrid(res=0.01, n=10)


def test_grid_invalid_n_too_small():
    """PSFGrid must reject n < 3."""
    with pytest.raises(Exception):
        PSFGrid(res=0.01, n=1)


def test_grid_as_coords_shape(grid):
    """as_coords must return n-length x_psf and y_psf coordinates."""
    coords = grid.as_coords()
    assert len(coords["x_psf"]) == grid.n
    assert len(coords["y_psf"]) == grid.n


def test_grid_as_coords_centered(grid):
    """as_coords coordinates must be centered on zero."""
    coords = grid.as_coords()
    assert coords["x_psf"].values[grid.n // 2] == pytest.approx(0.0, abs=1e-6)
    assert coords["y_psf"].values[grid.n // 2] == pytest.approx(0.0, abs=1e-6)


def test_grid_meshgrid_shape(grid):
    """meshgrid must return two (n, n) float32 tensors."""
    X, Y = grid.meshgrid()
    assert X.shape == (grid.n, grid.n)
    assert Y.shape == (grid.n, grid.n)
    assert X.dtype == torch.float32
    assert Y.dtype == torch.float32


def test_grid_meshgrid_center_is_zero(grid):
    """Center pixel of both meshgrid tensors must be 0."""
    X, Y = grid.meshgrid()
    c = grid.n // 2
    assert X[c, c].item() == pytest.approx(0.0, abs=1e-6)
    assert Y[c, c].item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Helpers shared across all PSF kernels
# ---------------------------------------------------------------------------


def _assert_kernel_valid(kernel: torch.Tensor, n: int) -> None:
    """Check shape, dtype, non-negativity and normalisation."""
    assert kernel.shape == (n, n)
    assert kernel.dtype == torch.float32
    assert (kernel >= 0).all()
    assert kernel.sum().item() == pytest.approx(1.0, abs=1e-5)


def _assert_dataarray_valid(da: xr.DataArray, n: int, band: SensorBand) -> None:
    """Check DataArray shape, dims, coords and mandatory attrs."""
    assert da.shape == (n, n)
    assert list(da.dims) == ["y_psf", "x_psf"]
    assert "x_psf" in da.coords
    assert "y_psf" in da.coords
    assert da.attrs.get("band") is band
    assert "adjeff:kind" in da.attrs


# ---------------------------------------------------------------------------
# GaussPSF
# ---------------------------------------------------------------------------


@pytest.fixture
def gauss_psf(grid, band) -> GaussPSF:
    return GaussPSF(grid=grid, band=band, sigma=1.0)


def test_gauss_forward_shape(gauss_psf, grid):
    """GaussPSF.forward must return a (n, n) float32 tensor."""
    _assert_kernel_valid(gauss_psf.forward(), grid.n)


def test_gauss_forward_peak_at_center(gauss_psf, grid):
    """Gaussian kernel must peak at the center pixel."""
    k = gauss_psf.forward()
    c = grid.n // 2
    assert k[c, c].item() == k.max().item()


def test_gauss_to_dataarray(gauss_psf, grid, band):
    """GaussPSF.to_dataarray must return a valid annotated DataArray."""
    da = gauss_psf.to_dataarray()
    _assert_dataarray_valid(da, grid.n, band)
    assert da.attrs["adjeff:model"] == "Gaussian"


def test_gauss_param_dict(gauss_psf):
    """GaussPSF.param_dict must return a dict with key 'sigma'."""
    p = gauss_psf.param_dict()
    assert set(p.keys()) == {"sigma"}
    assert isinstance(p["sigma"], float)


# ---------------------------------------------------------------------------
# GeneralizedGaussianPSF
# ---------------------------------------------------------------------------


@pytest.fixture
def gauss_general_psf(grid, band) -> GeneralizedGaussianPSF:
    return GeneralizedGaussianPSF(grid=grid, band=band, sigma=1.0, n=0.3)


def test_gauss_general_forward_shape(gauss_general_psf, grid):
    """GeneralizedGaussianPSF.forward must return a valid normalised kernel."""
    _assert_kernel_valid(gauss_general_psf.forward(), grid.n)


def test_gauss_general_forward_peak_at_center(gauss_general_psf, grid):
    """Generalised Gaussian kernel must peak at the center pixel."""
    k = gauss_general_psf.forward()
    c = grid.n // 2
    assert k[c, c].item() == k.max().item()


def test_gauss_general_to_dataarray(gauss_general_psf, grid, band):
    """GeneralizedGaussianPSF.to_dataarray must return a valid annotated DataArray."""
    da = gauss_general_psf.to_dataarray()
    _assert_dataarray_valid(da, grid.n, band)
    assert da.attrs["adjeff:model"] == "GeneralizedGaussian"


def test_gauss_general_param_dict(gauss_general_psf):
    """GeneralizedGaussianPSF.param_dict must return keys 'sigma' and 'n'."""
    p = gauss_general_psf.param_dict()
    assert set(p.keys()) == {"sigma", "n"}
    assert all(isinstance(v, float) for v in p.values())


# ---------------------------------------------------------------------------
# VoigtPSF
# ---------------------------------------------------------------------------


@pytest.fixture
def voigt_psf(grid, band) -> VoigtPSF:
    return VoigtPSF(grid=grid, band=band, sigma=1.0, gamma=1.0)


def test_voigt_forward_shape(voigt_psf, grid):
    """VoigtPSF.forward must return a valid normalised kernel."""
    _assert_kernel_valid(voigt_psf.forward(), grid.n)


def test_voigt_forward_peak_at_center(voigt_psf, grid):
    """Voigt kernel must peak at the center pixel."""
    k = voigt_psf.forward()
    c = grid.n // 2
    assert k[c, c].item() == k.max().item()


def test_voigt_to_dataarray(voigt_psf, grid, band):
    """VoigtPSF.to_dataarray must return a valid annotated DataArray."""
    da = voigt_psf.to_dataarray()
    _assert_dataarray_valid(da, grid.n, band)
    assert da.attrs["adjeff:model"] == "Voigt"


def test_voigt_param_dict(voigt_psf):
    """VoigtPSF.param_dict must return keys 'sigma' and 'gamma'."""
    p = voigt_psf.param_dict()
    assert set(p.keys()) == {"sigma", "gamma"}
    assert all(isinstance(v, float) for v in p.values())


def test_voigt_eta_range(voigt_psf):
    """_eta must be in [0, 1]."""
    eta = voigt_psf._eta().item()
    assert 0.0 <= eta <= 1.0


# ---------------------------------------------------------------------------
# KingPSF
# ---------------------------------------------------------------------------


@pytest.fixture
def king_psf(grid, band) -> KingPSF:
    return KingPSF(grid=grid, band=band, sigma=1.0, gamma=2.0)


def test_king_forward_shape(king_psf, grid):
    """KingPSF.forward must return a valid normalised kernel."""
    _assert_kernel_valid(king_psf.forward(), grid.n)


def test_king_forward_peak_at_center(king_psf, grid):
    """King kernel must peak at the center pixel."""
    k = king_psf.forward()
    c = grid.n // 2
    assert k[c, c].item() == k.max().item()


def test_king_to_dataarray(king_psf, grid, band):
    """KingPSF.to_dataarray must return a valid annotated DataArray."""
    da = king_psf.to_dataarray()
    _assert_dataarray_valid(da, grid.n, band)
    assert da.attrs["adjeff:model"] == "King"


def test_king_param_dict(king_psf):
    """KingPSF.param_dict must return keys 'sigma' and 'gamma'."""
    p = king_psf.param_dict()
    assert set(p.keys()) == {"sigma", "gamma"}
    assert all(isinstance(v, float) for v in p.values())


# ---------------------------------------------------------------------------
# MoffatGeneralizedPSF
# ---------------------------------------------------------------------------


@pytest.fixture
def moffat_psf(grid, band) -> MoffatGeneralizedPSF:
    return MoffatGeneralizedPSF(grid=grid, band=band, alpha=1.0, beta=1.0, gamma=1.0,)


def test_moffat_forward_shape(moffat_psf, grid):
    """MoffatGeneralizedPSF.forward must return a valid normalised kernel."""
    _assert_kernel_valid(moffat_psf.forward(), grid.n)


def test_moffat_forward_peak_at_center(moffat_psf, grid):
    """Moffat kernel must peak at the center pixel."""
    k = moffat_psf.forward()
    c = grid.n // 2
    assert k[c, c].item() == k.max().item()


def test_moffat_to_dataarray(moffat_psf, grid, band):
    """MoffatGeneralizedPSF.to_dataarray must return a valid annotated DataArray."""
    da = moffat_psf.to_dataarray()
    _assert_dataarray_valid(da, grid.n, band)
    assert da.attrs["adjeff:model"] == "MoffatGeneralized"


def test_moffat_param_dict(moffat_psf):
    """MoffatGeneralizedPSF.param_dict must return keys 'alpha', 'beta', 'gamma'."""
    p = moffat_psf.param_dict()
    assert set(p.keys()) == {"alpha", "beta", "gamma"}
    assert all(isinstance(v, float) for v in p.values())


# ---------------------------------------------------------------------------
# NonAnalyticalPSF
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_kernel(grid) -> np.ndarray:
    """Uniform (n, n) kernel — sums to 1 after normalisation."""
    return np.ones((grid.n, grid.n), dtype=np.float32)


@pytest.fixture
def non_analytical_psf(grid, band, flat_kernel) -> NonAnalyticalPSF:
    return NonAnalyticalPSF(grid=grid, band=band, kernel=flat_kernel)


def test_non_analytical_forward_shape(non_analytical_psf, grid):
    """NonAnalyticalPSF.forward must return a valid normalised kernel."""
    _assert_kernel_valid(non_analytical_psf.forward(), grid.n)


def test_non_analytical_forward_uniform(non_analytical_psf, grid):
    """Uniform input kernel must remain uniform after normalisation."""
    k = non_analytical_psf.forward()
    expected = 1.0 / (grid.n * grid.n)
    assert torch.allclose(k, torch.full_like(k, expected), atol=1e-6)


def test_non_analytical_accepts_tensor(grid, band):
    """NonAnalyticalPSF must accept a torch.Tensor kernel."""
    k = torch.ones(grid.n, grid.n)
    psf = NonAnalyticalPSF(grid=grid, band=band, kernel=k)
    _assert_kernel_valid(psf.forward(), grid.n)


def test_non_analytical_wrong_shape_raises(grid, band):
    """NonAnalyticalPSF must raise when kernel shape mismatches the grid."""
    bad_kernel = np.ones((5, 5), dtype=np.float32)
    with pytest.raises(Exception):
        NonAnalyticalPSF(grid=grid, band=band, kernel=bad_kernel)


def test_non_analytical_to_dataarray(non_analytical_psf, grid, band):
    """NonAnalyticalPSF.to_dataarray must return a valid annotated DataArray."""
    da = non_analytical_psf.to_dataarray()
    _assert_dataarray_valid(da, grid.n, band)
    assert da.attrs["adjeff:kind"] == "non_analytical"
    assert da.attrs.get("adjeff:source") == "SmartG"


def test_non_analytical_custom_source(grid, band, flat_kernel):
    """Custom source tag must appear in attrs."""
    psf = NonAnalyticalPSF(grid=grid, band=band, kernel=flat_kernel, source="MyTool")
    assert psf.to_dataarray().attrs["adjeff:source"] == "MyTool"


def test_non_analytical_no_grad(non_analytical_psf):
    """NonAnalyticalPSF kernel must not require gradients."""
    k = non_analytical_psf.forward()
    assert not k.requires_grad


def test_non_analytical_param_dict_empty(non_analytical_psf):
    """NonAnalyticalPSF.param_dict must return an empty dict."""
    assert non_analytical_psf.param_dict() == {}
