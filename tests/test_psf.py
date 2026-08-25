"""Tests for analytical and non-analytical PSF modules."""

import numpy as np
import pytest
import torch
import xarray as xr

from adjeff.core._psf import PSFGrid
from adjeff.core.analytical_psf import (
    GaussPSF,
    GeneralizedGaussianPSF,
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
    """Return the band every PSF fixture is built on."""
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
    """Meshgrid must return two (n, n) float32 tensors."""
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
    assert da.attrs.get("band") == band.id
    assert "adjeff:kind" in da.attrs


# ---------------------------------------------------------------------------
# Analytical PSF invariants — parametrized over all 5 models
# ---------------------------------------------------------------------------

_ANALYTICAL_PSF_CASES = pytest.mark.parametrize(
    "psf_cls,kwargs,model_name,param_keys",
    [
        pytest.param(
            GaussPSF,
            {"sigma": 1.0},
            "Gaussian",
            {"sigma"},
            id="Gauss",
        ),
        pytest.param(
            GeneralizedGaussianPSF,
            {"sigma": 1.0, "n": 0.3},
            "GeneralizedGaussian",
            {"sigma", "n"},
            id="GeneralizedGaussian",
        ),
        pytest.param(
            VoigtPSF,
            {"sigma": 1.0, "gamma": 1.0},
            "Voigt",
            {"sigma", "gamma"},
            id="Voigt",
        ),
        pytest.param(
            KingPSF,
            {"sigma": 1.0, "gamma": 2.0},
            "King",
            {"sigma", "gamma"},
            id="King",
        ),
        pytest.param(
            MoffatGeneralizedPSF,
            {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
            "MoffatGeneralized",
            {"alpha", "beta", "gamma"},
            id="MoffatGeneralized",
        ),
    ],
)


@_ANALYTICAL_PSF_CASES
def test_analytical_psf_kernel_valid(
    psf_cls, kwargs, model_name, param_keys, grid, band
):
    """Every analytical PSF returns a normalised float32 (n, n) kernel."""
    psf = psf_cls(grid=grid, band=band, **kwargs)
    _assert_kernel_valid(psf.forward(), grid.n)


@_ANALYTICAL_PSF_CASES
def test_analytical_psf_peak_at_center(
    psf_cls, kwargs, model_name, param_keys, grid, band
):
    """Every radially-symmetric PSF must peak at the center pixel."""
    psf = psf_cls(grid=grid, band=band, **kwargs)
    k = psf.forward()
    c = grid.n // 2
    assert k[c, c].item() == k.max().item()


@_ANALYTICAL_PSF_CASES
def test_analytical_psf_to_dataarray(
    psf_cls, kwargs, model_name, param_keys, grid, band
):
    """to_dataarray returns an annotated DataArray naming its model."""
    psf = psf_cls(grid=grid, band=band, **kwargs)
    da = psf.to_dataarray()
    _assert_dataarray_valid(da, grid.n, band)
    assert da.attrs["adjeff:model"] == model_name


@_ANALYTICAL_PSF_CASES
def test_analytical_psf_param_dict(psf_cls, kwargs, model_name, param_keys, grid, band):
    """param_dict must return float values keyed by the expected parameter names."""
    psf = psf_cls(grid=grid, band=band, **kwargs)
    p = psf.param_dict()
    assert set(p.keys()) == param_keys
    assert all(isinstance(v, float) for v in p.values())


# ---------------------------------------------------------------------------
# VoigtPSF specific
# ---------------------------------------------------------------------------


def test_voigt_eta_range(grid, band):
    """_eta must return a value in [0, 1]."""
    psf = VoigtPSF(grid=grid, band=band, sigma=1.0, gamma=1.0)
    assert 0.0 <= psf._eta().item() <= 1.0


# ---------------------------------------------------------------------------
# NonAnalyticalPSF
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_kernel(grid) -> np.ndarray:
    """Uniform (n, n) kernel — sums to 1 after normalisation."""
    return np.ones((grid.n, grid.n), dtype=np.float32)


@pytest.fixture
def non_analytical_psf(grid, band, flat_kernel) -> NonAnalyticalPSF:
    """Return a PSF wrapping a fixed, non-trainable kernel."""
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
    assert not non_analytical_psf.forward().requires_grad


def test_non_analytical_param_dict_empty(non_analytical_psf):
    """NonAnalyticalPSF.param_dict must return an empty dict."""
    assert non_analytical_psf.param_dict() == {}


# ---------------------------------------------------------------------------
# KingPSF — the power-law index stays in the integrable range
# ---------------------------------------------------------------------------


def test_king_gamma_is_confined_to_the_integrable_range(grid, band):
    """A King kernel with gamma below one has no scale of its own.

    Its radial integral diverges as ``R^(2-2g)``, so the grid rather than
    the profile sets the normalisation, and the loss surface turns
    concave there, which stalls a quasi-Newton step.
    """
    low, high = KingPSF.GAMMA_BOUNDS

    assert KingPSF(grid, band, sigma=0.3, gamma=0.2).param_dict()[
        "gamma"
    ] >= low
    assert KingPSF(grid, band, sigma=0.3, gamma=50.0).param_dict()[
        "gamma"
    ] <= high


def test_king_gamma_stays_bounded_under_gradient_steps(grid, band):
    """The bound must survive optimisation, not only construction."""
    import torch

    psf = KingPSF(grid, band, sigma=0.3, gamma=1.2)
    optimiser = torch.optim.SGD(psf.parameters(), lr=1e3)

    for _ in range(20):
        optimiser.zero_grad()
        # Push hard towards a flat kernel, which wants a small gamma.
        psf.forward().max().backward()
        optimiser.step()

    low, high = KingPSF.GAMMA_BOUNDS
    assert low <= psf.param_dict()["gamma"] <= high


# ---------------------------------------------------------------------------
# SensorBand — reverse wavelength lookup
# ---------------------------------------------------------------------------


def test_band_from_wavelength():
    """A figure names a band by its wavelength, so the enum should too."""
    from adjeff.core import S2Band

    assert S2Band.from_wl(560.0) is S2Band.B03
    assert S2Band.from_wl(665.0) is S2Band.B04


def test_band_from_wavelength_rejects_an_unknown_centre():
    """An approximate wavelength is an error, not a nearest neighbour."""
    from adjeff.core import S2Band

    with pytest.raises(KeyError, match="560"):
        S2Band.from_wl(600.0)
