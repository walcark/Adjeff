"""Tests for PSFDict."""

import numpy as np
import pytest
import xarray as xr

from adjeff.core import (
    GaussPSF,
    KingPSF,
    PSFDict,
    PSFGrid,
    S2Band,
    SensorBand,
    init_psf_dict,
    random_image_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid() -> PSFGrid:
    return PSFGrid(res=0.01, n=11)


@pytest.fixture
def gauss_b02(grid) -> GaussPSF:
    return GaussPSF(grid=grid, band=S2Band.B02, sigma=1.0)


@pytest.fixture
def gauss_b03(grid) -> GaussPSF:
    return GaussPSF(grid=grid, band=S2Band.B03, sigma=1.0)


@pytest.fixture
def psf_dict(gauss_b02, gauss_b03) -> PSFDict:
    return PSFDict.from_kernels(
        {
            S2Band.B02: gauss_b02.to_dataarray(),
            S2Band.B03: gauss_b03.to_dataarray(),
        }
    )


@pytest.fixture
def psf_grids() -> dict[SensorBand, PSFGrid]:
    grids: dict[SensorBand, PSFGrid] = {
        S2Band.B02: PSFGrid(res=0.01, n=11),
        S2Band.B03: PSFGrid(res=0.01, n=11),
    }
    return grids


# ---------------------------------------------------------------------------
# from_kernels
# ---------------------------------------------------------------------------


def test_from_kernels_single_band(gauss_b02):
    """from_kernels can be constructed with a single band."""
    psf_dict = PSFDict.from_kernels({S2Band.B02: gauss_b02.to_dataarray()})
    assert S2Band.B02 in psf_dict


def test_from_kernels_multiple_bands(gauss_b02, gauss_b03):
    """from_kernels accepts multiple bands."""
    psf_dict = PSFDict.from_kernels(
        {
            S2Band.B02: gauss_b02.to_dataarray(),
            S2Band.B03: gauss_b03.to_dataarray(),
        }
    )
    assert S2Band.B02 in psf_dict
    assert S2Band.B03 in psf_dict


def test_from_kernels_stores_kernel_variable(gauss_b02):
    """Each band Dataset must contain a 'kernel' variable."""
    psf_dict = PSFDict.from_kernels({S2Band.B02: gauss_b02.to_dataarray()})
    assert "kernel" in psf_dict[S2Band.B02].data_vars


# ---------------------------------------------------------------------------
# from_modules / is_trainable / get_module / to_frozen
# ---------------------------------------------------------------------------


def test_from_modules_is_trainable(grid):
    """from_modules produces a trainable PSFDict."""
    psf = GaussPSF(grid=grid, band=S2Band.B02, sigma=1.0)
    psf_dict = PSFDict.from_modules({S2Band.B02: psf})
    assert psf_dict.is_trainable


def test_from_kernels_is_not_trainable(gauss_b02):
    """from_kernels produces a frozen (non-trainable) PSFDict."""
    psf_dict = PSFDict.from_kernels({S2Band.B02: gauss_b02.to_dataarray()})
    assert not psf_dict.is_trainable


def test_get_module_returns_psf(grid):
    """get_module returns the original PSFModule."""
    psf = GaussPSF(grid=grid, band=S2Band.B02, sigma=1.0)
    psf_dict = PSFDict.from_modules({S2Band.B02: psf})
    assert psf_dict.get_module(S2Band.B02) is psf


def test_get_module_raises_on_frozen(gauss_b02):
    """get_module raises RuntimeError on a frozen PSFDict."""
    psf_dict = PSFDict.from_kernels({S2Band.B02: gauss_b02.to_dataarray()})
    with pytest.raises(RuntimeError):
        psf_dict.get_module(S2Band.B02)


def test_to_frozen_produces_frozen(grid):
    """to_frozen converts a trainable PSFDict to a frozen one."""
    psf = GaussPSF(grid=grid, band=S2Band.B02, sigma=1.0)
    trainable = PSFDict.from_modules({S2Band.B02: psf})
    frozen = trainable.to_frozen()
    assert not frozen.is_trainable
    assert S2Band.B02 in frozen


def test_to_frozen_idempotent_on_frozen(gauss_b02):
    """to_frozen on a frozen PSFDict returns the same object."""
    psf_dict = PSFDict.from_kernels({S2Band.B02: gauss_b02.to_dataarray()})
    assert psf_dict.to_frozen() is psf_dict


def test_to_dataarray_raises_on_trainable(grid):
    """to_dataarray raises RuntimeError in trainable mode."""
    psf = GaussPSF(grid=grid, band=S2Band.B02, sigma=1.0)
    psf_dict = PSFDict.from_modules({S2Band.B02: psf})
    with pytest.raises(RuntimeError):
        psf_dict.to_dataarray(S2Band.B02)


# ---------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------


def test_bands_sorted(psf_dict):
    """Bands must be sorted by wavelength."""
    assert psf_dict.bands == [S2Band.B02, S2Band.B03]


def test_bands_sorted_regardless_of_input_order(grid):
    """Bands must be sorted even when passed in reverse order."""
    psf_b03 = GaussPSF(grid=grid, band=S2Band.B03, sigma=1.0)
    psf_b02 = GaussPSF(grid=grid, band=S2Band.B02, sigma=1.0)
    psf_dict = PSFDict.from_kernels(
        {
            S2Band.B03: psf_b03.to_dataarray(),
            S2Band.B02: psf_b02.to_dataarray(),
        }
    )
    assert psf_dict.bands == [S2Band.B02, S2Band.B03]


# ---------------------------------------------------------------------------
# __getitem__ / __setitem__ / __contains__
# ---------------------------------------------------------------------------


def test_getitem_returns_dataset(psf_dict):
    """__getitem__ must return an xr.Dataset."""
    assert isinstance(psf_dict[S2Band.B02], xr.Dataset)


def test_setitem(psf_dict, gauss_b02):
    """__setitem__ must replace the Dataset for a given band."""
    new_ds = xr.Dataset({"kernel": gauss_b02.to_dataarray()})
    psf_dict[S2Band.B02] = new_ds
    assert psf_dict[S2Band.B02] is new_ds


def test_contains_present_band(psf_dict):
    """__contains__ returns True for a band that is present."""
    assert S2Band.B02 in psf_dict


def test_contains_absent_band(psf_dict):
    """__contains__ returns False for a band that is absent."""
    assert S2Band.B08 not in psf_dict


# ---------------------------------------------------------------------------
# to_dataarray / kernel
# ---------------------------------------------------------------------------


def test_to_dataarray_returns_dataarray(psf_dict):
    """to_dataarray must return an xr.DataArray."""
    assert isinstance(psf_dict.to_dataarray(S2Band.B02), xr.DataArray)


def test_to_dataarray_correct_dims(psf_dict):
    """to_dataarray must return a DataArray with y_psf and x_psf dims."""
    assert list(psf_dict.to_dataarray(S2Band.B02).dims) == ["y_psf", "x_psf"]


def test_kernel_is_alias_of_to_dataarray(psf_dict):
    """kernel() must return the same DataArray as to_dataarray()."""
    assert psf_dict.kernel(S2Band.B02).equals(psf_dict.to_dataarray(S2Band.B02))


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_contains_class_name(psf_dict):
    """__repr__ must return a string that mentions PSFDict."""
    assert "PSFDict" in repr(psf_dict)


# ---------------------------------------------------------------------------
# init_psf_dict
# ---------------------------------------------------------------------------


def test_init_psf_dict_bands(psf_grids):
    """init_psf_dict produces one PSF per band in grids."""
    psf_dict = init_psf_dict(
        grids=psf_grids,
        model=GaussPSF,
        init_parameters={"sigma": 1.0},
    )
    assert psf_dict.bands == [S2Band.B02, S2Band.B03]


def test_init_psf_dict_is_trainable(psf_grids):
    """init_psf_dict produces a trainable PSFDict."""
    psf_dict = init_psf_dict(
        grids=psf_grids,
        model=GaussPSF,
        init_parameters={"sigma": 1.0},
    )
    assert psf_dict.is_trainable


def test_init_psf_dict_per_band_params(psf_grids):
    """init_psf_dict accepts per-band parameter dicts."""
    psf_dict = init_psf_dict(
        grids=psf_grids,
        model=GaussPSF,
        init_parameters={S2Band.B02: {"sigma": 0.5}, S2Band.B03: {"sigma": 2.0}},
    )
    assert psf_dict.get_module(S2Band.B02).param_dict()["sigma"] == pytest.approx(
        0.5, rel=1e-3
    )
    assert psf_dict.get_module(S2Band.B03).param_dict()["sigma"] == pytest.approx(
        2.0, rel=1e-3
    )


def test_init_psf_dict_to_frozen_has_kernel(psf_grids):
    """to_frozen on init_psf_dict result contains a kernel variable."""
    psf_dict = init_psf_dict(
        grids=psf_grids,
        model=GaussPSF,
        init_parameters={"sigma": 1.0},
    )
    frozen = psf_dict.to_frozen()
    for band in frozen.bands:
        assert "kernel" in frozen[band].data_vars


# ---------------------------------------------------------------------------
# params
# ---------------------------------------------------------------------------


def test_params_single_combo_reads_attrs(gauss_b02):
    """params() reads from kernel attrs for a single-combo PSFDict."""
    psf_dict = PSFDict.from_kernels({S2Band.B02: gauss_b02.to_dataarray()})
    p = psf_dict.params(S2Band.B02)
    assert p is not None
    assert "sigma" in p
    assert isinstance(p["sigma"], float)


def test_params_non_analytical_returns_none(grid):
    """params() returns None for a NonAnalyticalPSF (no adjeff:params in attrs)."""
    from adjeff.core.non_analytical_psf import NonAnalyticalPSF

    psf = NonAnalyticalPSF(
        grid=grid, band=S2Band.B02, kernel=np.ones((grid.n, grid.n))
    )
    psf_dict = PSFDict.__new__(PSFDict)
    psf_dict._data = {S2Band.B02: xr.Dataset({"kernel": psf.to_dataarray()})}
    assert psf_dict.params(S2Band.B02) is None


def test_params_multi_combo_reads_dataset_variables():
    """params() reads param_* Dataset variables for multi-combo PSFDicts."""
    grid = PSFGrid(res=0.01, n=11)
    kernel_da = xr.DataArray(
        np.ones((2, grid.n, grid.n), dtype="float32"),
        dims=["aot", "y_psf", "x_psf"],
        coords={"aot": [0.1, 0.3], **grid.as_coords()},
    )
    sigma_da = xr.DataArray([0.5, 0.7], dims=["aot"], coords={"aot": [0.1, 0.3]})
    psf_dict = PSFDict.from_kernels(
        {S2Band.B02: kernel_da},
        params={S2Band.B02: {"sigma": sigma_da}},
    )
    p = psf_dict.params(S2Band.B02)
    assert p is not None
    assert "sigma" in p
    assert isinstance(p["sigma"], xr.DataArray)
    assert list(p["sigma"].dims) == ["aot"]


def test_from_kernels_with_params_stores_variable():
    """from_kernels with params must add a param_sigma variable to the Dataset."""
    grid = PSFGrid(res=0.01, n=11)
    kernel_da = xr.DataArray(
        np.ones((grid.n, grid.n), dtype="float32"),
        dims=["y_psf", "x_psf"],
        coords=grid.as_coords(),
    )
    sigma_da = xr.DataArray(0.5)
    psf_dict = PSFDict.from_kernels(
        {S2Band.B02: kernel_da},
        params={S2Band.B02: {"sigma": sigma_da}},
    )
    assert "param_sigma" in psf_dict[S2Band.B02].data_vars
