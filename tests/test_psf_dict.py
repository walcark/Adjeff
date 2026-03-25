"""Tests for PSFDict."""

import pytest
import xarray as xr

from adjeff.core import (
    GaussPSF, 
    KingPSF, 
    PSFDict, 
    PSFGrid, 
    S2Band, 
    SensorBand, 
    random_image_dict
)
from adjeff.exceptions import ConfigurationError


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
def king_b03(grid) -> KingPSF:
    return KingPSF(grid=grid, band=S2Band.B03, sigma=1.0, gamma=2.0)


@pytest.fixture
def psf_dict(gauss_b02, gauss_b03) -> PSFDict:
    return PSFDict([gauss_b02, gauss_b03])


@pytest.fixture
def scene():
    return random_image_dict([S2Band.B02, S2Band.B03], ["rho_s"], n=16)


@pytest.fixture
def psf_grids() -> dict[SensorBand, PSFGrid]:
    grids: dict[SensorBand, PSFGrid] = {
        S2Band.B02: PSFGrid(res=0.01, n=11),
        S2Band.B03: PSFGrid(res=0.01, n=11),
    }
    return grids


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_single_psf(gauss_b02):
    """PSFDict can be constructed with a single PSFModule."""
    psf_dict = PSFDict([gauss_b02])
    assert S2Band.B02 in psf_dict


def test_init_multiple_bands(gauss_b02, gauss_b03):
    """PSFDict accepts multiple PSFModules of the same type on different bands."""
    psf_dict = PSFDict([gauss_b02, gauss_b03])
    assert S2Band.B02 in psf_dict
    assert S2Band.B03 in psf_dict


def test_init_mixed_types_raises(gauss_b02, king_b03):
    """PSFDict must raise ConfigurationError when PSF types are mixed."""
    with pytest.raises(ConfigurationError):
        PSFDict([gauss_b02, king_b03])


def test_init_duplicate_band_raises(gauss_b02, grid):
    """PSFDict must raise ConfigurationError when the same band appears twice."""
    gauss_b02_bis = GaussPSF(grid=grid, band=S2Band.B02, sigma=2.0)
    with pytest.raises(ConfigurationError):
        PSFDict([gauss_b02, gauss_b02_bis])


def test_init_stores_kernel_variable(gauss_b02):
    """Each band Dataset must contain a 'kernel' variable."""
    psf_dict = PSFDict([gauss_b02])
    assert "kernel" in psf_dict[S2Band.B02].data_vars


# ---------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------


def test_bands_sorted(psf_dict):
    """Bands must be sorted by wavelength."""
    assert psf_dict.bands == [S2Band.B02, S2Band.B03]


def test_bands_sorted_regardless_of_input_order(grid):
    """Bands must be sorted even when PSFModules are passed in reverse order."""
    psf_b03 = GaussPSF(grid=grid, band=S2Band.B03, sigma=1.0)
    psf_b02 = GaussPSF(grid=grid, band=S2Band.B02, sigma=1.0)
    assert PSFDict([psf_b03, psf_b02]).bands == [S2Band.B02, S2Band.B03]


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
# training_input
# ---------------------------------------------------------------------------


def test_training_input_bands(scene, psf_grids):
    """training_input must produce one PSF per band present in the scene."""
    psf_dict = PSFDict.training_input(
        scene=scene,
        psf_type=GaussPSF,
        psf_init_params={"sigma": 1.0},
        psf_grids=psf_grids,
    )
    assert psf_dict.bands == scene.bands


def test_training_input_has_kernel_variable(scene, psf_grids):
    """Each band Dataset from training_input must contain a 'kernel' variable."""
    psf_dict = PSFDict.training_input(
        scene=scene,
        psf_type=GaussPSF,
        psf_init_params={"sigma": 1.0},
        psf_grids=psf_grids,
    )
    for band in psf_dict.bands:
        assert "kernel" in psf_dict[band].data_vars


def test_training_input_missing_grid_raises(scene):
    """training_input must raise ConfigurationError when a band grid is missing."""
    incomplete_grids: dict[SensorBand, PSFGrid] = {S2Band.B02: PSFGrid(res=0.01, n=11)}
    with pytest.raises(ConfigurationError):
        PSFDict.training_input(
            scene=scene,
            psf_type=GaussPSF,
            psf_init_params={"sigma": 1.0},
            psf_grids=incomplete_grids,
        )
