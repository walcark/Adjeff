"""Tests for the frozen PSF tree and its helpers."""

import numpy as np
import pytest
import xarray as xr

from adjeff.core import (
    GaussPSF,
    KingPSF,
    PSFGrid,
    S2Band,
    freeze,
    psf_kernel,
    psf_params,
    psf_tree,
)
from adjeff.core.psf_tree import tree_band_ids, write_band


@pytest.fixture
def modules():
    """Return live PSF modules on two bands with different grids."""
    return {
        S2Band.B02: GaussPSF(PSFGrid(0.01, 11), S2Band.B02, sigma=0.3),
        S2Band.B8A: KingPSF(
            PSFGrid(0.02, 5), S2Band.B8A, sigma=0.1, gamma=1.0
        ),
    }


@pytest.fixture
def swept_tree():
    """Return a tree whose kernel and parameter carry an `aot` dimension."""
    kernel = xr.DataArray(
        np.ones((2, 5, 5)) / 25.0,
        dims=["aot", "y_psf", "x_psf"],
        coords={"aot": [0.1, 0.4]},
    )
    sigma = xr.DataArray([0.2, 0.5], dims=["aot"], coords={"aot": [0.1, 0.4]})
    return psf_tree({S2Band.B03: kernel}, params={S2Band.B03: {"sigma": sigma}})


# --- Layout ---


def test_freeze_keeps_one_grid_per_band(modules):
    """Bands may hold different grids; a tree imposes no alignment."""
    tree = freeze(modules)

    assert tree_band_ids(tree) == ["B02", "B8A"]
    assert psf_kernel(tree, S2Band.B02).shape == (11, 11)
    assert psf_kernel(tree, S2Band.B8A).shape == (5, 5)


def test_kernels_are_normalised(modules):
    """Every analytical PSF integrates to one."""
    tree = freeze(modules)

    for band in modules:
        assert float(psf_kernel(tree, band).sum()) == pytest.approx(1.0)


def test_missing_band_names_what_the_tree_holds(modules):
    """The error must say which bands are available."""
    tree = freeze(modules)

    with pytest.raises(KeyError, match="B02, B8A"):
        psf_kernel(tree, S2Band.B12)


# --- Parameters ---


def test_params_carry_their_sweep_dimensions(swept_tree):
    """A per-combo optimisation stores one value per combo."""
    params = psf_params(swept_tree, S2Band.B03)

    assert set(params) == {"sigma"}
    assert params["sigma"].dims == ("aot",)
    assert params["sigma"].sel(aot=0.4) == pytest.approx(0.5)


def test_params_exclude_the_kernel(swept_tree):
    """The kernel is not a parameter."""
    assert "kernel" not in psf_params(swept_tree, S2Band.B03)


def test_params_are_empty_without_an_optimisation(modules):
    """freeze() alone records no fitted parameters."""
    assert psf_params(freeze(modules), S2Band.B02) == {}


# --- Persistence ---


def test_zarr_round_trip_preserves_heterogeneous_grids(modules, tmp_path):
    """A tree survives zarr, which a single Dataset could not.

    The two bands hold 11x11 and 5x5 kernels on different pixel sizes,
    so they cannot share a Dataset at all.
    """
    tree = freeze(modules)
    tree.to_zarr(tmp_path / "psf.zarr", mode="w")

    back = xr.open_datatree(tmp_path / "psf.zarr", engine="zarr")

    assert tree_band_ids(back) == ["B02", "B8A"]
    for band in modules:
        np.testing.assert_allclose(
            psf_kernel(back, band).values, psf_kernel(tree, band).values
        )


def test_band_attribute_survives_serialisation(modules, tmp_path):
    """The band is stored as its id, which zarr can write.

    Holding the SensorBand itself made `to_zarr` raise, since an enum is
    not JSON serialisable.
    """
    tree = freeze(modules)
    tree.to_zarr(tmp_path / "psf.zarr", mode="w")

    back = xr.open_datatree(tmp_path / "psf.zarr", engine="zarr")
    assert psf_kernel(back, S2Band.B02).attrs["band"] == "B02"


def test_write_band_chunks_one_combo_at_a_time(swept_tree, tmp_path):
    """Spatial dims stay whole so a single combo reads cheaply."""
    dest = tmp_path / "B03"
    write_band(dest, swept_tree["B03"].ds)

    back = xr.open_zarr(dest)
    assert back["kernel"].chunksizes["aot"] == (1, 1)
    assert back["kernel"].chunksizes["y_psf"] == (5,)
