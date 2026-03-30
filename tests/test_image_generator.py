"""Tests for image generator functions and _resolve_n utility."""

import numpy as np
import pytest

from adjeff.core import S2Band, disk_image_dict, gaussian_image_dict, random_image_dict
from adjeff.core.image_generator import _resolve_n

RES_B02 = 0.01  # km (10 m)
RES_B01 = 0.06  # km (60 m)


# --- _resolve_n ---


def test_resolve_n_scalar_n():
    """Apply a scalar n uniformly to all bands."""
    result = _resolve_n([S2Band.B02, S2Band.B01], res_km=RES_B02, n=100, extent_km=None)
    assert result == {S2Band.B02: 100, S2Band.B01: 100}


def test_resolve_n_dict_n():
    """Pass through a per-band n mapping unchanged."""
    mapping = {S2Band.B02: 50, S2Band.B01: 200}
    result = _resolve_n([S2Band.B02, S2Band.B01], res_km=RES_B02, n=mapping, extent_km=None)
    assert result == mapping


def test_resolve_n_scalar_extent_km():
    """Derive n from a scalar extent_km using the provided resolution."""
    result = _resolve_n([S2Band.B02], res_km=RES_B02, n=None, extent_km=1.0)
    assert result[S2Band.B02] == round(1.0 / RES_B02)


def test_resolve_n_dict_extent_km():
    """Derive n per band from a per-band extent_km mapping."""
    extents = {S2Band.B02: 1.0, S2Band.B01: 6.0}
    res = {S2Band.B02: RES_B02, S2Band.B01: RES_B01}
    result = _resolve_n([S2Band.B02, S2Band.B01], res_km=res, n=None, extent_km=extents)
    assert result[S2Band.B02] == round(1.0 / RES_B02)
    assert result[S2Band.B01] == round(6.0 / RES_B01)


def test_resolve_n_both_raises():
    """Raise ValueError when both n and extent_km are provided."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_n([S2Band.B02], res_km=RES_B02, n=10, extent_km=1.0)


def test_resolve_n_none_raises():
    """Raise ValueError when neither n nor extent_km is provided."""
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_n([S2Band.B02], res_km=RES_B02, n=None, extent_km=None)


# --- gaussian_image_dict ---


def test_gaussian_image_dict_shape():
    """Gaussian generator produces an n×n DataArray per band."""
    scene = gaussian_image_dict(sigma=1.0, res_km=RES_B02, n=32, bands=[S2Band.B02])
    assert scene[S2Band.B02]["rho_s"].shape == (32, 32)


def test_gaussian_image_dict_extent_km_shape():
    """Gaussian generator with extent_km produces a shape consistent with band resolution."""
    extent = 1.0
    scene = gaussian_image_dict(sigma=1.0, res_km=RES_B02, extent_km=extent, bands=[S2Band.B02])
    expected_n = round(extent / RES_B02)
    assert scene[S2Band.B02]["rho_s"].shape == (expected_n, expected_n)


def test_gaussian_image_dict_multiband_shape():
    """Gaussian generator with extent_km produces different n for bands with different resolution."""
    res = {S2Band.B02: RES_B02, S2Band.B01: RES_B01}
    scene = gaussian_image_dict(
        sigma=1.0, res_km=res, extent_km=1.0, bands=[S2Band.B02, S2Band.B01]
    )
    n_b02 = round(1.0 / RES_B02)
    n_b01 = round(1.0 / RES_B01)
    assert scene[S2Band.B02]["rho_s"].shape == (n_b02, n_b02)
    assert scene[S2Band.B01]["rho_s"].shape == (n_b01, n_b01)


def test_gaussian_center_equals_rho_max():
    """Gaussian generator peak value at center equals rho_max for very large sigma."""
    scene = gaussian_image_dict(sigma=1e6, res_km=RES_B02, rho_min=0.0, rho_max=0.8, n=11, bands=[S2Band.B02])
    da = scene[S2Band.B02]["rho_s"]
    center = da.shape[0] // 2
    assert da.values[center, center] == pytest.approx(0.8, abs=1e-3)


def test_gaussian_boundary_approaches_rho_min():
    """Gaussian generator values at the boundary approach rho_min for small sigma."""
    scene = gaussian_image_dict(sigma=0.001, res_km=RES_B02, rho_min=0.2, rho_max=1.0, n=11, bands=[S2Band.B02])
    da = scene[S2Band.B02]["rho_s"]
    assert da.values[0, 0] == pytest.approx(0.2, abs=1e-3)


# --- disk_image_dict ---


def test_disk_image_dict_shape():
    """Disk generator produces an n×n DataArray per band."""
    scene = disk_image_dict(radius=0.05, res_km=RES_B02, n=11, bands=[S2Band.B02])
    assert scene[S2Band.B02]["rho_s"].shape == (11, 11)


def test_disk_image_dict_values():
    """Disk generator assigns rho_max inside the disk and rho_min outside."""
    scene = disk_image_dict(radius=0.05, res_km=RES_B02, rho_min=0.0, rho_max=1.0, n=11, bands=[S2Band.B02])
    da = scene[S2Band.B02]["rho_s"]
    x = da.coords["x"].values
    y = da.coords["y"].values
    xx, yy = np.meshgrid(x, y)
    inside = (xx**2 + yy**2) <= 0.05**2
    assert np.allclose(da.values[inside], 1.0)
    assert np.allclose(da.values[~inside], 0.0)


# --- random_image_dict ---


def test_random_image_dict_shape():
    """Random generator produces an n×n DataArray per band and variable."""
    scene = random_image_dict(bands=[S2Band.B02], variables=["rho_s", "rho_toa"], res_km=RES_B02, n=16)
    assert scene[S2Band.B02]["rho_s"].shape == (16, 16)
    assert scene[S2Band.B02]["rho_toa"].shape == (16, 16)


def test_random_image_dict_reproducible():
    """Random generator with the same seed produces identical data."""
    s1 = random_image_dict(bands=[S2Band.B02], variables=["rho_s"], res_km=RES_B02, n=16, seed=42)
    s2 = random_image_dict(bands=[S2Band.B02], variables=["rho_s"], res_km=RES_B02, n=16, seed=42)
    np.testing.assert_array_equal(
        s1[S2Band.B02]["rho_s"].values, s2[S2Band.B02]["rho_s"].values
    )


def test_random_image_dict_different_seeds():
    """Random generator with distinct seeds produces different data."""
    s1 = random_image_dict(bands=[S2Band.B02], variables=["rho_s"], res_km=RES_B02, n=16, seed=0)
    s2 = random_image_dict(bands=[S2Band.B02], variables=["rho_s"], res_km=RES_B02, n=16, seed=1)
    assert not np.array_equal(
        s1[S2Band.B02]["rho_s"].values, s2[S2Band.B02]["rho_s"].values
    )
