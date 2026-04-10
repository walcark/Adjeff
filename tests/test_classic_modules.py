"""Tests for the 5S forward and inverse modules: Unif2Toa and Toa2Unif."""

import numpy as np
import xarray as xr

from adjeff.core import ImageDict, S2Band
from adjeff.modules.classic.toa_to_unif import Toa2Unif
from adjeff.modules.classic.unif_to_toa import Unif2Toa

_SHAPE = (4, 4)
_COORDS = {"y": np.arange(_SHAPE[0], dtype=np.float32), "x": np.arange(_SHAPE[1], dtype=np.float32)}

# 5S model parameter constants used across tests.
_T_DIR_UP = 0.6
_T_DIF_UP = 0.1
_T_DIR_DOWN = 0.6
_T_DIF_DOWN = 0.1
_RHO_ATM = 0.05
_SPH_ALB = 0.1


def _const(val: float) -> xr.DataArray:
    """Return a constant DataArray over the test spatial grid."""
    return xr.DataArray(
        np.full(_SHAPE, val, dtype=np.float32), dims=["y", "x"], coords=_COORDS
    )


def _make_unif2toa_scene(rho_unif: float = 0.1) -> ImageDict:
    """Build a scene with the variables required by Unif2Toa."""
    ds = xr.Dataset({
        "rho_unif": _const(rho_unif),
        "tdir_up": _const(_T_DIR_UP),
        "tdif_up": _const(_T_DIF_UP),
        "tdir_down": _const(_T_DIR_DOWN),
        "tdif_down": _const(_T_DIF_DOWN),
        "rho_atm": _const(_RHO_ATM),
        "sph_alb": _const(_SPH_ALB),
    })
    return ImageDict({S2Band.B02: ds})


def _make_toa2unif_scene(rho_toa: float) -> ImageDict:
    """Build a scene with the variables required by Toa2Unif."""
    ds = xr.Dataset({
        "rho_toa": _const(rho_toa),
        "tdir_up": _const(_T_DIR_UP),
        "tdif_up": _const(_T_DIF_UP),
        "tdir_down": _const(_T_DIR_DOWN),
        "tdif_down": _const(_T_DIF_DOWN),
        "rho_atm": _const(_RHO_ATM),
        "sph_alb": _const(_SPH_ALB),
    })
    return ImageDict({S2Band.B02: ds})


# --- Unif2Toa ---


def test_unif2toa_produces_rho_toa():
    """Unif2Toa writes rho_toa into the output scene."""
    result = Unif2Toa()(_make_unif2toa_scene())
    assert "rho_toa" in result[S2Band.B02]


def test_unif2toa_formula():
    """Unif2Toa applies the 5S forward model: rho_toa = rho_atm + t_up * t_down * rho_unif / (1 - S * rho_unif)."""
    rho_unif = 0.1
    result = Unif2Toa()(_make_unif2toa_scene(rho_unif=rho_unif))
    t_up = _T_DIR_UP + _T_DIF_UP
    t_down = _T_DIR_DOWN + _T_DIF_DOWN
    expected = _RHO_ATM + t_up * t_down * rho_unif / (1 - _SPH_ALB * rho_unif)
    np.testing.assert_allclose(
        result[S2Band.B02]["rho_toa"].values, expected, rtol=1e-5
    )


# --- Toa2Unif ---


def test_toa2unif_produces_rho_unif():
    """Toa2Unif writes rho_unif into the output scene."""
    t_up = _T_DIR_UP + _T_DIF_UP
    t_down = _T_DIR_DOWN + _T_DIF_DOWN
    rho_unif = 0.1
    rho_toa = _RHO_ATM + t_up * t_down * rho_unif / (1 - _SPH_ALB * rho_unif)
    result = Toa2Unif()(_make_toa2unif_scene(rho_toa=rho_toa))
    assert "rho_unif" in result[S2Band.B02]


def test_toa2unif_formula():
    """Toa2Unif inverts the 5S model: rho_unif = (rho_toa - rho_atm) / (S * (rho_toa - rho_atm) + t_up * t_down)."""
    t_up = _T_DIR_UP + _T_DIF_UP
    t_down = _T_DIR_DOWN + _T_DIF_DOWN
    rho_unif_orig = 0.15
    rho_toa = _RHO_ATM + t_up * t_down * rho_unif_orig / (1 - _SPH_ALB * rho_unif_orig)
    result = Toa2Unif()(_make_toa2unif_scene(rho_toa=rho_toa))
    np.testing.assert_allclose(
        result[S2Band.B02]["rho_unif"].values, rho_unif_orig, rtol=1e-4
    )


# --- Roundtrip ---


def test_unif2toa_toa2unif_roundtrip():
    """Composing Unif2Toa then Toa2Unif recovers the original rho_unif."""
    rho_unif_orig = 0.2
    scene = _make_unif2toa_scene(rho_unif=rho_unif_orig)
    scene = Unif2Toa()(scene)
    # Toa2Unif requires rho_toa (now present) and the 5S params (already there).
    scene = Toa2Unif()(scene)
    np.testing.assert_allclose(
        scene[S2Band.B02]["rho_unif"].values, rho_unif_orig, rtol=1e-4
    )
