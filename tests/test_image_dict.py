"""Tests for ImageDict, the per-band container."""

import numpy as np
import pytest
import xarray as xr

from adjeff.core import ImageDict, S2Band, random_image_dict
from adjeff.exceptions import MissingVariableError


def _make_scene(
    bands=(S2Band.B02, S2Band.B03), n=16, variables=("rho_s",)
) -> ImageDict:
    return random_image_dict(list(bands), list(variables), res_km=0.01, n=n)


class TestMapping:
    """An ImageDict as a mapping of band to dataset."""

    def test_bands_sorted(self):
        scene = _make_scene([S2Band.B04, S2Band.B02, S2Band.B03])
        assert scene.bands == [S2Band.B02, S2Band.B03, S2Band.B04]

    def test_getitem_returns_dataset(self):
        scene = _make_scene()
        ds = scene[S2Band.B02]
        assert isinstance(ds, xr.Dataset)

    def test_setitem(self):
        scene = _make_scene()
        new_ds = xr.Dataset(
            {"rho_s": xr.DataArray(np.zeros((16, 16)), dims=["y", "x"])}
        )
        scene[S2Band.B11] = new_ds
        assert S2Band.B11 in scene

    def test_contains(self):
        scene = _make_scene()
        assert S2Band.B02 in scene
        assert "B99" not in scene

    def test_variables(self):
        scene = _make_scene(variables=["rho_s", "rho_toa"])
        assert set(scene.variables(S2Band.B02)) == {"rho_s", "rho_toa"}

    def test_require_vars_pass(self):
        """require_vars does not raise when all required variables are present."""
        scene = _make_scene(variables=["rho_s"])
        scene.require_vars(["rho_s"])

    def test_require_vars_raises(self):
        scene = _make_scene(variables=["rho_s"])
        with pytest.raises(MissingVariableError):
            scene.require_vars(["rho_toa"])
