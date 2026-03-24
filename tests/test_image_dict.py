import numpy as np
import pytest
import xarray as xr

from adjeff.core import ImageDict, S2Band, random_image_dict
from adjeff.exceptions import MissingVariableError


def _make_scene(
    band_ids=(S2Band.B02, S2Band.B03),
    n=16,
    variables=("rho_s",)
) -> ImageDict:
    return random_image_dict(list(band_ids), list(variables), n=n)


def test_band_ids_sorted():
    """Band IDs are returned sorted by wavelength."""
    scene = _make_scene([S2Band.B04, S2Band.B02, S2Band.B03])
    assert scene.band_ids == [S2Band.B02, S2Band.B03, S2Band.B04]

def test_getitem_returns_dataset():
    """__getitem__ returns the xr.Dataset for a given band."""
    scene = _make_scene()
    ds = scene[S2Band.B02]
    assert isinstance(ds, xr.Dataset)

def test_setitem():
    """__setitem__ inserts a new band Dataset into the ImageDict."""
    scene = _make_scene()
    new_ds = xr.Dataset({"rho_s": xr.DataArray(np.zeros((16, 16)), dims=["y", "x"])})
    scene[S2Band.B11] = new_ds
    assert S2Band.B11 in scene

def test_contains():
    """__contains__ returns True for a present band and False for an absent one."""
    scene = _make_scene()
    assert S2Band.B02 in scene
    assert "B99" not in scene

def test_variables():
    """variables() returns all DataArray names for a given band."""
    scene = _make_scene(variables=["rho_s", "rho_toa"])
    assert set(scene.variables(S2Band.B02)) == {"rho_s", "rho_toa"}

def test_has_var_true():
    """has_var returns True when all bands contain the variable."""
    scene = _make_scene(variables=["rho_s"])
    assert scene.has_var("rho_s") is True

def test_has_var_false():
    """has_var returns False when the variable is absent from at least one band."""
    scene = _make_scene(variables=["rho_s"])
    assert scene.has_var("rho_toa") is False

def test_require_vars_pass():
    """require_vars does not raise when all required variables are present."""
    scene = _make_scene(variables=["rho_s"])
    scene.require_vars(["rho_s"])

def test_require_vars_raises():
    """require_vars raises MissingVariableError when a variable is absent."""
    scene = _make_scene(variables=["rho_s"])
    with pytest.raises(MissingVariableError):
        scene.require_vars(["rho_toa"])

def test_write_to_directory_no_extra_dims(tmp_path):
    """write_to_directory writes a single .npy file named {var}__{band}.npy."""
    scene = _make_scene([S2Band.B02], n=8, variables=["rho_s"])
    written = scene.write_to_directory(tmp_path / "out", "rho_s")
    assert len(written) == 1
    assert written[0].name == "rho_s__S2Band.B02.npy"
    arr = np.load(written[0])
    assert arr.shape == (8, 8)

def test_write_to_directory_extra_dim(tmp_path):
    """write_to_directory writes one file per extra-dimension slice."""
    data = np.random.rand(2, 8, 8).astype(np.float32)
    da = xr.DataArray(data, dims=["aot", "y", "x"], coords={"aot": [0.1, 0.2]})
    ds = xr.Dataset({"rho_corrected": da})
    scene = ImageDict({S2Band.B02: ds})
    written = scene.write_to_directory(tmp_path / "out", "rho_corrected")
    assert len(written) == 2
    names = {p.name for p in written}
    assert "rho_corrected__aot=0.1__S2Band.B02.npy" in names
    assert "rho_corrected__aot=0.2__S2Band.B02.npy" in names

def test_write_missing_var_raises(tmp_path):
    """write_to_directory raises MissingVariableError for an absent variable."""
    scene = _make_scene(variables=["rho_s"])
    with pytest.raises(MissingVariableError):
        scene.write_to_directory(tmp_path, "rho_toa")

def test_progressive_enrichment():
    """Datasets can be enriched in-place by adding new variables."""
    scene = _make_scene(variables=["rho_s"])
    for band_id in scene.band_ids:
        ds = scene[band_id]
        ds["rho_toa"] = xr.DataArray(
            np.random.rand(16, 16).astype(np.float32), dims=["y", "x"]
        )
    assert scene.has_var("rho_toa")

