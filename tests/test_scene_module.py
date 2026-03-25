"""Tests for SceneModule base-class behaviour using TestModule as a fixture."""

import numpy as np
import pytest

from adjeff.core import S2Band, random_image_dict
from adjeff.exceptions import MissingVariableError
from adjeff.modules import TestModule
from adjeff.utils import CacheStore


@pytest.fixture
def scene():
    """Return a small single-band scene with rho_s."""
    return random_image_dict(bands=[S2Band.B02], variables=["rho_s"], n=8, seed=0)


# --- TestModule compute ---


def test_testmodule_produces_rho_toa(scene):
    """TestModule writes rho_toa into the output scene."""
    result = TestModule()(scene)
    assert "rho_toa" in result[S2Band.B02]


def test_testmodule_shift_value(scene):
    """TestModule computes rho_toa as rho_s + 0.05."""
    rho_s = scene[S2Band.B02]["rho_s"].values.copy()
    result = TestModule()(scene)
    np.testing.assert_allclose(
        result[S2Band.B02]["rho_toa"].values, rho_s + 0.05, rtol=1e-5
    )


# --- Provenance ---


def test_provenance_is_stamped(scene):
    """Output DataArrays carry _adjeff_provenance after compute."""
    result = TestModule()(scene)
    prov = result[S2Band.B02]["rho_toa"].attrs.get("_adjeff_provenance")
    assert prov is not None
    assert prov["module"] == "TestModule"
    assert "key" in prov


# --- Cache ---


def test_cache_hit_returns_same_values(tmp_path, scene):
    """Second call with identical input hits the cache and returns the same values."""
    cache = CacheStore(tmp_path)
    module = TestModule(cache=cache)
    result1 = module(scene)
    result2 = module(scene)
    np.testing.assert_array_equal(
        result1[S2Band.B02]["rho_toa"].values,
        result2[S2Band.B02]["rho_toa"].values,
    )


def test_cache_different_inputs_differ(tmp_path):
    """Two scenes with different content produce different outputs."""
    cache = CacheStore(tmp_path)
    scene_a = random_image_dict(bands=[S2Band.B02], variables=["rho_s"], n=8, seed=0)
    scene_b = random_image_dict(bands=[S2Band.B02], variables=["rho_s"], n=8, seed=1)
    module = TestModule(cache=cache)
    r_a = module(scene_a)
    r_b = module(scene_b)
    assert not np.array_equal(
        r_a[S2Band.B02]["rho_toa"].values,
        r_b[S2Band.B02]["rho_toa"].values,
    )


# --- Validation ---


def test_missing_required_var_raises():
    """TestModule raises MissingVariableError when rho_s is absent."""
    scene = random_image_dict(bands=[S2Band.B02], variables=["rho_toa"], n=8, seed=0)
    with pytest.raises(MissingVariableError):
        TestModule()(scene)
