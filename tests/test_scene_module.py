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
    return random_image_dict(bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=0)


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
    scene_a = random_image_dict(bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=0)
    scene_b = random_image_dict(bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=1)
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
    scene = random_image_dict(bands=[S2Band.B02], variables=["rho_toa"], res_km=0.01, n=8, seed=0)
    with pytest.raises(MissingVariableError):
        TestModule()(scene)


# --- Pipeline ---


def test_pipeline_wrong_dependency_raises():
    """Pipeline raises ConfigurationError when a dependency is declared
    as a pipeline output but not produced before it is needed."""
    from adjeff.exceptions import ConfigurationError
    from adjeff.modules import Pipeline
    from adjeff.modules.test_module import TestModule as TM

    # Module A requires rho_s and produces rho_toa.
    # Module B requires rho_toa and produces rho_unif.
    # Declaring them in reverse order (B then A) is a configuration error.
    class ModuleB(TM):
        required_vars = ["rho_toa"]
        output_vars = ["rho_unif"]

        def _compute(self, scene):  # type: ignore[override]
            return scene

    with pytest.raises(ConfigurationError):
        Pipeline([ModuleB(), TM()])


# --- Cache key completeness ---


def test_config_dict_raises_on_privately_stored_param():
    """A param stored under a private name must not be silently dropped.

    Auto-detection reads __init__ params off same-named attributes, so a
    private name would leave the param out of the cache key and let two
    different configurations collide on one entry.
    """
    from adjeff.exceptions import ConfigurationError
    from adjeff.modules.test_module import TestModule as TM

    class Hidden(TM):
        def __init__(self, shift, cache=None):
            self._shift = shift  # wrong: not visible to _config_dict()
            super().__init__(cache=cache)

    with pytest.raises(ConfigurationError, match="shift"):
        Hidden(shift=0.1)._config_dict()


def test_sweep_params_reach_the_cache_key():
    """deduplicate_dims and sweep_chunks must change the cache key.

    Both alter the shape of a sampler's output, so two runs that differ
    only by one of them must not share a cache entry.
    """
    import xarray as xr

    from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
    from adjeff.core import ImageDict
    from adjeff.modules.samplers import TdirDownSampler

    common = dict(
        atmo_config=AtmoConfig(
            aot=0.1, rh=50.0, h=0.0, href=2.0, species={"sulphate": 1.0}
        ),
        geo_config=GeoConfig(sza=30.0, vza=0.0, saa=120.0, vaa=120.0),
        spectral_config=SpectralConfig.from_bands([S2Band.B02]),
        remove_rayleigh=False,
    )
    scene = ImageDict({S2Band.B02: xr.Dataset()})

    plain = TdirDownSampler(**common)._cache_key(scene)
    dedup = TdirDownSampler(
        **common, deduplicate_dims=["x", "y"]
    )._cache_key(scene)
    chunked = TdirDownSampler(**common, sweep_chunks={"wl": 2})._cache_key(
        scene
    )

    assert len({plain, dedup, chunked}) == 3


def test_loader_resolution_reaches_the_cache_key(tmp_path):
    """ProductLoader.res must change the cache key.

    The same product loaded at two target resolutions holds different
    pixels, so the entries must not collide.
    """
    from adjeff.modules.loaders.product_loader import ProductLoader

    class Loader(ProductLoader):
        def ensure_correct_folder(self, path):
            return None

        def extract_metadata(self):
            return None

        def reflectance(self, band, btype="SRE"):
            raise NotImplementedError

        def _compute(self, scene):
            return scene

    def key(res):
        return Loader(
            product_path=tmp_path, bands=[S2Band.B02], res=res
        )._config_dict()["res"]

    assert key(0.12) != key(0.06)
