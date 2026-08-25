"""Tests for SceneModule base-class behaviour using TestModule as a fixture."""

import numpy as np
import pytest
from _test_module import TestModule

from adjeff.core import ImageDict, S2Band, random_image_dict
from adjeff.exceptions import MissingVariableError
from adjeff.utils import CacheStore


@pytest.fixture
def scene():
    """Return a small single-band scene with rho_s."""
    return random_image_dict(
        bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=0
    )


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
    scene_a = random_image_dict(
        bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=0
    )
    scene_b = random_image_dict(
        bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=1
    )
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
    scene = random_image_dict(
        bands=[S2Band.B02], variables=["rho_toa"], res_km=0.01, n=8, seed=0
    )
    with pytest.raises(MissingVariableError):
        TestModule()(scene)


# --- Pipeline ---


def test_pipeline_wrong_dependency_raises():
    """Pipeline rejects a dependency declared but not yet produced.

    The variable is a pipeline output, so the mistake is one of order
    rather than of declaration.
    """
    from _test_module import TestModule as TM

    from adjeff.exceptions import ConfigurationError
    from adjeff.modules import Pipeline

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
    from _test_module import TestModule as TM

    from adjeff.exceptions import ConfigurationError

    class Hidden(TM):
        def __init__(self, shift, cache=None):
            self._shift = shift  # wrong: not visible to _config_dict()
            super().__init__(cache=cache)

    with pytest.raises(ConfigurationError, match="shift"):
        Hidden(shift=0.1)._config_dict()


def test_sweep_params_reach_the_cache_key():
    """batch_size and dedup must change the cache key.

    Neither can change a value, but both are __init__ parameters, and the
    guard that keeps `res` and the old `deduplicate_dims` in the key is
    what would catch either being stored under a private name.
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
    deduped = TdirDownSampler(**common, dedup=True)._cache_key(scene)
    grouped = TdirDownSampler(**common, batch_size=8)._cache_key(scene)

    assert len({plain, deduped, grouped}) == 3


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


def test_truncated_cache_entry_reads_as_a_miss(tmp_path):
    """A cache entry short of an output var must not count as a hit.

    Returning the variables that happen to be present would hand back a
    scene silently missing an output, reported as ``cached=True``.
    """
    import shutil

    import xarray as xr
    from _test_module import TestModule as TM

    class TwoOut(TM):
        output_vars = ["rho_toa", "rho_unif"]

        def _compute(self, scene):  # type: ignore[override]
            for band in scene.bands:
                ds = scene[band]
                ds["rho_toa"] = ds["rho_s"] + 0.05
                ds["rho_unif"] = ds["rho_s"] + 0.10
            return scene

    cache = CacheStore(tmp_path)
    scene = random_image_dict(
        bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=0
    )
    module = TwoOut(cache=cache)
    module(scene)

    # Drop one output from the stored entry, as an interrupted write or a
    # changed output_vars would.
    path = tmp_path / module._cache_key(scene) / f"{S2Band.B02}.zarr"
    stored = xr.open_zarr(path).load().drop_vars("rho_unif")
    shutil.rmtree(path)
    stored.to_zarr(path, mode="w")

    assert (
        cache.load_vars(module._cache_key(scene), [S2Band.B02], TwoOut.output_vars)
        is None
    )
    assert "rho_unif" in TwoOut(cache=cache)(scene)[S2Band.B02]


# --- Pipeline streaming ---


@pytest.fixture
def streamed_scene():
    """Return a scene with two swept dims and a species attribute."""
    import xarray as xr

    da = xr.DataArray(
        np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4).astype(float),
        dims=["aot", "rh", "y", "x"],
        coords={"aot": [0.1, 0.2], "rh": [40.0, 50.0, 60.0]},
    )
    return ImageDict(
        {
            S2Band.B02: xr.Dataset(
                {"rho_s": da}, attrs={"adjeff:species": {"sulphate": 1.0}}
            )
        }
    )


class _Doubler(TestModule):
    """Write ``out = 2 * rho_s``, so streaming must not change the result."""

    output_vars = ["out"]

    def _compute(self, scene):  # type: ignore[override]
        for band in scene.bands:
            scene[band]["out"] = scene[band]["rho_s"] * 2
        return scene


@pytest.mark.parametrize(
    "stream_dims",
    [{"aot": 1}, {"rh": 2}, {"aot": 1, "rh": 1}, {"aot": 1, "rh": 2}],
)
def test_streaming_matches_the_full_run(streamed_scene, stream_dims):
    """Streaming over one or several dims must reproduce the full run.

    Folding the Cartesian product of chunks along a single dimension
    would stack n0 * n1 pieces on one axis instead of rebuilding the grid.
    """
    from adjeff.modules import Pipeline

    reference = Pipeline([_Doubler()])(streamed_scene)[S2Band.B02]["out"]
    streamed = Pipeline([_Doubler()], stream_dims=stream_dims)(streamed_scene)[
        S2Band.B02
    ]["out"]

    assert streamed.transpose(*reference.dims).shape == reference.shape
    np.testing.assert_allclose(
        streamed.transpose(*reference.dims).values, reference.values
    )


def test_streaming_preserves_dataset_attrs(streamed_scene):
    """Streaming must carry the band attrs through.

    They hold the aerosol species written by load_scene(); losing them
    makes load_config() fall back to sulphate without a word.
    """
    from adjeff.modules import Pipeline

    out = Pipeline([_Doubler()], stream_dims={"aot": 1})(streamed_scene)
    assert out[S2Band.B02].attrs["adjeff:species"] == {"sulphate": 1.0}


def test_cache_ignores_the_encoding_of_a_reloaded_array(tmp_path, scene):
    """An array read back from zarr must be storable again.

    `forward` swaps its outputs for lazy zarr-backed views, which carry
    the file's own `encoding["chunks"]`.  `to_zarr` honours that encoding
    over the chunking the cache asks for and refuses the write when the
    two disagree, which is what happens as soon as a swept dimension is
    longer than one.
    """
    import xarray as xr

    cache = CacheStore(tmp_path)
    module = TestModule(cache=cache)
    result = module(scene)

    # Stand in for a reload: an encoding that contradicts the cache's
    # own one-per-combo chunking along a swept dim.
    stacked = xr.concat([result[S2Band.B02]["rho_toa"]] * 3, dim="aot").chunk(
        {"aot": 1}
    )
    stacked.encoding["chunks"] = (3, *stacked.shape[1:])
    result[S2Band.B02]["rho_toa"] = stacked

    cache.save_vars("some-key", result, ["rho_toa"])

    back = cache.load_vars("some-key", [S2Band.B02], ["rho_toa"])
    assert back is not None
    assert back[S2Band.B02]["rho_toa"].sizes["aot"] == 3


def test_optional_vars_enter_the_key_only_when_present(tmp_path, scene):
    """An optional input must key the entry it contributed to.

    A module that computes a variable when the scene lacks it, and reuses
    the scene's own when it has one, produces two different outputs from
    the same declared inputs.  Leaving that variable out of the key lets
    the two share an entry.
    """
    import xarray as xr

    class OptionalModule(TestModule):
        optional_vars = ["rho_atm"]

    cache = CacheStore(tmp_path)
    module = OptionalModule(cache=cache)
    plain = module._cache_key(scene)

    with_var = scene.shallow_copy()
    with_var[S2Band.B02]["rho_atm"] = xr.DataArray(0.06)
    keyed = module._cache_key(with_var)

    other = scene.shallow_copy()
    other[S2Band.B02]["rho_atm"] = xr.DataArray(0.07)

    assert plain != keyed, "an optional input present must change the key"
    assert keyed != module._cache_key(other), "two values, two keys"
    assert plain == module._cache_key(scene.shallow_copy())
