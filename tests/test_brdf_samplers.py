"""Tests for the non-lambertian ``tdif_up`` and ``sph_alb`` samplers.

None of these drives Smart-G.  What they cover is everything around the
Monte-Carlo call: the relative azimuth the BRDF responds to, the cache
key that has to separate two surfaces, the normalisation that turns a
raw draw into a 5S term, and the pipeline swap.  The physics itself is
checked on a GPU, in ``test_integration.py``.
"""

import numpy as np
import pytest
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import ImageDict, S2Band
from adjeff.modules.samplers import (
    RadiativePipeline,
    SphAlbBrdfSampler,
    SphAlbSampler,
    TdifUpBrdfSampler,
    TdifUpSampler,
)
from adjeff.modules.samplers._ensure import ensure_downward

BAND = S2Band.B02


def _build(cls, **kwargs):
    """Return a BRDF sampler configured the way a user would."""
    defaults = dict(
        atmo_config=AtmoConfig(
            aot=[0.1, 0.4], rh=50.0, h=0.0, href=2.0, species={"sulphate": 1.0}
        ),
        geo_config=GeoConfig(sza=40.0, vza=10.0, saa=30.0, vaa=120.0),
        spectral_config=SpectralConfig.from_bands([BAND]),
        remove_rayleigh=False,
    )
    return cls(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "saa,vaa,expected",
    [(30.0, 120.0, 90.0), (0.0, 0.0, 0.0), (350.0, 10.0, 20.0), (10.0, 350.0, 340.0)],
)
def test_the_relative_azimuth_is_derived_from_the_two_absolute_ones(saa, vaa, expected):
    """A BRDF responds to ``vaa - saa``, wrapped into ``[0, 360)``."""
    sampler = _build(
        TdifUpBrdfSampler, geo_config=GeoConfig(sza=40.0, vza=10.0, saa=saa, vaa=vaa)
    )

    assert sampler._geo_static("raa") == pytest.approx(expected)


def test_the_statics_carry_the_kernel_weights_and_the_relative_azimuth():
    """The physics reads the surface in the terms the surface model uses."""
    statics = _build(TdifUpBrdfSampler, k0=0.4, k1p=0.2, k2p=-0.1)._statics()

    assert statics["k0"] == 0.4
    assert statics["k1p"] == 0.2
    assert statics["k2p"] == -0.1
    assert statics["raa"] == pytest.approx(90.0)
    assert "vaa" not in statics, "the BRDF sweeps a relative azimuth, not an absolute"


def test_both_angles_are_swept_for_tdif_up_and_only_one_for_sph_alb():
    """A BRDF breaks the reciprocity that collapsed ``sza`` and ``vza``.

    ``sph_alb`` stays on one angle: it is a hemispheric integral, so
    there is no viewing direction to carry.
    """
    tdif_up = set(_build(TdifUpBrdfSampler)._contract.inputs)
    sph_alb = set(_build(SphAlbBrdfSampler)._contract.inputs)

    assert {"sza", "vza"} <= tdif_up
    assert "sza" in sph_alb and "vza" not in sph_alb


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weights", [{"k0": 0.5}, {"k1p": 0.3}, {"k2p": 0.3}])
def test_two_surfaces_never_share_a_cache_entry(weights):
    """Each kernel weight must reach the key, or two BRDFs collide.

    They travel as floats rather than as a built Smart-G surface for
    exactly this reason: nothing guarantees such an object hashes the
    same twice.
    """
    reference = _build(TdifUpBrdfSampler)._config_dict()
    other = _build(TdifUpBrdfSampler, **weights)._config_dict()

    assert reference != other


def test_the_dependency_photon_count_defaults_to_the_sampler_s_own():
    """The result is a ratio: precision spent on one side alone is wasted."""
    assert _build(TdifUpBrdfSampler, n_ph=int(1e6)).n_ph_tdif_down == int(1e6)
    assert _build(
        TdifUpBrdfSampler, n_ph=int(1e6), n_ph_tdif_down=int(1e8)
    ).n_ph_tdif_down == int(1e8)


def test_the_reused_variables_are_optional_not_required():
    """A standalone call must be able to compute them itself.

    Declaring them required would forbid it; leaving them out entirely
    would let two different draws share one cache entry.
    """
    sampler = _build(TdifUpBrdfSampler)

    assert sampler.required_vars == []
    assert set(sampler.optional_vars) == {"tdir_down", "tdif_down", "tdir_up"}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _scene_with_transmittances() -> ImageDict:
    """Return a scene carrying the three downward quantities."""
    aot = xr.DataArray([0.1, 0.4], dims=["aot"], coords={"aot": [0.1, 0.4]})
    return ImageDict(
        {
            BAND: xr.Dataset(
                {
                    "tdir_down": aot * 0 + 0.8,
                    "tdif_down": aot * 0 + 0.2,
                    "tdir_up": aot * 0 + 0.7,
                }
            )
        }
    )


@pytest.mark.parametrize(
    "cls,expected",
    [(TdifUpBrdfSampler, 0.5 / 1.0 - 0.7), (SphAlbBrdfSampler, 0.5 / 1.0)],
    ids=["tdif_up", "sph_alb"],
)
def test_the_raw_draw_is_divided_by_the_downward_path(cls, expected, monkeypatch):
    """The normalisation is the one that was living in a caller's script.

    ``tdif_up`` also removes the direct beam, ``sph_alb`` does not.
    """
    scene = _scene_with_transmittances()
    sampler = _build(cls)
    raw = xr.DataArray(
        np.full((2, 1), 0.5),
        dims=["aot", "wl"],
        coords={"aot": [0.1, 0.4], "wl": [BAND.wl_nm]},
    )
    monkeypatch.setattr(type(sampler), "_sweep", lambda self, **kw: raw)

    out = sampler._compute(scene)[BAND][sampler.output_vars[0]]

    np.testing.assert_allclose(out.values.ravel(), [expected, expected])


def test_a_rename_moves_both_what_is_read_and_what_is_written(monkeypatch):
    """The reuse must follow a renamed slot, like everywhere else."""
    scene = _scene_with_transmittances()
    scene[BAND] = scene[BAND].rename({"tdif_down": "tdif_down_ref"})
    sampler = _build(
        TdifUpBrdfSampler,
        rename={"tdif_down": "tdif_down_ref", "tdif_up": "tdif_up_brdf"},
    )
    raw = xr.DataArray(
        np.full((2, 1), 0.5),
        dims=["aot", "wl"],
        coords={"aot": [0.1, 0.4], "wl": [BAND.wl_nm]},
    )
    monkeypatch.setattr(type(sampler), "_sweep", lambda self, **kw: raw)

    out = sampler._compute(scene)[BAND]

    assert "tdif_up_brdf" in out
    assert "tdif_up" not in out


def test_ensure_downward_leaves_a_complete_scene_alone():
    """Recomputing a draw the scene already holds would shift every ratio."""
    scene = _scene_with_transmittances()

    same = ensure_downward(
        scene,
        atmo_config=AtmoConfig(
            aot=[0.1, 0.4], rh=50.0, h=0.0, href=2.0, species={"sulphate": 1.0}
        ),
        geo_config=GeoConfig(sza=40.0, vza=10.0, saa=30.0, vaa=120.0),
        spectral_config=SpectralConfig.from_bands([BAND]),
        remove_rayleigh=False,
        afgl_type="afgl_exp_h8km",
        n_ph_tdif_down=1,
        cache=None,
    )

    assert same is scene


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _pipeline(**kwargs):
    """Return a radiative pipeline over a small sweep."""
    return RadiativePipeline(
        atmo_config=AtmoConfig(
            aot=[0.1, 0.4], rh=50.0, h=0.0, href=2.0, species={"sulphate": 1.0}
        ),
        geo_config=GeoConfig(sza=40.0, vza=10.0, saa=30.0, vaa=120.0),
        spectral_config=SpectralConfig.from_bands([BAND]),
        remove_rayleigh=False,
        **kwargs,
    )


def test_the_lambertian_pipeline_is_unchanged():
    """Passing no surface must keep exactly the six standard samplers."""
    kinds = [type(m) for m in _pipeline()._modules]

    assert SphAlbSampler in kinds and TdifUpSampler in kinds
    assert SphAlbBrdfSampler not in kinds and TdifUpBrdfSampler not in kinds
    assert len(kinds) == 6


def test_a_surface_replaces_the_two_terms_that_see_the_ground():
    """Only ``tdif_up`` and ``sph_alb`` depend on the surface model.

    The two are replaced rather than added: they write the same slots,
    and they are the most expensive samplers of the six.
    """
    kinds = [type(m) for m in _pipeline(rtls=(1.0, 0.3, 0.1))._modules]

    assert SphAlbBrdfSampler in kinds and TdifUpBrdfSampler in kinds
    assert SphAlbSampler not in kinds and TdifUpSampler not in kinds
    assert len(kinds) == 6


def test_the_brdf_variants_come_after_what_they_read():
    """They divide by the downward terms, so those must exist first."""
    modules = _pipeline(rtls=(1.0, 0.3, 0.1))._modules
    produced: set[str] = set()

    for module in modules:
        assert set(module.optional_vars) <= produced | {
            v for v in module.optional_vars if v not in produced
        }
        if isinstance(module, (SphAlbBrdfSampler, TdifUpBrdfSampler)):
            assert set(module.optional_vars) <= produced, (
                f"{type(module).__name__} reads {module.optional_vars} "
                f"but only {sorted(produced)} exist by then"
            )
        produced.update(module.output_vars)


def test_the_pipeline_forwards_the_kernel_weights():
    """A surface given once must reach both samplers."""
    modules = _pipeline(rtls=(0.6, 0.3, 0.1))._modules
    brdf = [m for m in modules if isinstance(m, (SphAlbBrdfSampler, TdifUpBrdfSampler))]

    assert len(brdf) == 2
    for module in brdf:
        assert (module.k0, module.k1p, module.k2p) == (0.6, 0.3, 0.1)
