"""End-to-end tests driving Smart-G on a real GPU.

The unit suite never reaches the physics: it exercises configuration,
caching and array plumbing, leaving ``_smartg.py`` at 16% coverage and
``api.py`` at 29%.  These tests close that gap by running the real chain
on a grid small enough to finish in seconds.

They check shapes, finiteness and physical bounds, never exact values:
Smart-G is a Monte-Carlo engine and the photon counts here are tiny on
purpose.  What is being verified is that the chain still runs and still
returns something of the right shape, which is what breaks when the API
moves.

Excluded from the default run by ``addopts`` in ``pyproject.toml`` and
skipped without CUDA, so a CI runner never attempts them.  Run with::

    pixi run -e dev-gpu test-integration
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from adjeff.api import (
    make_full_config,
    run_forward_pipeline,
    sample_psf_atm,
)
from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import ImageDict, S2Band, disk_image_dict
from adjeff.modules.samplers import (
    PsfAtmSampler,
    RadiativePipeline,
    TdirDownSampler,
)
from adjeff.utils import CacheStore
from conftest import requires_cuda

pytestmark = [pytest.mark.integration, requires_cuda]

BAND = S2Band.B03

# Coarse pixels over a wide field rather than a small field: the article
# works at 50 m over 3999 pixels, and shrinking the extent instead of the
# resolution would make the adjacency contribution leave the grid.
RES_KM = 1.0
N = 65
N_PH = int(1e3)

# The six quantities the 5S formula needs.
RADIATIVE_VARS = (
    "tdir_down",
    "tdir_up",
    "tdif_down",
    "tdif_up",
    "rho_atm",
    "sph_alb",
)


@pytest.fixture(scope="module")
def config():
    """Return a single-band, single-state configuration."""
    return make_full_config(
        bands=[BAND],
        aot=0.4,
        rh=50.0,
        h=0.0,
        href=2.0,
        sza=40.0,
        vza=8.0,
        saa=0.0,
        vaa=0.0,
        species={"sulphate": 1.0},
    )


@pytest.fixture
def surface():
    """Return a scene holding one uniform disk."""
    return disk_image_dict(
        radius=5.0, res_km=RES_KM, rho_min=0.0, rho_max=0.5, bands=[BAND], n=N
    )


def _finite_in(da: xr.DataArray, low: float, high: float) -> None:
    """Assert *da* is finite and within ``[low, high]``."""
    values = np.asarray(da.values, dtype=float)
    assert np.isfinite(values).all(), "Smart-G returned NaN or inf"
    assert values.min() >= low, f"{float(values.min())} below {low}"
    assert values.max() <= high, f"{float(values.max())} above {high}"


# --- Radiative chain ---


def test_radiative_pipeline_produces_all_six_quantities(config):
    """RadiativePipeline writes the six 5S quantities, all in [0, 1]."""
    scene = RadiativePipeline(**config, remove_rayleigh=False, n_ph_sph_alb=N_PH,
                              n_ph_rho_atm=N_PH, n_ph_tdif_up=N_PH,
                              n_ph_tdif_down=N_PH)(
        ImageDict({BAND: xr.Dataset()})
    )

    for var in RADIATIVE_VARS:
        assert var in scene[BAND], f"{var} missing from the output"
        _finite_in(scene[BAND][var], 0.0, 1.0)


def test_transmittance_decreases_with_aerosol_load(config):
    """Direct downward transmittance must drop as AOT grows.

    A monotonicity check rather than a value check: it holds for any
    photon count, and it is the cheapest way to tell a working chain from
    one that returns a constant.
    """
    atmo = AtmoConfig(
        aot=xr.DataArray([0.05, 0.8], dims=["aot"]),
        rh=50.0,
        h=0.0,
        href=2.0,
        species={"sulphate": 1.0},
    )
    sampler = TdirDownSampler(
        atmo_config=atmo,
        geo_config=config["geo_config"],
        spectral_config=config["spectral_config"],
        remove_rayleigh=False,
    )
    tdir = sampler(ImageDict({BAND: xr.Dataset()}))[BAND]["tdir_down"]

    # Reduce over whatever else the sampler kept (wl, rh, ...): only the
    # AOT ordering is under test.
    low = float(tdir.isel(aot=0).mean())
    high = float(tdir.isel(aot=1).mean())
    assert low > high, f"tdir_down did not decrease with AOT: {low} -> {high}"


# --- Forward pipeline ---


def test_forward_pipeline_reaches_rho_unif(config, surface):
    """run_forward_pipeline chains radiatives, rho_toa and Toa2Unif."""
    scene = run_forward_pipeline(surface, **config, n_ph=N_PH, nr=32)

    for var in ("rho_toa", "rho_unif", *RADIATIVE_VARS):
        assert var in scene[BAND], f"{var} missing from the output"

    rho_s = scene[BAND]["rho_s"]
    rho_toa = scene[BAND]["rho_toa"].squeeze(drop=True)
    assert rho_toa.shape == rho_s.shape
    _finite_in(rho_toa, 0.0, 1.0)


def test_adjacency_blurs_the_disk_edge(config, surface):
    """rho_unif must be smoother than rho_s at the disk edge.

    The whole point of the adjacency effect is that a sharp edge is not
    sharp any more at the top of the atmosphere.  Comparing the largest
    single-pixel jump of the two fields is a physical check that survives
    any photon count.
    """
    scene = run_forward_pipeline(surface, **config, n_ph=N_PH, nr=32)

    def max_step(da: xr.DataArray) -> float:
        values = np.asarray(da.squeeze(drop=True).values, dtype=float)
        return float(np.abs(np.diff(values, axis=-1)).max())

    assert max_step(scene[BAND]["rho_unif"]) < max_step(scene[BAND]["rho_s"])


# --- Atmospheric PSF ---


def test_sample_psf_atm_returns_a_normalised_kernel(config):
    """sample_psf_atm returns a finite, positive, normalised kernel."""
    psf_dict = sample_psf_atm(
        bands=[BAND],
        res_km=RES_KM,
        n=N,
        atmo_config=config["atmo_config"],
        geo_config=config["geo_config"],
        n_ph=N_PH,
    )
    kernel = psf_dict.kernel(BAND).squeeze(drop=True)

    assert kernel.shape == (N, N)
    _finite_in(kernel, 0.0, 1.0)
    assert float(kernel.sum()) == pytest.approx(1.0, rel=1e-3)


# --- Cache ---


def test_sampler_round_trips_through_the_cache(config, tmp_path):
    """A cached sampler must return the same values without recomputing.

    Smart-G is stochastic, so two computed runs differ.  Equality here is
    therefore proof that the second run read the store rather than
    running the engine again.
    """
    cache = CacheStore(tmp_path)
    common = dict(
        atmo_config=config["atmo_config"],
        geo_config=config["geo_config"],
        spectral_config=config["spectral_config"],
        remove_rayleigh=False,
        n_ph=int(2e6),
        cache=cache,
    )
    from adjeff.modules.samplers import RhoAtmSampler

    first = RhoAtmSampler(**common)(ImageDict({BAND: xr.Dataset()}))
    second = RhoAtmSampler(**common)(ImageDict({BAND: xr.Dataset()}))

    xr.testing.assert_identical(
        first[BAND]["rho_atm"].compute(), second[BAND]["rho_atm"].compute()
    )


def test_deduplication_matches_the_plain_sweep(config):
    """A deduplicated spatial sweep must equal the broadcast one.

    Two pixels sharing an atmospheric state are computed once and the
    result expanded back.  This is the path a wrong cache key silently
    corrupted before 0.7.0, and nothing else exercises it.
    """
    aot_map = xr.DataArray(
        np.array([[0.1, 0.4], [0.4, 0.1]]), dims=["y", "x"]
    )
    atmo = AtmoConfig(
        aot=aot_map, rh=50.0, h=0.0, href=2.0, species={"sulphate": 1.0}
    )
    sampler = TdirDownSampler(
        atmo_config=atmo,
        geo_config=config["geo_config"],
        spectral_config=config["spectral_config"],
        remove_rayleigh=False,
        deduplicate_dims=["x", "y"],
    )
    tdir = sampler(ImageDict({BAND: xr.Dataset()}))[BAND]["tdir_down"]

    assert {"y", "x"} <= set(tdir.dims)
    values = np.asarray(tdir.squeeze(drop=True).values, dtype=float)
    # The two AOT=0.1 corners must hold the same value, and so must the
    # two AOT=0.4 ones: that is what the expansion has to restore.
    assert values[0, 0] == pytest.approx(values[1, 1])
    assert values[0, 1] == pytest.approx(values[1, 0])
    assert values[0, 0] > values[0, 1]


# --- PSF sampler on a scene ---


def test_psf_atm_sampler_writes_into_the_scene(config, surface):
    """PsfAtmSampler enriches the scene rather than replacing it."""
    scene = PsfAtmSampler(
        atmo_config=config["atmo_config"],
        geo_config=config["geo_config"],
        remove_rayleigh=False,
        n_ph=N_PH,
    )(surface)

    assert "rho_s" in scene[BAND], "the input variable was dropped"
    assert "psf_atm" in scene[BAND]
    _finite_in(scene[BAND]["psf_atm"], 0.0, 1.0)
