"""Tests for DimSweeper and SceneModuleSweep.

Run with verbose structlog output:
    pixi run --environment dev-gpu pytest tests/test_scene_module_sweep.py -v -s
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import structlog
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig
from adjeff.core import S2Band, random_image_dict
from adjeff.exceptions import MissingVariableError
from adjeff.modules import SceneModuleSweep
from adjeff.utils import CacheStore
from adjeff.utils.sweep import DimSweeper

# ---------------------------------------------------------------------------
# structlog configuration — shows DEBUG logs with context during -s runs
# ---------------------------------------------------------------------------


def pytest_configure(config):  # noqa: ARG001
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        logger_factory=structlog.PrintLoggerFactory(),
    )


# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------


@pytest.fixture
def scalar_atmo():
    """AtmoConfig with all scalar parameters."""
    return AtmoConfig(
        aot=0.2, h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0}
    )


@pytest.fixture
def scalar_geo():
    """GeoConfig with all scalar parameters."""
    return GeoConfig(sza=30.0, vza=10.0, saa=120.0, vaa=0.0)


@pytest.fixture
def sweeper():
    """DimSweeper with typical Smart-G dim split."""
    return DimSweeper(
        scalar_dims=["aot", "sza", "vza", "saa", "vaa"],
        vector_dims=["h", "rh"],
    )


# ---------------------------------------------------------------------------
# DimSweeper — collect
# ---------------------------------------------------------------------------


class TestDimSweeperCollect:
    def test_scalar_config_promotes_to_size1(self, sweeper, scalar_atmo, scalar_geo):
        """Scalar (0-dim) parameters are promoted to size-1 DataArrays."""
        dim_arrays, inverse_map = sweeper.collect(scalar_atmo, scalar_geo)

        assert inverse_map is None
        for name, da in dim_arrays.items():
            assert da.ndim == 1, f"{name} should be 1-D, got {da.ndim}"
            assert da.dims[0] == name, f"{name} dim name mismatch"

    def test_1d_config_kept_as_is(self, sweeper, scalar_geo):
        """1-D DataArray parameters are kept as-is."""
        atmo = AtmoConfig(
            aot=[0.1, 0.2, 0.3], h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0}
        )
        dim_arrays, _ = sweeper.collect(atmo, scalar_geo)
        assert dim_arrays["aot"].shape == (3,)

    def test_2d_param_without_dedup_raises(self, sweeper, scalar_geo):
        """2-D config parameter without deduplicate_dims raises ValueError."""
        aot_2d = xr.DataArray(
            np.array([[0.1, 0.2], [0.3, 0.4]]), dims=["x", "y"]
        )
        atmo = AtmoConfig(
            aot=aot_2d, h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0}
        )
        with pytest.raises(ValueError, match="deduplicate_dims"):
            sweeper.collect(atmo, scalar_geo)

    def test_dedup_reduces_to_unique_combinations(self, scalar_geo):
        """Deduplication collapses a (3, 2) spatial grid to unique (aot, rh) pairs."""
        x, y = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])
        # 6 pixels but only 3 unique (aot, rh) pairs
        aot_2d = xr.DataArray(
            [[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]], dims=["x", "y"],
            coords={"x": x, "y": y},
        )
        rh_2d = xr.DataArray(
            [[50.0, 60.0], [50.0, 70.0], [60.0, 70.0]], dims=["x", "y"],
            coords={"x": x, "y": y},
        )
        atmo = AtmoConfig(
            aot=aot_2d, h=2.0, rh=rh_2d, href=2.0, species={"aerosol": 1.0}
        )
        sweeper = DimSweeper(
            scalar_dims=["aot", "rh", "sza", "vza", "saa", "vaa"],
            vector_dims=["h"],
            deduplicate_dims=["x", "y"],
        )
        dim_arrays, inverse_map = sweeper.collect(atmo, scalar_geo)

        assert inverse_map is not None
        assert inverse_map.dims == ("x", "y")
        assert inverse_map.shape == (3, 2)
        # 3 unique pairs out of 6 pixels
        n_unique = int(inverse_map.max()) + 1
        assert n_unique == 3, f"Expected 3 unique pairs, got {n_unique}"

    def test_dedup_with_no_spatial_config_raises(self, scalar_atmo, scalar_geo):
        """deduplicate_dims set but no config param lives on those dims raises."""
        sweeper = DimSweeper(
            scalar_dims=["aot", "sza"],
            vector_dims=["h"],
            deduplicate_dims=["x", "y"],
        )
        with pytest.raises(ValueError, match="None of the provided configs"):
            sweeper.collect(scalar_atmo, scalar_geo)


# ---------------------------------------------------------------------------
# DimSweeper — apply
# ---------------------------------------------------------------------------


class TestDimSweeperApply:
    def test_scalar_config_returns_scalar_result(self, sweeper, scalar_atmo, scalar_geo):
        """Scalar config produces a result with one value per swept dim."""
        dim_arrays, inverse_map = sweeper.collect(scalar_atmo, scalar_geo)

        def fake_core(h, rh, aot, sza, vza, saa, vaa):
            return np.float32(aot * 10.0)

        result = sweeper.apply(fake_core, [], dim_arrays, inverse_map)
        assert result.shape == (1, 1, 1, 1, 1)  # one value per scalar dim
        assert float(result.mean()) == pytest.approx(2.0, rel=1e-5)

    def test_1d_sweep_reconstructs_grid(self, sweeper, scalar_geo):
        """Sweeping over a 1-D aot produces a result with an aot dimension."""
        atmo = AtmoConfig(
            aot=[0.1, 0.2, 0.3], h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0}
        )
        dim_arrays, inverse_map = sweeper.collect(atmo, scalar_geo)

        def fake_core(h, rh, aot, sza, vza, saa, vaa):
            return np.float32(aot)

        result = sweeper.apply(fake_core, [], dim_arrays, inverse_map)
        assert "aot" in result.dims
        np.testing.assert_allclose(
            result.isel(sza=0, vza=0, saa=0, vaa=0).values.ravel(),
            [0.1, 0.2, 0.3],
            rtol=1e-5,
        )

    def test_dedup_result_reconstructed_to_spatial_shape(self, scalar_geo):
        """Dedup result is reconstructed to the original (x, y) spatial shape."""
        x, y = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])
        aot_2d = xr.DataArray(
            [[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]], dims=["x", "y"],
            coords={"x": x, "y": y},
        )
        rh_2d = xr.DataArray(
            [[50.0, 60.0], [50.0, 70.0], [60.0, 70.0]], dims=["x", "y"],
            coords={"x": x, "y": y},
        )
        atmo = AtmoConfig(
            aot=aot_2d, h=2.0, rh=rh_2d, href=2.0, species={"aerosol": 1.0}
        )
        sweeper = DimSweeper(
            scalar_dims=["aot", "rh", "sza", "vza", "saa", "vaa"],
            vector_dims=["h"],
            deduplicate_dims=["x", "y"],
        )
        dim_arrays, inverse_map = sweeper.collect(atmo, scalar_geo)

        def fake_core(h, aot, rh, sza, vza, saa, vaa):
            return np.float32(aot * 10.0 + rh * 0.1)

        result = sweeper.apply(fake_core, [], dim_arrays, inverse_map)

        assert "x" in result.dims
        assert "y" in result.dims
        assert result.sizes["x"] == 3
        assert result.sizes["y"] == 2

        # Verify pixel values match direct computation
        for xi, xv in enumerate(x):
            for yi, yv in enumerate(y):
                aot_v = float(aot_2d.sel(x=xv, y=yv))
                rh_v = float(rh_2d.sel(x=xv, y=yv))
                expected = aot_v * 10.0 + rh_v * 0.1
                got = float(result.sel(x=xv, y=yv).mean())
                assert got == pytest.approx(expected, rel=1e-4), (
                    f"Mismatch at (x={xv}, y={yv}): expected {expected}, got {got}"
                )

    def test_vector_dims_passed_as_array(self, sweeper, scalar_geo):
        """Vector dims arrive as np.ndarray inside _core_band, not as scalars."""
        atmo = AtmoConfig(
            aot=0.2, h=[1.0, 2.0, 3.0], rh=[40.0, 60.0], href=2.0,
            species={"aerosol": 1.0},
        )
        dim_arrays, inverse_map = sweeper.collect(atmo, scalar_geo)

        received_h_shape = {}
        received_rh_shape = {}

        def fake_core(h, rh, aot, sza, vza, saa, vaa):
            received_h_shape["shape"] = h.shape
            received_rh_shape["shape"] = rh.shape
            return np.float32(0.0)

        sweeper.apply(fake_core, [], dim_arrays, inverse_map)
        assert received_h_shape["shape"] == (3,), "h should arrive as 1-D array"
        assert received_rh_shape["shape"] == (2,), "rh should arrive as 1-D array"


# ---------------------------------------------------------------------------
# SceneModuleSweep integration
# ---------------------------------------------------------------------------


class FakeSmartGModule(SceneModuleSweep):
    """Minimal SceneModuleSweep subclass for testing.

    Computes rho_atm = aot * 10 + rho_s.mean() (fake physics).
    """

    required_vars = ["rho_s"]
    output_vars = ["rho_atm"]
    scalar_dims = ["aot", "sza", "vza", "saa", "vaa"]
    vector_dims = ["h", "rh"]

    def __init__(self, atmo, geo, **kwargs):
        super().__init__(**kwargs)
        self._atmo = atmo
        self._geo = geo

    def _get_configs(self):
        return (self._atmo, self._geo)

    def _core_band(self, rho_s, h, rh, aot, sza, vza, saa, vaa):
        return (rho_s + np.float32(aot * 10.0)).astype(np.float32)


class TestSceneModuleSweep:
    def test_output_var_written_to_scene(self, scalar_atmo, scalar_geo):
        """FakeSmartGModule writes rho_atm into every band dataset."""
        scene = random_image_dict(
            bands=[S2Band.B02, S2Band.B03], variables=["rho_s"], res_km=0.01, n=8, seed=0
        )
        module = FakeSmartGModule(scalar_atmo, scalar_geo)
        result = module(scene)

        for band in [S2Band.B02, S2Band.B03]:
            assert "rho_atm" in result[band]

    def test_output_values_correct(self, scalar_atmo, scalar_geo):
        """rho_atm equals rho_s + aot * 10 for each pixel."""
        scene = random_image_dict(
            bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=42
        )
        rho_s = scene[S2Band.B02]["rho_s"].values.copy()
        module = FakeSmartGModule(scalar_atmo, scalar_geo)
        result = module(scene)

        rho_atm = result[S2Band.B02]["rho_atm"]
        # squeeze scalar swept dims to get (y, x)
        rho_atm_spatial = rho_atm.squeeze()
        np.testing.assert_allclose(
            rho_atm_spatial.values, rho_s + 0.2 * 10.0, rtol=1e-5
        )

    def test_missing_required_var_raises(self, scalar_atmo, scalar_geo):
        """SceneModuleSweep raises MissingVariableError when rho_s is absent."""
        scene = random_image_dict(
            bands=[S2Band.B02], variables=["rho_toa"], res_km=0.01, n=8, seed=0
        )
        module = FakeSmartGModule(scalar_atmo, scalar_geo)
        with pytest.raises(MissingVariableError):
            module(scene)

    def test_provenance_is_stamped(self, scalar_atmo, scalar_geo):
        """Output DataArrays carry _adjeff_provenance after compute."""
        scene = random_image_dict(
            bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=0
        )
        result = FakeSmartGModule(scalar_atmo, scalar_geo)(scene)
        prov = result[S2Band.B02]["rho_atm"].attrs.get("_adjeff_provenance")
        assert prov is not None
        assert prov["module"] == "FakeSmartGModule"
        assert "key" in prov

    def test_cache_hit(self, tmp_path, scalar_atmo, scalar_geo):
        """Second call with identical input hits the cache."""
        scene = random_image_dict(
            bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=0
        )
        cache = CacheStore(tmp_path)
        module = FakeSmartGModule(scalar_atmo, scalar_geo, cache=cache)
        r1 = module(scene)
        r2 = module(scene)
        np.testing.assert_array_equal(
            r1[S2Band.B02]["rho_atm"].values,
            r2[S2Band.B02]["rho_atm"].values,
        )

    def test_sweep_over_aot_adds_dim(self, scalar_geo):
        """Sweeping over multiple aot values adds an aot dimension to the output."""
        atmo = AtmoConfig(
            aot=[0.1, 0.2, 0.3], h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0}
        )
        scene = random_image_dict(
            bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=8, seed=0
        )
        result = FakeSmartGModule(atmo, scalar_geo)(scene)
        rho_atm = result[S2Band.B02]["rho_atm"]
        assert "aot" in rho_atm.dims
        assert rho_atm.sizes["aot"] == 3

    def test_chunks_splits_vector_dim(self, scalar_geo):
        """Chunking a vector dim splits calls and recombines along that dim."""
        atmo = AtmoConfig(
            aot=0.2, h=[1.0, 2.0, 3.0, 4.0], rh=50.0,
            href=2.0, species={"aerosol": 1.0},
        )
        call_log: list = []

        class FakeModuleWithHOutput(SceneModuleSweep):
            # no required_vars — output lives entirely on swept/vector dims
            required_vars: list[str] = []
            output_vars = ["rho_atm"]
            scalar_dims = ["aot", "sza", "vza", "saa", "vaa"]
            vector_dims = ["h", "rh"]

            def __init__(self, atmo, geo, **kwargs):
                super().__init__(**kwargs)
                self._atmo = atmo
                self._geo = geo

            def _get_configs(self):
                return (self._atmo, self._geo)

            def _core_band(self, h, rh, aot, sza, vza, saa, vaa):
                call_log.append(h.copy())
                # return an array along h: shape (len(h),)
                return (h + float(aot) * 10.0).astype(np.float32)

        scene = random_image_dict(
            bands=[S2Band.B02], variables=["rho_s"], res_km=0.01, n=4, seed=0
        )
        module = FakeModuleWithHOutput(
            atmo, scalar_geo,
            chunks={"h": 2},
        )
        result = module(scene)
        assert "rho_atm" in result[S2Band.B02]
        assert result[S2Band.B02]["rho_atm"].sizes["h"] == 4
        assert len(call_log) == 2  # 4 h values split into 2 chunks of 2
