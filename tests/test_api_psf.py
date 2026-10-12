"""`apply_psf`, on the two scene shapes a caller actually has.

Neither it nor `fit_psf` had a test, which is how the Smart-G 2.0
migration left them broken until a notebook ran them.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from adjeff.api import apply_psf
from adjeff.core import ImageDict, KingPSF, PSFGrid, S2Band, freeze

BAND, RES, N = S2Band.B03, 1.0, 21
SWEPT = {"aot": 0.4, "rh": 50.0, "h": 0.0, "href": 2.0}
ANGLES = {"sza": 40.0, "vza": 0.0, "saa": 0.0, "vaa": 0.0}


def _tree() -> xr.DataTree:
    """Return a frozen tree holding one King kernel."""
    psf = KingPSF(PSFGrid(RES, N), BAND, sigma=0.3, gamma=1.3)
    return freeze({BAND: psf})


def _scene(carries_atmosphere: bool) -> ImageDict:
    """Return a scene with what `Unif2Surface` reads, swept over four dims.

    `carries_atmosphere` adds the eight fields a loaded product holds as
    variables, which is the shape `fit_psf` requires of its input.
    """
    xy = (np.arange(N) - N // 2) * RES
    field = xr.DataArray(
        np.full((1, 1, 1, 1, N, N), 0.4),
        dims=[*SWEPT, "y", "x"],
        coords={**{k: [v] for k, v in SWEPT.items()}, "y": xy, "x": xy},
    )
    ds = xr.Dataset(
        {
            name: field.copy()
            for name in ("rho_s", "rho_unif", "tdir_up", "tdif_up", "sph_alb")
        }
    )
    if carries_atmosphere:
        ds = ds.squeeze().reset_coords(list(SWEPT))
        for name, value in ANGLES.items():
            ds[name] = xr.DataArray(value)
    return ImageDict({BAND: ds})


class TestApplyPsf:
    """Where the estimate lands, and on which scenes it runs at all."""

    def test_the_truth_is_left_where_the_scene_put_it(self):
        scene = _scene(carries_atmosphere=False)
        truth = scene[BAND]["rho_s"].copy()

        out = apply_psf(scene, _tree(), BAND, device="cpu")

        assert "rho_s_est" in out[BAND]
        xr.testing.assert_identical(out[BAND]["rho_s"], truth)

    def test_an_empty_rename_writes_the_model_s_own_names(self):
        scene = _scene(carries_atmosphere=False)

        out = apply_psf(scene, _tree(), BAND, rename={}, device="cpu")

        assert "rho_s_est" not in out[BAND]
        assert "rho_s" in out[BAND]

    def test_a_scene_carrying_its_atmosphere_is_accepted(self):
        """The shape `fit_psf` demands has to be the shape `apply_psf` takes."""
        scene = _scene(carries_atmosphere=True)

        out = apply_psf(scene, _tree(), BAND, device="cpu")

        assert "rho_s_est" in out[BAND]


class TestSampledKernel:
    """A sampled kernel is a kernel: on a PSF grid, with its provenance."""

    @staticmethod
    def _wu_like() -> xr.DataTree:
        """Return a tree shaped as `sample_psf_atm` now returns one."""
        from adjeff.core import psf_tree

        grid = PSFGrid(RES, N)
        kernel = xr.DataArray(
            np.full((N, N), 1.0 / (N * N)),
            dims=["y_psf", "x_psf"],
            coords=grid.as_coords(),
            attrs={"adjeff:kind": "arbitrary", "band": BAND.id},
        )
        return psf_tree({BAND: kernel})

    def test_apply_psf_accepts_it(self):
        """It carried the scene's own x and y, which collided on convolution."""
        out = apply_psf(
            _scene(carries_atmosphere=False), self._wu_like(), BAND, device="cpu"
        )

        assert "rho_s_est" in out[BAND]

    def test_it_is_not_mistaken_for_an_analytical_kernel(self):
        """`grid_share` extrapolates past the grid, which a sample cannot."""
        from adjeff.analysis._plane import grid_share
        from adjeff.exceptions import ConfigurationError

        kernel = self._wu_like()[BAND.id].ds["kernel"]
        with pytest.raises(ConfigurationError):
            grid_share(kernel, radius_km=RES * N)


class TestOneSceneForBoth:
    """The scene the forward pipeline writes is the scene `fit_psf` reads."""

    def test_the_pipeline_records_the_azimuths_it_simulated_at(self):
        """`saa` and `vaa` are sampler statics, so nothing else carries them."""
        from adjeff.api import make_full_config
        from adjeff.api._pipeline import _with_azimuths

        cfg = make_full_config(bands=[BAND], saa=120.0, vaa=30.0)
        out = _with_azimuths(_scene(carries_atmosphere=False), cfg["geo_config"])

        assert float(out[BAND].coords["saa"]) == 120.0
        assert float(out[BAND].coords["vaa"]) == 30.0

    def test_load_config_reads_the_eight_fields_off_coordinates(self):
        """They need not be data variables, which is what clashed on merge."""
        from adjeff.api import load_config, make_full_config
        from adjeff.api._pipeline import _with_azimuths

        cfg = make_full_config(bands=[BAND], saa=0.0, vaa=0.0)
        scene = _with_azimuths(_scene(carries_atmosphere=False), cfg["geo_config"])
        ds = scene[BAND].assign_coords(sza=40.0, vza=0.0)
        scene[BAND] = ds

        read = load_config(scene, BAND, aggregate=True)

        assert not [n for n in ("aot", "saa") if n in ds.data_vars]
        assert float(read["atmo_config"].aot.squeeze()) == SWEPT["aot"]
