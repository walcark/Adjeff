"""Compare computed radiative parameters against pre-validated LUT references.

LUTs were computed and validated with the SOS_ABS_V5.0 successive order of
scattering model.  They serve as regression targets for each aerosol species.

Key mapping between scene and LUT
----------------------------------
- Each ``scene[band]`` variable has the ``wl`` dimension removed (the band
  already encodes the wavelength).  The reference LUT is selected at
  ``wl=band.wl_nm`` before comparison.
- ``href`` is a scene dimension absent from the LUT.  A single ``href``
  value is used so the dimension can be squeezed away before comparing.
- Only variables present in *both* the computed scene band and the LUT are
  compared — extra variables in either dataset are silently ignored.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import ImageDict, S2Band, SensorBand
from adjeff.modules.samplers import RadiativePipeline

DATA_PATH = Path(__file__).resolve().parent / "data" / "radiative_luts"

# LUT variables to compare (all variables present in the reference files)
LUT_VARS = [
    "rho_atm",
    "tdir_up",
    "tdir_down",
    "tdif_up",
    "tdif_down",
    "sph_alb",
]

# tdir_* are analytical (exp(-OD/cos(angle))) → tight tolerance.
# tdif_* and sph_alb are full Monte-Carlo diffuse integrals → looser tolerance
# to account for variance between Smart-G and the SOS_ABS_V5.0 reference.
_RTOL: dict[str, float] = {
    "tdir_down": 1e-2,
    "tdir_up": 1e-2,
    "rho_atm": 2e-2,
    "tdif_up": 1e-2,
    "tdif_down": 1e-2,
    "sph_alb": 1e-2,
}

# Coordinate values must be a subset of the LUT grid
# LUT sza: [0, 20, 40, 60, 80] — vza: [0, 4, 8, 12, 16]
# LUT aot: [0, 0.05, …, 0.8]   — rh:  [0, 50, 75, 85, 90, 95]
# LUT h:   [0, 1, 2, 3]
TEST_SZA = xr.DataArray([0.0, 40.0], dims=["sza"])
TEST_VZA = xr.DataArray([0.0, 8.0], dims=["vza"])
TEST_AOT = xr.DataArray([0.05, 0.3], dims=["aot"])
TEST_RH = xr.DataArray([0.0, 50.0], dims=["rh"])
TEST_H = xr.DataArray([0.0, 2.0], dims=["h"])
TEST_HREF = xr.DataArray([2.0], dims=["href"])  # single value; not a LUT dim

# Bands whose wl_nm values exist in the LUT (443, 490, 560, …, 1610, 2190)
TEST_BANDS: list[SensorBand] = [S2Band.B01, S2Band.B03, S2Band.B8A, S2Band.B11]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _load_lut(specie: str) -> xr.Dataset:
    """Load the reference LUT for *specie*."""
    return xr.load_dataset(DATA_PATH / f"lut__specie-{specie}.nc")


def _compare_variable(
    scene_ds: xr.Dataset,
    ref_ds: xr.Dataset,
    var: str,
    band: SensorBand,
    *,
    rtol: float | None = None,
    atol: float = 1e-4,
) -> None:
    """Assert that ``scene_ds[var]`` matches ``ref_ds[var]`` at *band*.

    Steps
    -----
    1. Drop ``href`` from the scene variable (not a LUT dimension).
    2. Select ``wl=band.wl_nm`` from the reference.
    3. Inner-join on all shared coordinates so only sampled points are tested.
    4. Reorder ref dimensions to match the scene order and compare.
    """
    if rtol is None:
        rtol = _RTOL.get(var, 1e-2)

    scene_var: xr.DataArray = scene_ds[var]
    if "href" in scene_var.dims:
        scene_var = scene_var.squeeze("href", drop=True)

    ref_var: xr.DataArray = ref_ds[var].sel(wl=band.wl_nm)

    scene_aligned, ref_aligned = xr.align(scene_var, ref_var, join="inner")
    ref_aligned = ref_aligned.transpose(*scene_aligned.dims)

    np.testing.assert_allclose(
        scene_aligned.values,
        ref_aligned.values,
        rtol=rtol,
        atol=atol,
        err_msg=f"{var} mismatch for {band} (wl={band.wl_nm} nm)",
    )


# ------------------------------------------------------------------
# Fixtures — run the pipeline once per species, share across tests
# ------------------------------------------------------------------


def _run_pipeline(species: dict[str, float]) -> ImageDict:
    atmo_cfg = AtmoConfig(
        aot=TEST_AOT,
        rh=TEST_RH,
        h=TEST_H,
        href=TEST_HREF,
        species=species,
    )
    geo_cfg = GeoConfig(
        sza=TEST_SZA,
        vza=TEST_VZA,
        saa=xr.DataArray([120.0], dims=["saa"]),
        vaa=xr.DataArray([120.0], dims=["vaa"]),
    )
    spectral_cfg = SpectralConfig.from_bands(TEST_BANDS)
    pipeline = RadiativePipeline(
        atmo_config=atmo_cfg,
        geo_config=geo_cfg,
        spectral_config=spectral_cfg,
        n_ph_rho_atm=int(1e8),
        n_ph_sph_alb=int(1e8),
        n_ph_tdif_down=int(1e8),
        n_ph_tdif_up=int(1e8),
        remove_rayleigh=False,
    )
    return pipeline(ImageDict({}))


@pytest.fixture(scope="module")
def scene_sulphate() -> ImageDict:
    return _run_pipeline({"sulphate": 1.0})


@pytest.fixture(scope="module")
def ref_sulphate() -> xr.Dataset:
    return _load_lut("sulphate")


@pytest.fixture(scope="module")
def scene_blackcar() -> ImageDict:
    return _run_pipeline({"blackcar": 1.0})


@pytest.fixture(scope="module")
def ref_blackcar() -> xr.Dataset:
    return _load_lut("blackcar")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.parametrize("band", TEST_BANDS)
@pytest.mark.parametrize("var", LUT_VARS)
def test_sulphate_variable(
    scene_sulphate: ImageDict,
    ref_sulphate: xr.Dataset,
    band: SensorBand,
    var: str,
) -> None:
    """Radiative variables match the sulphate LUT for each test band."""
    scene_ds = scene_sulphate[band]
    if var not in scene_ds:
        pytest.skip(f"{var} not computed in scene[{band}]")
    if var not in ref_sulphate:
        pytest.skip(f"{var} absent from sulphate LUT")
    _compare_variable(scene_ds, ref_sulphate, var, band)


@pytest.mark.parametrize("band", TEST_BANDS)
@pytest.mark.parametrize("var", LUT_VARS)
def test_blackcar_variable(
    scene_blackcar: ImageDict,
    ref_blackcar: xr.Dataset,
    band: SensorBand,
    var: str,
) -> None:
    """Radiative variables match the blackcar LUT for each test band."""
    scene_ds = scene_blackcar[band]
    if var not in scene_ds:
        pytest.skip(f"{var} not computed in scene[{band}]")
    if var not in ref_blackcar:
        pytest.skip(f"{var} absent from blackcar LUT")
    _compare_variable(scene_ds, ref_blackcar, var, band)
