"""
Test script for all radiative samplers.

Covers:
- Scalar configs (single values everywhere)
- Vector sweeps over atmospheric params
- Vector sweeps over angular params
- Mixed scalar/vector
- Large sweeps with chunking
- Spatial [x, y] dims for analytical modules
- Note on the cartesian product limitation of SmartG for spatial angular dims

NOTE ON DESIGN LIMITATION
--------------------------
SmartG creates a cartesian product between the atmosphere (wl, aot, rh, h, href
flattened as "wavelength") and the angular dims (vza, sza). This is fine when all
dims are 1D.

However, if vza or sza carry spatial dims [x, y], the bundle passes them as a flat
1D vector to SmartG (after stacking), but SmartG would produce
  n_atmo_combos × n_vza_pixels
outputs instead of the expected n_pixels. The unstack step would then be wrong.

The only fix in that case is to force vza and sza into scalar_dims so that the
bundle iterates over them one at a time. This is a SmartG API constraint, not
something we can absorb in post-processing.

For analytical modules (tdir_up, tdir_down), there is no such limitation since
they never call SmartG with angular dims: vza/sza enter only through cos(angle),
so spatial dims work fine.
"""

import logging

import structlog
import numpy as np
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import ImageDict, S2Band
from adjeff.modules.samplers import (
    RhoAtmSampler,
    SphAlbSampler,
    TdifDownSampler,
    TdifUpSampler,
    TdirDownSampler,
    TdirUpSampler,
)
from adjeff.utils.logger import MultilineConsoleRenderer

logging.basicConfig(level=logging.INFO)



structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        MultilineConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    logger_factory=structlog.PrintLoggerFactory(),
)



BANDS = [S2Band.B02, S2Band.B03]
COMMON_SMARTG = dict(remove_rayleigh=False, n_ph=int(1e4))


def run(label: str, samplers: list, scene: ImageDict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for s in samplers:
        scene = s._compute(scene)
    for band in scene.bands:
        print(f"\n  [{band}]")
        print(scene[band])


def make_scene(spectral_config: SpectralConfig) -> ImageDict:
    return ImageDict({band: xr.Dataset() for band in spectral_config.bands})


# ---------------------------------------------------------------------------
# 1. All scalar (single value per param)
# ---------------------------------------------------------------------------
def test_scalar_all() -> None:
    spectral_config = SpectralConfig.from_bands([S2Band.B02])
    atmo = AtmoConfig(
        aot=xr.DataArray([0.1], dims=["aot"]),
        rh=xr.DataArray([50.0], dims=["rh"]),
        h=xr.DataArray([0.0], dims=["h"]),
        href=xr.DataArray([2.0], dims=["href"]),
        species={"sulphate": 1.0},
    )
    geo = GeoConfig(
        sza=xr.DataArray([30.0], dims=["sza"]),
        vza=xr.DataArray([10.0], dims=["vza"]),
        saa=xr.DataArray([120.0], dims=["saa"]),
        vaa=xr.DataArray([120.0], dims=["vaa"]),
    )
    common = dict(spectral_config=spectral_config, atmo_config=atmo, **COMMON_SMARTG)
    run(
        "1. All scalar",
        [
            TdirUpSampler(geo_config=geo, **common),
            TdirDownSampler(geo_config=geo, **common),
            TdifUpSampler(geo_config=geo, **common),
            TdifDownSampler(geo_config=geo, **common),
            RhoAtmSampler(geo_config=geo, **common),
            SphAlbSampler(atmo_config=atmo, spectral_config=spectral_config, **COMMON_SMARTG),
        ],
        make_scene(spectral_config),
    )


# ---------------------------------------------------------------------------
# 2. Sweep over atmospheric params, single angle
# ---------------------------------------------------------------------------
def test_vector_atmo() -> None:
    spectral_config = SpectralConfig.from_bands([S2Band.B02])
    atmo = AtmoConfig(
        aot=xr.DataArray([0.05, 0.1, 0.3, 0.6], dims=["aot"]),
        rh=xr.DataArray([30.0, 60.0, 90.0], dims=["rh"]),
        h=xr.DataArray([0.0], dims=["h"]),
        href=xr.DataArray([2.0], dims=["href"]),
        species={"sulphate": 1.0},
    )
    geo = GeoConfig(
        sza=xr.DataArray([30.0], dims=["sza"]),
        vza=xr.DataArray([0.0], dims=["vza"]),
        saa=xr.DataArray([120.0], dims=["saa"]),
        vaa=xr.DataArray([120.0], dims=["vaa"]),
    )
    common = dict(spectral_config=spectral_config, atmo_config=atmo, **COMMON_SMARTG)
    run(
        "2. Vector atmo (aot×rh sweep), scalar angle",
        [
            TdirUpSampler(geo_config=geo, **common),
            TdirDownSampler(geo_config=geo, **common),
            SphAlbSampler(atmo_config=atmo, spectral_config=spectral_config, **COMMON_SMARTG),
        ],
        make_scene(spectral_config),
    )


# ---------------------------------------------------------------------------
# 3. Sweep over angular params, scalar atmo
# ---------------------------------------------------------------------------
def test_vector_angular() -> None:
    spectral_config = SpectralConfig.from_bands([S2Band.B02])
    atmo = AtmoConfig(
        aot=xr.DataArray([0.1], dims=["aot"]),
        rh=xr.DataArray([50.0], dims=["rh"]),
        h=xr.DataArray([0.0], dims=["h"]),
        href=xr.DataArray([2.0], dims=["href"]),
        species={"sulphate": 1.0},
    )
    geo = GeoConfig(
        sza=xr.DataArray([0.0, 20.0, 40.0, 60.0], dims=["sza"]),
        vza=xr.DataArray([0.0, 10.0, 20.0, 30.0], dims=["vza"]),
        saa=xr.DataArray([120.0], dims=["saa"]),
        vaa=xr.DataArray([120.0], dims=["vaa"]),
    )
    common = dict(spectral_config=spectral_config, atmo_config=atmo, **COMMON_SMARTG)
    run(
        "3. Vector angular (vza, sza sweep), scalar atmo",
        [
            TdirUpSampler(geo_config=geo, **common),
            TdirDownSampler(geo_config=geo, **common),
            TdifUpSampler(geo_config=geo, **common),
            TdifDownSampler(geo_config=geo, **common),
            RhoAtmSampler(geo_config=geo, **common),
        ],
        make_scene(spectral_config),
    )


# ---------------------------------------------------------------------------
# 4. Multi-band + mixed vector params
# ---------------------------------------------------------------------------
def test_multiband_mixed() -> None:
    spectral_config = SpectralConfig.from_bands(BANDS)
    atmo = AtmoConfig(
        aot=xr.DataArray([0.05, 0.2, 0.5], dims=["aot"]),
        rh=xr.DataArray([40.0, 70.0], dims=["rh"]),
        h=xr.DataArray([0.0, 0.5], dims=["h"]),
        href=xr.DataArray([2.0], dims=["href"]),
        species={"sulphate": 1.0},
    )
    geo = GeoConfig(
        sza=xr.DataArray([20.0, 45.0], dims=["sza"]),
        vza=xr.DataArray([0.0, 15.0], dims=["vza"]),
        saa=xr.DataArray([90.0], dims=["saa"]),
        vaa=xr.DataArray([90.0], dims=["vaa"]),
    )
    common = dict(spectral_config=spectral_config, atmo_config=atmo, **COMMON_SMARTG)
    run(
        "4. Multi-band + mixed vector (aot, rh, h, sza, vza)",
        [
            TdirUpSampler(geo_config=geo, **common),
            TdirDownSampler(geo_config=geo, **common),
        ],
        make_scene(spectral_config),
    )


# ---------------------------------------------------------------------------
# 5. Large sweep with chunking
# ---------------------------------------------------------------------------
def test_large_with_chunks() -> None:
    spectral_config = SpectralConfig.from_bands(BANDS)
    atmo = AtmoConfig(
        aot=xr.DataArray(np.linspace(0.01, 0.8, 8), dims=["aot"]),
        rh=xr.DataArray(np.linspace(20.0, 90.0, 5), dims=["rh"]),
        h=xr.DataArray([0.0], dims=["h"]),
        href=xr.DataArray([2.0], dims=["href"]),
        species={"sulphate": 1.0},
    )
    geo = GeoConfig(
        sza=xr.DataArray(np.linspace(0.0, 60.0, 4), dims=["sza"]),
        vza=xr.DataArray([0.0], dims=["vza"]),
        saa=xr.DataArray([120.0], dims=["saa"]),
        vaa=xr.DataArray([120.0], dims=["vaa"]),
    )
    common = dict(
        spectral_config=spectral_config,
        atmo_config=atmo,
        **COMMON_SMARTG,
        chunks={"wl": 4},
    )
    run(
        "5. Large sweep (8 aot × 5 rh × 4 sza × 2 wl) with chunks={'wl': 4}",
        [
            TdirUpSampler(geo_config=geo, **common),
            TdirDownSampler(geo_config=geo, **common),
        ],
        make_scene(spectral_config),
    )


# ---------------------------------------------------------------------------
# 6. Spatial [x, y] dims — analytical modules only (works correctly)
# ---------------------------------------------------------------------------
def test_spatial_dims_analytical() -> None:
    """
    tdir_up and tdir_down with spatial [x, y] dims on vza/sza/aot.
    SmartG is not involved in the angular step (only in OD), so the
    cartesian product issue does not apply.
    """
    spectral_config = SpectralConfig.from_bands([S2Band.B02])

    nx, ny = 3, 4
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)

    atmo = AtmoConfig(
        aot=xr.DataArray(
            np.random.uniform(0.05, 0.4, (nx, ny)),
            dims=["x", "y"],
            coords={"x": x, "y": y},
        ),
        rh=xr.DataArray([50.0], dims=["rh"]),
        h=xr.DataArray([0.0], dims=["h"]),
        href=xr.DataArray([2.0], dims=["href"]),
        species={"sulphate": 1.0},
    )
    geo = GeoConfig(
        sza=xr.DataArray(
            np.random.uniform(10.0, 60.0, (nx, ny)),
            dims=["x", "y"],
            coords={"x": x, "y": y},
        ),
        vza=xr.DataArray(
            np.random.uniform(0.0, 30.0, (nx, ny)),
            dims=["x", "y"],
            coords={"x": x, "y": y},
        ),
        saa=xr.DataArray([120.0], dims=["saa"]),
        vaa=xr.DataArray([120.0], dims=["vaa"]),
    )
    common = dict(spectral_config=spectral_config, atmo_config=atmo, **COMMON_SMARTG)
    run(
        "6. Spatial [x, y] dims — analytical only (tdir_up, tdir_down)",
        [
            TdirUpSampler(geo_config=geo, **common),
            TdirDownSampler(geo_config=geo, **common),
        ],
        make_scene(spectral_config),
    )


# ---------------------------------------------------------------------------
# 7. Spatial [x, y] + deduplication
# ---------------------------------------------------------------------------
def test_spatial_dims_deduplication() -> None:
    """
    Spatial [x, y] dims with deduplicate_dims=["x", "y"] to avoid recomputing
    identical atmosphere combinations.
    """
    spectral_config = SpectralConfig.from_bands([S2Band.B02])

    nx, ny = 4, 4
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)

    # Only 3 distinct aot values spread across 16 pixels → deduplication helps
    aot_grid = np.random.choice([0.05, 0.2, 0.5], size=(nx, ny))
    atmo = AtmoConfig(
        aot=xr.DataArray(aot_grid, dims=["x", "y"], coords={"x": x, "y": y}),
        rh=xr.DataArray([50.0], dims=["rh"]),
        h=xr.DataArray([0.0], dims=["h"]),
        href=xr.DataArray([2.0], dims=["href"]),
        species={"sulphate": 1.0},
    )
    geo = GeoConfig(
        sza=xr.DataArray([30.0], dims=["sza"]),
        vza=xr.DataArray([0.0], dims=["vza"]),
        saa=xr.DataArray([120.0], dims=["saa"]),
        vaa=xr.DataArray([120.0], dims=["vaa"]),
    )
    common = dict(
        spectral_config=spectral_config,
        atmo_config=atmo,
        **COMMON_SMARTG,
        deduplicate_dims=["x", "y"],
    )
    run(
        "7. Spatial [x, y] + deduplicate_dims=['x', 'y']",
        [
            TdirUpSampler(geo_config=geo, **common),
        ],
        make_scene(spectral_config),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_scalar_all()
    test_vector_atmo()
    test_vector_angular()
    test_multiband_mixed()
    test_large_with_chunks()
    test_spatial_dims_analytical()
    test_spatial_dims_deduplication()
