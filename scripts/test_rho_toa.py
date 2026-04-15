"""Test / observation script for src/adjeff/modules/samplers/rho_toa.py.

Covers:
- _radial_sensors          : sensor positions for a set of radial distances
- SmartgSampler_Rho_toa    : _compute() called on a gaussian ImageDict so we
                             can observe the intermediate radial profile printed
                             inside rho_toa() at its current breakpoint
"""

import logging

import numpy as np
import structlog
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import S2Band, disk_image_dict
from adjeff.modules.samplers.rho_toa import SmartgSampler_Rho_toa, _radial_sensors
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

BAND = S2Band.B02
SAT_HEIGHT = 786.0  # km


# ---------------------------------------------------------------------------
# 1. _radial_sensors
# ---------------------------------------------------------------------------
def test_radial_sensors() -> None:
    print(f"\n{'='*60}")
    print("  1. _radial_sensors")
    print(f"{'='*60}")

    r_vals = np.linspace(0.0, 0.5, 6)  # km
    vza, vaa = 10.0, 120.0

    sensors = _radial_sensors(r_vals, vza=vza, vaa=vaa, sat_height=SAT_HEIGHT)

    print(f"  vza={vza}°  vaa={vaa}°  sat_height={SAT_HEIGHT} km")
    for r, s in zip(r_vals, sensors):
        print(s.dict)


# ---------------------------------------------------------------------------
# 2. SmartgSampler_Rho_toa — observe the approximate radial profile
# ---------------------------------------------------------------------------
def test_rho_toa_profile() -> None:
    """Call _compute() on a gaussian scene to reach print(profile) / sys.exit()."""
    print(f"\n{'='*60}")
    print("  2. SmartgSampler_Rho_toa — radial profile observation")
    print(f"{'='*60}")

    spectral_config = SpectralConfig.from_bands([BAND])

    scene = disk_image_dict(
        radius=5.0,
        res_km=0.12,
        rho_min=0.0,
        rho_max=0.30,
        bands=[BAND],
        var="rho_s",
        extent_km=200.0,
    )

    atmo_cfg = AtmoConfig(
        aot=xr.DataArray(np.linspace(0.0, 1.0, 5), dims=["aot"]),
        rh=xr.DataArray([50.0], dims=["rh"]),
        h=xr.DataArray([0.0], dims=["h"]),
        href=xr.DataArray([2.0], dims=["href"]),
        species={"sulphate": 1.0},
    )
    geo_cfg = GeoConfig(
        sza=xr.DataArray([30.0], dims=["sza"]),
        vza=xr.DataArray([10.0], dims=["vza"]),
        saa=xr.DataArray([120.0], dims=["saa"]),
        vaa=xr.DataArray([120.0], dims=["vaa"]),
        sat_height=SAT_HEIGHT,
    )

    sampler = SmartgSampler_Rho_toa(
        atmo_config=atmo_cfg,
        geo_config=geo_cfg,
        spectral_config=spectral_config,
        remove_rayleigh=False,
        afgl_type="afgl_exp_h8km",
        nr=500,
        n_ph=int(1e5),
    )

    scene = sampler._compute(scene)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.flatten()

    scene[S2Band.B02]["rho_s"].plot.imshow(ax=axes[0])
    scene[S2Band.B02]["rho_toa"].isel(aot=0).squeeze().plot.imshow(ax=axes[1])
    scene[S2Band.B02]["rho_toa"].isel(aot=1).squeeze().plot.imshow(ax=axes[2])
    scene[S2Band.B02]["rho_toa"].isel(aot=2).squeeze().plot.imshow(ax=axes[3])
    scene[S2Band.B02]["rho_toa"].isel(aot=3).squeeze().plot.imshow(ax=axes[4])
    scene[S2Band.B02]["rho_toa"].isel(aot=4).squeeze().plot.imshow(ax=axes[5])
    plt.show()
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_radial_sensors()
    test_rho_toa_profile()
