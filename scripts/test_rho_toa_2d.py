"""End-to-end test for SmartgSampler_Rho_toa (2D, no symmetry assumption).

Sections
--------
1. _grid_sensors      — check sensor (x, y) positions for a small grid.
2. custom_environment — build an Albedo_map from a small random scene.
3. SmartgSampler_Rho_toa — full run on an arbitrary scene, visualisation
                            of rho_s vs rho_toa for all AOT values.
4. Sym vs 2D comparison — run both samplers on the same Gaussian scene
                           and compare radial profiles.
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import structlog
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig
from adjeff.core import S2Band, gaussian_image_dict, random_image_dict
from adjeff.modules.samplers.rho_toa import (
    SmartgSampler_Rho_toa,
)
from adjeff.utils import CacheStore
from adjeff.utils.logger import MultilineConsoleRenderer

assert "SMARTG_DIR_AUXDATA" in os.environ, (
    "Set SMARTG_DIR_AUXDATA to the Smart-G auxiliary data directory."
)

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

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SAT_HEIGHT = 786.0  # km
CACHE = CacheStore("/tmp/adjeff_rho_toa_2d_cache")

AOT_VALUES = [0.1, 0.4, 0.8]

atmo_cfg = AtmoConfig(
    aot=xr.DataArray(AOT_VALUES, dims=["aot"]),
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


def test_rho_toa_2d() -> None:
    print(f"\n{'='*60}")
    print("  3. SmartgSampler_Rho_toa — arbitrary scene")
    print(f"{'='*60}")

    # Build an arbitrary scene: random reflectance map in [0.05, 0.60]
    scene = gaussian_image_dict(
        sigma=2.0, 
        res_km=0.120, 
        bands=[S2Band.B02],
        var="rho_s",
        n=199,
        analytical=False
    )

    sampler = SmartgSampler_Rho_toa(
        atmo_config=atmo_cfg,
        geo_config=geo_cfg,
        remove_rayleigh=False,
        afgl_type="afgl_exp_h8km",
        nx=20,
        ny=199,           # 30×30 = 900 sensors
        topleft_pix=(90, 0),
        n_ph=int(1e4),
        n_alb=1000,
    )

    scene = sampler(scene)

    rho_s = scene[S2Band.B02]["rho_s"]
    rho_toa = scene[S2Band.B02]["rho_toa"].squeeze()  # drop scalar rh, h, href, ...

    n_aot = len(AOT_VALUES)
    fig, axes = plt.subplots(1, n_aot + 1, figsize=(4 * (n_aot + 1), 4))

    rho_s.plot.imshow(ax=axes[0], cmap="YlGn", vmin=0.0, vmax=0.7)
    axes[0].set_title("rho_s (input)")

    for i, aot_val in enumerate(AOT_VALUES):
        rho_toa.sel(aot=aot_val, method="nearest").plot.imshow(
            ax=axes[i + 1], cmap="YlGn", vmin=0.0, vmax=0.4
        )
        axes[i + 1].set_title(f"rho_toa  AOT={aot_val}")

    plt.suptitle(f"SmartgSampler_Rho_toa — {S2Band.B02}  (nx=30, ny=30, arbitrary scene)")
    plt.tight_layout()
    plt.savefig("/tmp/rho_toa_2d_arbitrary.png", dpi=120)
    logger.info("Figure saved.", path="/tmp/rho_toa_2d_arbitrary.png")
    plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_rho_toa_2d()
