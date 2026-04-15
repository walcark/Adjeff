"""
Radiative pipeline test on a 1000×1000 spatial grid.

Grid: 1000×1000 pixels divided into 5×5 zones of 200×200 pixels each.
Each zone has a unique (aot, h) pair → 25 unique combinations,
dedup reduces to 25 × 2 (rh) × 3 (href) unique atmosphere states.

Params: rh=[50, 60], href=[1.0, 2.0, 3.0]
        sza=[0.0, 40.0, 80.0], vza=[10.0, 30.0, 60.0], wl=560 nm (B03)

Output: 3×3 plot
    Row 0 — rho_atm: one panel per sza (vza=10, href=2.0)
    Row 1 — tdif_up: one panel per vza (href=2.0)
    Row 2 — sph_alb: one panel per href
    All at rh=50.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import ImageDict, S2Band
from adjeff.modules.samplers import RadiativePipeline

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BAND = S2Band.B03  # 560 nm
NX, NY = 1000, 1000
N_ZONES = 5  # 5×5 = 25 zones, each 200×200 pixels

AOT_VALS = [0.05, 0.1, 0.2, 0.3, 0.5]
H_VALS = [0.0, 0.5, 1.0, 2.0, 3.0]

SZA_VALS = [0.0, 40.0, 80.0]
VZA_VALS = [10.0, 30.0, 60.0]
HREF_VALS = [1.0, 2.0, 3.0]

# ---------------------------------------------------------------------------
# Build spatial aot and h grids (zone-based)
# ---------------------------------------------------------------------------
x = np.arange(NX, dtype=float)
y = np.arange(NY, dtype=float)
zone_size = NX // N_ZONES

aot_grid = np.empty((NX, NY))
h_grid = np.empty((NX, NY))

for iz in range(N_ZONES):
    for jz in range(N_ZONES):
        xs = slice(iz * zone_size, (iz + 1) * zone_size)
        ys = slice(jz * zone_size, (jz + 1) * zone_size)
        aot_grid[xs, ys] = AOT_VALS[iz]
        h_grid[xs, ys] = H_VALS[jz]

atmo = AtmoConfig(
    aot=xr.DataArray(aot_grid, dims=["x", "y"], coords={"x": x, "y": y}),
    h=xr.DataArray(h_grid, dims=["x", "y"], coords={"x": x, "y": y}),
    rh=xr.DataArray([50.0, 60.0], dims=["rh"]),
    href=xr.DataArray(HREF_VALS, dims=["href"]),
    species={"sulphate": 1.0},
)

geo = GeoConfig(
    sza=xr.DataArray(SZA_VALS, dims=["sza"], coords={"sza": SZA_VALS}),
    vza=xr.DataArray(VZA_VALS, dims=["vza"], coords={"vza": VZA_VALS}),
    saa=xr.DataArray([120.0], dims=["saa"]),
    vaa=xr.DataArray([120.0], dims=["vaa"]),
)

spectral_config = SpectralConfig.from_bands([BAND])

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
scene = ImageDict({BAND: xr.Dataset()})

pipeline = RadiativePipeline(
    atmo_config=atmo,
    geo_config=geo,
    spectral_config=spectral_config,
    remove_rayleigh=False,
    n_ph_sph_alb=int(1e6),
    n_ph_rho_atm=int(1e6),
    n_ph_tdif_up=int(1e6),
    n_ph_tdif_down=int(1e6),
    deduplicate_dims=["x", "y"],
)

scene = pipeline(scene)
print("\nResult:")
print(scene[BAND])

# ---------------------------------------------------------------------------
# Plot: 3×3 — rho_atm (sza), tdif_up (vza), sph_alb (href), all at rh=50
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
fig.suptitle("Radiative pipeline — 1000×1000 spatial grid (rh=50, wl=560 nm)")

# Row 0: rho_atm — vary sza, fix vza=10.0, href=2.0
da_rho = scene[BAND]["rho_atm"].sel(rh=50.0, vza=10.0, href=2.0)
for ax, sza in zip(axes[0], SZA_VALS):
    da_rho.sel(sza=float(sza)).plot(ax=ax, cmap="plasma")
    ax.set_title(f"rho_atm  sza={sza}°")
    ax.set_aspect("equal")

# Row 1: tdif_up — vary vza, fix href=2.0
da_tdif = scene[BAND]["tdif_up"].sel(rh=50.0, href=2.0)
for ax, vza in zip(axes[1], VZA_VALS):
    da_tdif.sel(vza=float(vza)).plot(ax=ax, cmap="viridis")
    ax.set_title(f"tdif_up  vza={vza}°")
    ax.set_aspect("equal")

# Row 2: sph_alb — vary href, fix rh=50
da_sph = scene[BAND]["sph_alb"].sel(rh=50.0)
for ax, href in zip(axes[2], HREF_VALS):
    da_sph.sel(href=float(href)).plot(ax=ax, cmap="cividis")
    ax.set_title(f"sph_alb  href={href} km")
    ax.set_aspect("equal")

plt.savefig("radiative_pipeline_spatial.png", dpi=150)
print("\nPlot saved to radiative_pipeline_spatial.png")
plt.show()
