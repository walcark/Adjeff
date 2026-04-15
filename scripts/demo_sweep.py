"""Interactive demo for DimSweeper and SceneModuleSweep.

Run with:
    pixi run --environment dev-gpu python scripts/demo_sweep.py

Each section is independent and clearly labelled.  structlog is configured
at DEBUG level so every internal step is visible.
"""

from __future__ import annotations

import logging

import numpy as np
import structlog
import xarray as xr
from adjeff.utils.logger import MultilineConsoleRenderer

# ---------------------------------------------------------------------------
# structlog — DEBUG output so every internal step is visible
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        MultilineConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    logger_factory=structlog.PrintLoggerFactory(),
)

# ---------------------------------------------------------------------------
# Imports (after logging config so structlog is ready)
# ---------------------------------------------------------------------------

from adjeff.atmosphere import AtmoConfig, GeoConfig  # noqa: E402
from adjeff.core import S2Band, random_image_dict  # noqa: E402
from adjeff.modules import SceneModuleSweep  # noqa: E402
from adjeff.utils.sweep import DimSweeper  # noqa: E402


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Section 1 — scalar config, no sweep
# ---------------------------------------------------------------------------

separator("1 · Scalar config — all parameters are single values")

atmo = AtmoConfig(aot=0.2, h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0})
geo = GeoConfig(sza=30.0, vza=10.0, saa=120.0, vaa=0.0)

sweeper = DimSweeper(
    scalar_dims=["aot", "sza", "vza", "saa", "vaa"],
    vector_dims=["h", "rh"],
)

dim_arrays, inverse_map = sweeper.collect(atmo, geo)


def fake_core_no_spatial(h, rh, aot, sza, vza, saa, vaa):
    # h and rh are 1D numpy arrays (vector dims) — use .mean() to get a scalar
    # aot, sza, etc. are 0-dim numpy scalars (swept dims)
    return np.float32(float(aot) * 10.0 + float(rh.mean()) * 0.01)


result = sweeper.apply(fake_core_no_spatial, [], dim_arrays, inverse_map)
print(f"\nResult dims  : {list(result.dims)}")
print(f"Result shape : {list(result.shape)}")
print(f"Result value : {float(result.squeeze()):.4f}  (expected {0.2*10 + 50*0.01:.4f}  = aot*10 + rh.mean()*0.01)")

# ---------------------------------------------------------------------------
# Section 2 — 1D sweep over aot
# ---------------------------------------------------------------------------

separator("2 · 1D sweep — aot varies over 4 values")

atmo2 = AtmoConfig(
    aot=[0.05, 0.10, 0.20, 0.40], h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0}
)

dim_arrays2, _ = sweeper.collect(atmo2, geo)
result2 = sweeper.apply(fake_core_no_spatial, [], dim_arrays2, None)

print(f"\nResult dims  : {list(result2.dims)}")
print(f"Result shape : {list(result2.shape)}")
print(f"aot values   : {dim_arrays2['aot'].values}")
print(f"Result along aot: {result2.isel(sza=0, vza=0, saa=0, vaa=0).values.ravel()}")

# ---------------------------------------------------------------------------
# Section 3 — vector dims passed as arrays
# ---------------------------------------------------------------------------

separator("3 · Vector dims — h and rh passed as full arrays to _core_band")

atmo3 = AtmoConfig(
    aot=0.2,
    h=[0.5, 1.0, 2.0, 3.0],
    rh=[30.0, 50.0, 70.0],
    href=2.0,
    species={"aerosol": 1.0},
)

dim_arrays3, _ = sweeper.collect(atmo3, geo)
received: dict = {}


def fake_core_check_shapes(h, rh, aot, sza, vza, saa, vaa):
    # h  → np.ndarray of shape (4,)  — full vector dim
    # rh → np.ndarray of shape (3,)  — full vector dim
    # aot, sza, ... → 0-dim numpy scalars
    received["h_shape"] = h.shape
    received["rh_shape"] = rh.shape
    received["aot_type"] = type(aot).__name__
    return np.float32(float(h.mean()) + float(rh.mean()) * 0.01)


result3 = sweeper.apply(fake_core_check_shapes, [], dim_arrays3, None)
print(f"\nh  received as : {received['h_shape']}  (full array — vector dim)")
print(f"rh received as : {received['rh_shape']}  (full array — vector dim)")
print(f"aot received as: {received['aot_type']}  (scalar — swept dim)")
print(f"Result dims    : {list(result3.dims)}")

# ---------------------------------------------------------------------------
# Section 4 — 2D spatial config with deduplication
# ---------------------------------------------------------------------------

separator("4 · Deduplication — aot(x,y) and rh(x,y) collapsed to unique index")

x = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 1.0])

# 6 pixels, only 3 unique (aot, rh) pairs
aot_2d = xr.DataArray(
    [[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]],
    dims=["x", "y"],
    coords={"x": x, "y": y},
)
rh_2d = xr.DataArray(
    [[50.0, 60.0], [50.0, 70.0], [60.0, 70.0]],
    dims=["x", "y"],
    coords={"x": x, "y": y},
)

atmo4 = AtmoConfig(aot=aot_2d, h=2.0, rh=rh_2d, href=2.0, species={"aerosol": 1.0})

sweeper_dedup = DimSweeper(
    scalar_dims=["aot", "rh", "sza", "vza", "saa", "vaa"],
    vector_dims=["h"],
    deduplicate_dims=["x", "y"],
)

dim_arrays4, inverse_map4 = sweeper_dedup.collect(atmo4, geo)

print(f"\naot after dedup : {dim_arrays4['aot'].values}  (3 unique values from 6 pixels)")
print(f"rh  after dedup : {dim_arrays4['rh'].values}")
print(f"inverse_map     : shape {list(inverse_map4.shape)}, dims {list(inverse_map4.dims)}")
print(f"inverse_map values:\n{inverse_map4.values}")


def fake_core_dedup(h, aot, rh, sza, vza, saa, vaa):
    # h is a 1D array (vector dim); aot, rh, sza, ... are 0-dim scalars
    return np.float32(float(aot) * 10.0 + float(rh) * 0.1)


result4 = sweeper_dedup.apply(fake_core_dedup, [], dim_arrays4, inverse_map4)
print(f"\nResult dims  : {list(result4.dims)}")
print(f"Result shape : {list(result4.shape)}")
print("\nResult values (x, y) — squeezed over scalar geo dims:")
print(result4.squeeze().values)
print("\nExpected values:")
print((aot_2d * 10.0 + rh_2d * 0.1).values)

# ---------------------------------------------------------------------------
# Section 5 — 2D param without dedup should raise
# ---------------------------------------------------------------------------

separator("5 · Error case — 2D param without deduplicate_dims")

sweeper_no_dedup = DimSweeper(
    scalar_dims=["aot", "sza", "vza", "saa", "vaa"],
    vector_dims=["h"],
)
try:
    sweeper_no_dedup.collect(atmo4, geo)
    print("ERROR: should have raised ValueError")
except ValueError as e:
    print(f"\nValueError correctly raised:\n  {e}")

# ---------------------------------------------------------------------------
# Section 6 — SceneModuleSweep with spatial input (rho_s)
# ---------------------------------------------------------------------------

separator("6 · SceneModuleSweep — sweep over aot with spatial rho_s input")


class FakeRadiativeModule(SceneModuleSweep):
    """Fake Smart-G module: rho_atm = rho_s + aot * 10."""

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


atmo5 = AtmoConfig(
    aot=[0.1, 0.2, 0.3], h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0}
)
scene = random_image_dict(bands=[S2Band.B02], variables=["rho_s"], n=4, seed=0)

module = FakeRadiativeModule(atmo5, geo)
result_scene = module(scene)

rho_atm = result_scene[S2Band.B02]["rho_atm"]
print(f"\nrho_atm dims  : {list(rho_atm.dims)}")
print(f"rho_atm shape : {list(rho_atm.shape)}")
print(f"aot dim size  : {rho_atm.sizes['aot']}  (one slice per aot value)")
print("\nrho_atm[aot index=1, i.e. aot=0.2] (first 2x2):")
print(rho_atm.isel(aot=1).squeeze().values[:2, :2])
print("\nrho_s (first 2x2) + 0.2*10 (expected):")
print(scene[S2Band.B02]["rho_s"].values[:2, :2] + 2.0)

# ---------------------------------------------------------------------------
# Section 7 — chunks (requires dask)
# ---------------------------------------------------------------------------

separator("7 · Chunks — h split into chunks of 2 before each GPU call")

try:
    import dask  # noqa: F401

    atmo7 = AtmoConfig(
        aot=0.2,
        h=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],  # 6 values → 3 chunks of 2
        rh=50.0,
        href=2.0,
        species={"aerosol": 1.0},
    )

    sweeper7 = DimSweeper(
        scalar_dims=["aot", "sza", "vza", "saa", "vaa"],
        vector_dims=["h", "rh"],
        chunks={"h": 2},
    )

    dim_arrays7, inverse_map7 = sweeper7.collect(atmo7, geo)

    call_log: list[tuple] = []

    def fake_core_chunked(h, rh, aot, sza, vza, saa, vaa):
        call_log.append(h.copy())
        return (h + float(aot) * 10.0).astype(np.float32)

    result7 = sweeper7.apply(fake_core_chunked, [], dim_arrays7, inverse_map7)

    print(f"\nh values      : {dim_arrays7['h'].values}  (6 values, chunks of 2)")
    print(f"Number of _core_band calls : {len(call_log)}  (expected 3 chunks)")
    for i, h_chunk in enumerate(call_log):
        print(f"  chunk {i}: h = {h_chunk}")
    print(f"\nResult dims   : {list(result7.dims)}")
    print(f"Result shape  : {list(result7.shape)}")

except ImportError:
    print("\ndask not installed in this environment — skipping chunks demo.")
    print("Install dask to enable GPU memory chunking on vector dims.")
