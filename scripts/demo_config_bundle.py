"""Comprehensive demo for ConfigBundle and SceneModuleSweep.

Run with:
    pixi run --environment dev-gpu python scripts/demo_config_bundle.py

Sections
--------
1. Aggregation — scalars from two configs, non-DA attrs in ``other``
2. 1D sweep — aot over 4 values, xarray outer-broadcasts with sza
3. Vector dim — wl passed as full 1D array (SpectralConfig)
4. Joint dedup — aot(x,y) + sza(x,y) on 1000×1000 grid → 20 unique pairs
5. Broadcast dedup — aot(x,y) + sza(x) → sza broadcast before joint unique → 5 unique pairs
6. Vector demotion — aot(x,y) listed as vector → demoted to scalar after dedup
7. Dedup + vector — joint dedup on (x,y) + wl(wl) → reconstruct to (x,y,wl)
8. SpectralConfig — from_bands + find_band bijection
9. SceneModuleSweep — aot × sza sweep + wl vector + reconstruct per band
"""

from __future__ import annotations

import logging
import time

import numpy as np
import structlog
import xarray as xr
from adjeff.utils.logger import MultilineConsoleRenderer

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        MultilineConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    logger_factory=structlog.PrintLoggerFactory(),
)

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig  # noqa: E402
from adjeff.core import S2Band, random_image_dict  # noqa: E402
from adjeff.modules import SceneModuleSweep  # noqa: E402
from adjeff.sweep import SweepBundle  # noqa: E402
from adjeff.utils import ConfigProtocol  # noqa: E402


def sep(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def check(label: str, ok: bool) -> None:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {label}")
    assert ok, f"assertion failed: {label}"


# ===========================================================================
# Spatial fixtures — 1000 × 1000
# ===========================================================================

NX, NY = 1000, 1000
x = np.linspace(-50.0, 50.0, NX)
y = np.linspace(-50.0, 50.0, NY)

#   aot: 5 discrete levels along x  (constant along y)
#   sza: 4 discrete levels along y  (constant along x)
#   → joint unique pairs: 5 × 4 = 20  (out of 1 000 000 pixels)

N_AOT, N_SZA = 5, 4
AOT_LEVELS = np.array([0.05, 0.10, 0.20, 0.30, 0.40])
SZA_LEVELS = np.array([20.0, 30.0, 45.0, 60.0])

xi_zone = np.arange(NX) * N_AOT // NX   # (NX,) ∈ {0,1,2,3,4}
yj_zone = np.arange(NY) * N_SZA // NY   # (NY,) ∈ {0,1,2,3}

aot_map = AOT_LEVELS[xi_zone][:, None] * np.ones((NX, NY))
sza_map = np.ones((NX, NY)) * SZA_LEVELS[yj_zone][None, :]

aot_2d = xr.DataArray(aot_map, dims=["x", "y"], coords={"x": x, "y": y})
sza_2d = xr.DataArray(sza_map, dims=["x", "y"], coords={"x": x, "y": y})

#   For broadcast dedup: sza varies only along x (5 levels, one per x-zone)
#   → after broadcast, all y pixels at the same x have the same (aot, sza)
#   → joint unique = N_AOT = 5  (out of 1 000 000 pixels)
SZA_X_LEVELS = np.array([25.0, 35.0, 45.0, 55.0, 65.0])
sza_x = xr.DataArray(SZA_X_LEVELS[xi_zone], dims=["x"], coords={"x": x})


# ---------------------------------------------------------------------------
# Section 1 — scalars only: aggregation and ``other``
# ---------------------------------------------------------------------------

sep("1 · Aggregation — scalars from two configs, non-DA attrs in other")

atmo1 = AtmoConfig(aot=0.2, h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0})
geo1 = GeoConfig(sza=30.0, vza=10.0, saa=120.0, vaa=0.0)

b1 = SweepBundle.from_configs(
    configs=[atmo1, geo1],
    scalar_names=["aot", "sza", "vza"],
    vector_names=[]
)

print(f"\naot : dims={list(b1.arrays['aot'].dims)}, values={b1.arrays['aot'].values}")
print(f"sza : dims={list(b1.arrays['sza'].dims)}, values={b1.arrays['sza'].values}")
print(f"vza : dims={list(b1.arrays['vza'].dims)}, values={b1.arrays['vza'].values}")
print(f"other keys: {sorted(b1.other.keys())}")

check("aot is 1D with dim 'aot'", list(b1.arrays["aot"].dims) == ["aot"])
check("sza scalar value is 30.0", float(b1.arrays["sza"].squeeze()) == 30.0)
check("species is in other", "species" in b1.other)
check("inverse_map is None (no dedup)", b1.inverse_map is None)
check(
    "aot*10 + sza*0.1 gives 5.0 after squeeze",
    abs(float((b1.arrays["aot"] * 10.0 + b1.arrays["sza"] * 0.1).squeeze()) - 5.0) < 1e-9,
)

# ---------------------------------------------------------------------------
# Section 2 — 1D sweep: aot × sza outer-broadcast
# ---------------------------------------------------------------------------

sep("2 · 1D sweep — aot over 4 values, xarray outer-broadcasts with sza")

atmo2 = AtmoConfig(
    aot=[0.05, 0.10, 0.20, 0.40], h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0}
)
b2 = SweepBundle.from_configs(
    configs=[atmo2, geo1],
    scalar_names=["aot", "sza"],
    vector_names=[]
)

result2 = b2.arrays["aot"] * 10.0 + b2.arrays["sza"] * 0.1
print(f"\naot : {b2.arrays['aot'].values}")
print(f"sza : {b2.arrays['sza'].values}")
print(f"result dims={list(result2.dims)}, shape={result2.shape}")
print(f"result:\n{result2.values}")

check("aot has 4 values", len(b2.arrays["aot"]) == 4)
check("result dims are ['aot', 'sza']", list(result2.dims) == ["aot", "sza"])
check("result shape is (4, 1)", result2.shape == (4, 1))
check(
    "aot=0.2, sza=30 → 5.0",
    abs(float(result2.sel(aot=0.2, sza=30.0)) - 5.0) < 1e-9,
)

# ---------------------------------------------------------------------------
# Section 3 — vector dim: wl as full 1D array (SpectralConfig)
# ---------------------------------------------------------------------------

sep("3 · Vector dim — wl passed as full 1D array (SpectralConfig)")

spectral3 = SpectralConfig.from_bands([S2Band.B02, S2Band.B03, S2Band.B04])
b3 = SweepBundle.from_configs(
    configs=[atmo1, spectral3],
    scalar_names=["aot"],
    vector_names=["wl"]
)

result3 = b3.arrays["aot"] + b3.arrays["wl"] * 0.001
print(f"\nwl  : dims={list(b3.arrays['wl'].dims)}, values={b3.arrays['wl'].values}")
print(f"aot : dims={list(b3.arrays['aot'].dims)}, values={b3.arrays['aot'].values}")
print(f"result dims={list(result3.dims)}, shape={result3.shape}")

check("wl dim is 'wl'", list(b3.arrays["wl"].dims) == ["wl"])
check("wl has 3 values", len(b3.arrays["wl"]) == 3)
check("result dims contain 'aot' and 'wl'", set(result3.dims) == {"aot", "wl"})
check(
    "value at wl=490 is 0.2 + 490*0.001",
    abs(float(result3.sel(wl=490.0, aot=0.2)) - (0.2 + 490 * 0.001)) < 1e-9,
)

# ---------------------------------------------------------------------------
# Section 4 — joint dedup: aot(x,y) + sza(x,y), 1 000 000 pixels → 20 pairs
# ---------------------------------------------------------------------------

sep("4 · Joint dedup — aot(x,y) + sza(x,y), 1 000 000 pixels → 20 unique pairs")

atmo4 = AtmoConfig(aot=aot_2d, h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0})
geo4 = GeoConfig(sza=sza_2d, vza=10.0, saa=120.0, vaa=0.0)

t0 = time.perf_counter()
b4 = ConfigBundle(
    [atmo4, geo4], scalars=["aot", "sza"], vectors=[], deduplicate_dims=["x", "y"]
)
t_bundle = time.perf_counter() - t0

print(f"\nn_original={NX * NY:,}  n_unique={len(b4.arrays['aot'])}  (bundle: {t_bundle:.3f}s)")
print(f"aot unique values: {np.unique(b4.arrays['aot'].values)}")
print(f"sza unique values: {np.unique(b4.arrays['sza'].values)}")
print(f"inverse_map shape: {list(b4.inverse_map.shape)}")

result4 = b4.arrays["aot"] * 10.0 + b4.arrays["sza"] * 0.1  # 20 elements only

t0 = time.perf_counter()
recon4 = b4.reconstruct(result4)
t_recon = time.perf_counter() - t0

expected4 = aot_2d * 10.0 + sza_2d * 0.1
print(f"reconstruct: {t_recon:.3f}s  →  shape={recon4.shape}")

check("n_unique == N_AOT × N_SZA == 20", len(b4.arrays["aot"]) == N_AOT * N_SZA)
check("inverse_map shape is (NX, NY)", list(b4.inverse_map.shape) == [NX, NY])
check("reconstructed dims are ['x', 'y']", list(recon4.dims) == ["x", "y"])
check("reconstruct roundtrip matches direct", np.allclose(recon4.values, expected4.values))

# ---------------------------------------------------------------------------
# Section 5 — broadcast dedup: sza(x) broadcast to (x,y) before joint unique
# ---------------------------------------------------------------------------

sep("5 · Broadcast dedup — aot(x,y) + sza(x), 1 000 000 pixels → 5 unique pairs")

print(f"\nsza input: dims=['x'], shape=({NX},)  →  broadcast to (x,y) before joint unique")
print(f"aot constant along y, sza constant along y after broadcast")
print(f"→ all NY pixels at same x share the same (aot, sza) pair")

atmo5 = AtmoConfig(aot=aot_2d, h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0})
geo5 = GeoConfig(sza=sza_x, vza=10.0, saa=120.0, vaa=0.0)

t0 = time.perf_counter()
b5 = ConfigBundle(
    [atmo5, geo5], scalars=["aot", "sza"], vectors=[], deduplicate_dims=["x", "y"]
)
t_bundle = time.perf_counter() - t0

print(f"\nn_unique={len(b5.arrays['aot'])}  (bundle: {t_bundle:.3f}s)")
print(f"aot unique values: {b5.arrays['aot'].values}")
print(f"sza unique values: {b5.arrays['sza'].values}")

result5 = b5.arrays["aot"] * 10.0 + b5.arrays["sza"] * 0.1
recon5 = b5.reconstruct(result5)

sza_bc = xr.broadcast(sza_x, aot_2d)[0]
expected5 = aot_2d * 10.0 + sza_bc * 0.1

check("n_unique == N_AOT == 5 (all y pixels at same x identical)", len(b5.arrays["aot"]) == N_AOT)
check("reconstructed shape is (NX, NY)", recon5.shape == (NX, NY))
check("reconstruct roundtrip matches direct", np.allclose(recon5.values, expected5.values))

# ---------------------------------------------------------------------------
# Section 6 — vector demotion: aot(x,y) listed as vector → demoted after dedup
# ---------------------------------------------------------------------------

sep("6 · Vector demotion — aot(x,y) listed as vector, demoted to scalar after dedup")

print("\nSince aot(x,y) has dims ⊆ deduplicate_dims, its vector dim gets")
print("absorbed into 'index' → aot is demoted from vector to scalar.")

# aot takes N_AOT unique values (constant along y)
atmo6 = AtmoConfig(aot=aot_2d, h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0})
b6 = SweepBundle.from_configs(
    configs=[atmo6],
    scalar_names=[],
    vector_names=["aot"],, deduplicate_dims=["x", "y"]
)

print(f"\n_vectors before dedup: ['aot']")
print(f"_scalars after  dedup: {b6._scalars}")
print(f"_vectors after  dedup: {b6._vectors}")
print(f"aot dims: {list(b6.arrays['aot'].dims)}, unique values: {b6.arrays['aot'].values}")

check("aot demoted out of vectors", "aot" not in b6._vectors)
check("aot promoted into scalars", "aot" in b6._scalars)
check("aot has dim 'index'", list(b6.arrays["aot"].dims) == ["index"])
check("n_unique == N_AOT == 5", len(b6.arrays["aot"]) == N_AOT)
check(
    "unique values match AOT_LEVELS",
    np.allclose(np.sort(b6.arrays["aot"].values), np.sort(AOT_LEVELS)),
)

# ---------------------------------------------------------------------------
# Section 7 — dedup + independent vector: reconstruct to (x,y,wl)
# ---------------------------------------------------------------------------

sep("7 · Dedup + vector — dedup on (x,y) + wl(wl) → result(index,wl) → (x,y,wl)")

spectral7 = SpectralConfig.from_bands([S2Band.B02, S2Band.B03, S2Band.B04])

t0 = time.perf_counter()
b7 = ConfigBundle(
    [atmo4, geo4, spectral7],
    scalars=["aot", "sza"],
    vectors=["wl"],
    # deduplicate: use UniqueIndex.build() first
    # deduplicate_dims=["x", "y"],
)
t_bundle = time.perf_counter() - t0

print(f"\narrays: {{{', '.join(f'{k}: {list(v.dims)}' for k, v in b7.arrays.items())}}}")

result7 = b7.arrays["aot"] * 10.0 + b7.arrays["sza"] * 0.1 + b7.arrays["wl"] * 0.001
print(f"compute on {result7.size} elements (vs {NX * NY * len(spectral7.wl):,} direct)")
print(f"result dims={list(result7.dims)}, shape={result7.shape}")

t0 = time.perf_counter()
recon7 = b7.reconstruct(result7)
t_recon = time.perf_counter() - t0

expected7 = (aot_2d * 10.0 + sza_2d * 0.1 + spectral7.wl * 0.001).transpose("x", "y", "wl")
print(f"reconstruct: {t_recon:.3f}s  →  shape={recon7.shape}")

check("result dims before reconstruct: {'index','wl'}", set(result7.dims) == {"index", "wl"})
check("result size == n_unique × n_wl == 60", result7.size == N_AOT * N_SZA * 3)
check("reconstructed dims contain x, y, wl", set(recon7.dims) == {"x", "y", "wl"})
check("reconstructed shape is (NX, NY, 3)", recon7.shape == (NX, NY, 3))
check(
    "reconstruct roundtrip matches direct",
    np.allclose(recon7.transpose("x", "y", "wl").values, expected7.values),
)

# ---------------------------------------------------------------------------
# Section 8 — SceneModuleSweep: aot × sza + wl vector + reconstruct per band
# ---------------------------------------------------------------------------

sep("8 · SceneModuleSweep — aot(3) × sza(2) sweep + wl vector, verify per band")


class FakeTdir(SceneModuleSweep):
    """Fake direct transmittance: exp(-aot/cos(sza)) * (1 - wl*1e-4)."""

    required_vars = []
    output_vars = ["tdir"]
    scalar_dims = ["aot", "sza"]
    vector_dims = ["wl"]

    def __init__(
        self,
        atmo: AtmoConfig,
        geo: GeoConfig,
        spectral: SpectralConfig,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._atmo = atmo
        self._geo = geo
        self._spectral = spectral

    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        return (self._atmo, self._geo, self._spectral)

    def _compute(self, scene):
        bundle = self._make_bundle()
        arrays = bundle.arrays

        tdir = np.exp(
            -arrays["aot"] / np.cos(np.radians(arrays["sza"]))
        ) * (1.0 - arrays["wl"] * 1e-4)

        tdir = bundle.reconstruct(tdir)   # → (aot, sza, wl)
        
        for band in self._spectral.bands:
            scene[band]["tdir"] = tdir.sel(wl=band.wl_nm).drop_vars("wl")

        return scene


atmo9 = AtmoConfig(
    aot=[0.1, 0.2, 0.3], h=2.0, rh=50.0, href=2.0, species={"aerosol": 1.0}
)
geo9 = GeoConfig(sza=[30.0, 45.0], vza=10.0, saa=120.0, vaa=0.0)
spectral9 = SpectralConfig.from_bands([S2Band.B02, S2Band.B03, S2Band.B04])

scene9 = random_image_dict(
    bands=[S2Band.B02, S2Band.B03, S2Band.B04],
    res_km=0.01,
    variables=["rho_s"],
    n=1,
    seed=0,
)

module = FakeTdir(atmo9, geo9, spectral9)
result_scene = module(scene9)

for band, wl_nm in [
    (S2Band.B02, 490.0),
    (S2Band.B03, 560.0),
    (S2Band.B04, 665.0),
]:
    tdir = result_scene[band]["tdir"]
    print(f"\n{band.name}: dims={list(tdir.dims)}, shape={list(tdir.shape)}")
    check(f"{band.name} dims are ['aot', 'sza']", list(tdir.dims) == ["aot", "sza"])
    check(f"{band.name} shape is (3, 2)", list(tdir.shape) == [3, 2])

    for aot_val in [0.1, 0.2, 0.3]:
        for sza_val in [30.0, 45.0]:
            val = float(tdir.sel(aot=aot_val, sza=sza_val))
            exp_val = float(
                np.exp(-aot_val / np.cos(np.radians(sza_val))) * (1.0 - wl_nm * 1e-4)
            )
            check(
                f"  tdir(aot={aot_val}, sza={sza_val}) = {exp_val:.6f}",
                abs(val - exp_val) < 1e-9,
            )

print("\n\nAll assertions passed.")
