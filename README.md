# Adjeff

<p align="center">
  <img src="https://github.com/walcark/Adjeff/actions/workflows/ci.yml/badge.svg">
  <a href="https://pixi.sh"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://pypi.org/project/adjeff/"><img src="https://img.shields.io/pypi/v/Adjeff.svg"></a>
  <img src="https://img.shields.io/github/license/walcark/Adjeff">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue">
</p>

<p align="center">
  Python library for simulating adjacency effects in satellite imagery (Sentinel-2).<br>
  Forward simulation of TOA reflectance, GPU-accelerated Monte Carlo radiative transfer,<br>
  and PSF learning for atmospheric correction.
</p>

---

## Table of contents

1. [What is an adjacency effect?](#1-what-is-an-adjacency-effect)
2. [Library overview](#2-library-overview)
3. [ImageDict — the central data structure](#3-imagedict--the-central-data-structure)
4. [Image generators](#4-image-generators)
5. [Atmospheric and geometric configuration](#5-atmospheric-and-geometric-configuration)
6. [SceneModule — transforming scenes](#6-scenemodule--transforming-scenes)
7. [SceneSource — creating scenes from scratch](#7-scenesource--creating-scenes-from-scratch)
8. [Pipeline — chaining modules](#8-pipeline--chaining-modules)
9. [SceneModuleSweep — parameter sweeps and deduplication](#9-scenemodule-sweep--parameter-sweeps-and-deduplication)
10. [Smart-G radiative samplers](#10-smart-g-radiative-samplers)
11. [PSF models and PSFDict](#11-psf-models-and-psfdict)
12. [Atmospheric correction (5S model)](#12-atmospheric-correction-5s-model)
13. [PSF optimization](#13-psf-optimization)
14. [xarray accessor and caching](#14-xarray-accessor-and-caching)
15. [Installation](#15-installation)
16. [Roadmap](#16-roadmap)

---

## 1. What is an adjacency effect?

When a satellite observes the Earth, the atmosphere scatters photons laterally: some light from *neighbouring* pixels is redirected towards the sensor and mixes with the signal from the target pixel. This is the **adjacency effect**.

For a surface reflectance map $\rho_s$, the observed top-of-atmosphere (TOA) reflectance $\rho_{toa}$ is:

$$\rho_{toa} = \rho_{atm} + T^\downarrow \frac{T^\uparrow_{dir} \cdot \rho_s + T^\uparrow_{dif} \cdot (\rho_s \ast P)}{1 - s \cdot (\rho_s \ast P)}$$

where $P$ is the **Point Spread Function** (PSF) that encodes the lateral redistribution of energy. Its shape depends on wavelength, aerosol loading, geometry, and ground elevation.

<details>
<summary>Radiative quantity symbols</summary>

| Symbol | Meaning |
|---|---|
| $T_{dir}^\downarrow$ | Direct solar transmittance (Sun → surface) |
| $T_{dir}^\uparrow$ | Direct upward transmittance (surface → sensor) |
| $T_{dif}^\downarrow$ | Diffuse downward transmittance |
| $T_{dif}^\uparrow$ | Diffuse upward transmittance |
| $\rho_{atm}$ | Intrinsic atmospheric reflectance (path radiance) |
| $s$ | Spherical albedo |
| $P$ | Point Spread Function (lateral energy redistribution) |

</details>

The PSF width and shape depend on:
- Aerosol Optical Thickness (`aot`), relative humidity (`rh`), scale height (`href`), species
- Solar and viewing zenith/azimuth angles (`sza`, `vza`, `saa`, `vaa`)
- Wavelength (`wl`) and ground elevation (`h`)

---

## 2. Library overview

`adjeff` has three complementary roles:

| Role | Description |
|---|---|
| **Forward simulation** | Given $\rho_s$ and an atmospheric state, compute $\rho_{toa}$ via Monte Carlo radiative transfer |
| **PSF characterisation** | Determine the shape of $P$ as a function of atmospheric and geometric parameters |
| **Inverse correction** | Recover $\rho_s$ from $\rho_{toa}$ using a learned PSF model |

The design philosophy is to make multi-parameter sweeps first-class: every configuration object accepts `xr.DataArray` inputs with named dimensions, and the library automatically builds the outer product, deduplicates identical parameter combinations, and reconstructs the full-dimensional result — with no extra code from the user.

---

## 3. ImageDict — the central data structure

`ImageDict` is the central container. It is a mapping of `SensorBand` → `xr.Dataset`, one dataset per spectral band. Each band can have a different spatial resolution, so a common array is not always possible.

```python
from adjeff.core import ImageDict, S2Band
import xarray as xr

scene = ImageDict({
    S2Band.B02: xr.Dataset({"rho_s": rho_s_b02}),  # 10 m
    S2Band.B03: xr.Dataset({"rho_s": rho_s_b03}),  # 10 m
    S2Band.B04: xr.Dataset({"rho_s": rho_s_b04}),  # 10 m
    S2Band.B8A: xr.Dataset({"rho_s": rho_s_b8a}),  # 20 m
})

# Access a single band dataset
ds_b02 = scene[S2Band.B02]           # xr.Dataset
rho_s  = scene[S2Band.B02]["rho_s"]  # xr.DataArray
```

As modules are applied, variables accumulate inside each dataset — `rho_toa`, `tdir_down`, etc. — without ever duplicating the spatial arrays. Extra parameter dimensions (e.g. `aot`, `wl`) appear as named xarray dimensions on the result arrays.

`SensorBand` is an abstract base; `S2Band` is the Sentinel-2 implementation (`S2Band.B02` … `S2Band.B12`).

---

## 4. Image generators

Several factory functions create custom scenes for experiments:

```python
from adjeff.core import gaussian_image_dict, disk_image_dict, random_image_dict

bands = [S2Band.B02, S2Band.B03, S2Band.B04]

# Gaussian bright target on a dark background
scene = gaussian_image_dict(sigma=0.5, res_km=0.01, rho_min=0.05, rho_max=0.6,
                            bands=bands, n=101)

# Uniform disk
scene = disk_image_dict(radius=1.0, res_km=0.01, rho_min=0.05, rho_max=0.6,
                        bands=bands, n=101)

# Random spatially heterogeneous scene
scene = random_image_dict(res_km=0.01, bands=bands, n=101)
```

Each generator returns an `ImageDict` with a `rho_s` variable by default per band, ready to pass to any module that declares `required_vars = ["rho_s"]`.

---

## 5. Atmospheric and geometric configuration

All modules that interact with the atmosphere are configured through three Pydantic objects. Each field can be a scalar or an `xr.DataArray` with a named dimension — that dimension will propagate to the output.

```python
import xarray as xr
from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig

atmo = AtmoConfig(
    aot=xr.DataArray([0.05, 0.10, 0.20], dims=["aot"]),  # 3 AOT values
    h=xr.DataArray([0.0], dims=["h"]),
    rh=xr.DataArray([50.0], dims=["rh"]),
    href=xr.DataArray([2.0], dims=["href"]),
    species={"sulphate": 1.0},
)

geo = GeoConfig(
    sza=xr.DataArray([30.0], dims=["sza"]),
    vza=xr.DataArray([10.0], dims=["vza"]),
    saa=xr.DataArray([120.0], dims=["saa"]),
    vaa=xr.DataArray([120.0], dims=["vaa"]),
)

spectral = SpectralConfig.from_bands(bands)
```

<details>
<summary>AtmoConfig fields reference</summary>

| Field | Unit | Description |
|---|---|---|
| `aot` | — | Aerosol optical thickness at 550 nm |
| `h` | km | Ground elevation |
| `rh` | % | Relative humidity |
| `href` | km | Aerosol scale height |
| `species` | dict | Aerosol species mix (weights must sum to 1) |

</details>

<details>
<summary>GeoConfig fields reference</summary>

| Field | Unit | Description |
|---|---|---|
| `vza` | — | Viewing zenith angle [°] |
| `vaa` | km | Viewing azimuth angle [°]  |
| `sza` | % | Solar Zenith angle [°]|
| `saa` | km | solar Azimuth angle [°] |

</details>


---

## 6. SceneModule — transforming scenes

`SceneModule` is the base class for every operation on an `ImageDict`. Each subclass declares:

- `required_vars` — variables that must already exist in the input scene
- `output_vars` — variables it will write into the output scene

```python
class MyModule(SceneModule):
    required_vars = ["rho_s"]
    output_vars   = ["rho_toa"]

    def _compute(self, scene: ImageDict) -> ImageDict:
        ...
        return scene
```

Calling a module enriches the scene without modifying pre-existing variables:

```
ImageDict(B02: [rho_s])  →  MyModule  →  ImageDict(B02: [rho_s, rho_toa])
```

`forward()` handles validation, cache lookup, delegation to `_compute()`, and cache save. Subclasses only implement `_compute()`.

<details>
<summary>Complete module catalogue</summary>

| Class | `required_vars` | `output_vars` | Notes |
|---|---|---|---|
| `SmartgSampler_Tdir_down` | — | `tdir_down` | Direct solar transmittance ↓ |
| `SmartgSampler_Tdir_up` | — | `tdir_up` | Direct transmittance ↑ |
| `SmartgSampler_Tdif_down` | — | `tdif_down` | Diffuse transmittance ↓ |
| `SmartgSampler_Tdif_up` | — | `tdif_up` | Diffuse transmittance ↑ |
| `SmartgSampler_Rho_atm` | — | `rho_atm` | Path reflectance |
| `SmartgSampler_Sph_alb` | — | `sph_alb` | Spherical albedo |
| `SmartgSampler_PSF_Atm` | — | `psf_atm` | Atmospheric PSF kernel |
| `SmartgSampler_Rho_toa_sym` | `rho_s` | `rho_toa` | TOA simulation (GPU) |
| `RadiativePipeline` | — | all radiative quantities | Convenience chain |
| `Toa2Unif` | `rho_toa` + all radiative quantities | `rho_unif` | 5S inversion |
| `Unif2Toa` | `rho_unif` + all radiative quantities | `rho_toa` | 5S forward (no PSF) |
| `Unif2Surface` | `rho_unif`, `sph_alb`, `tdir_up`, `tdif_up` | `rho_s` | 5S + PSF deconvolution |
| `MajaLoader` | — | `rho_s`, `aot`, `rh`, geometry… | Loads MAJA L2A output |

</details>

---

## 7. SceneSource — creating scenes from scratch

`SceneSource` specialises `SceneModule` for modules that produce an `ImageDict` from an external source (disk, satellite product) rather than transforming an existing one. The `required_vars` list is always empty, and calling a `SceneSource` without an input scene is valid.

```python
from adjeff.modules.loaders import MajaLoader

loader = MajaLoader(path="/data/MAJA_L2A/", bands=bands)

scene = loader()        # fresh scene from product
scene = loader(scene)   # or enrich an existing one

print(list(scene[S2Band.B02].data_vars))
# ['rho_s', 'aot', 'rh', 'href', 'vza', 'vaa', 'sza', 'saa', 'h']
```

`ProductLoader` is the abstract base for all product loaders. Mixin classes (`GeometryMixin`, `AtmosphereMixin`, `ElevationMixin`) declare which ancillary variables a loader contributes.

---

## 8. Pipeline — chaining modules

`Pipeline` chains an ordered list of modules and validates at construction that each module's `required_vars` are satisfied by the `output_vars` of the preceding ones.

```python
from adjeff.modules import Pipeline
from adjeff.modules.samplers import (
    SmartgSampler_Tdir_down, SmartgSampler_Tdir_up,
    SmartgSampler_Rho_atm,  SmartgSampler_Sph_alb,
)

pipeline = Pipeline([
    SmartgSampler_Tdir_down(atmo_config=atmo, geo_config=geo,
                            spectral_config=spectral, remove_rayleigh=False),
    SmartgSampler_Tdir_up(atmo_config=atmo, geo_config=geo,
                          spectral_config=spectral, remove_rayleigh=False),
    SmartgSampler_Rho_atm(atmo_config=atmo, geo_config=geo,
                          spectral_config=spectral, remove_rayleigh=False),
    SmartgSampler_Sph_alb(atmo_config=atmo, geo_config=geo,
                          spectral_config=spectral, remove_rayleigh=False),
])

scene = pipeline(scene)
```

If a dependency is missing, a `ValueError` is raised at construction time — not at runtime.

---

## 9. SceneModuleSweep — parameter sweeps and deduplication

`SceneModuleSweep` extends `SceneModule` for computationally intensive modules that must be invoked once per scalar parameter combination. Subclasses declare:

- `scalar_dims` — attributes iterated one value at a time (e.g. `sza`)
- `vector_dims` — attributes passed as a full array in a single call (e.g. `wl`)

The sweep and assembly logic is handled by `ConfigBundle`, which builds the outer product of all scalar dimensions, calls the core function for each combination, and stacks the results into a single xarray output.

<details>
<summary>Spatial deduplication</summary>

When atmospheric parameters vary spatially (e.g. `aot(x, y)` from a MAJA product), a large image may contain only a small number of unique parameter values. The `deduplicate_dims` argument collapses them before the GPU call and reconstructs the full spatial map after:

```python
sampler = SmartgSampler_Tdir_down(
    atmo_config=atmo_spatial,      # aot has dims ["x", "y"]
    geo_config=geo_spatial,
    spectral_config=spectral,
    remove_rayleigh=False,
    deduplicate_dims=["x", "y"],   # 1000×1000 image → N unique pairs
)
scene = sampler(scene)
# tdir_down has dims (wl, x, y) — full spatial map, computed on N points
```

</details>

<details>
<summary>Chunking large vector dimensions</summary>

The `chunks` argument limits how many values are sent to Smart-G in a single call, bounding GPU memory usage:

```python
sampler = SmartgSampler_Tdir_down(
    atmo_config=atmo,
    geo_config=geo,
    spectral_config=spectral,
    remove_rayleigh=False,
    chunks={"wl": 20},
)
```

</details>

---

## 10. Smart-G radiative samplers

All radiative samplers are `SceneModuleSweep` subclasses. They delegate to [Smart-G](https://github.com/hygeos/smartg), a GPU Monte Carlo radiative transfer code, and require CUDA 12.6.

```python
from adjeff.modules.samplers import RadiativePipeline

pipeline = RadiativePipeline(
    atmo_config=atmo,
    geo_config=geo,
    spectral_config=spectral,
    remove_rayleigh=False,
)
scene = pipeline(ImageDict({b: xr.Dataset() for b in bands}))

print(scene[S2Band.B02]["tdir_down"])  # dims: (wl, aot)
print(scene[S2Band.B02]["rho_atm"])    # dims: (wl, aot)
```

`RadiativePipeline` chains the six radiative samplers in the correct order. Use individual sampler classes when only a subset is needed.

### TOA simulation

`SmartgSampler_Rho_toa_sym` combines all radiative quantities with a PSF convolution (under the azimuthal symmetry assumption) to produce $\rho_{toa}$ directly from a surface image:

```python
from adjeff.modules.samplers import SmartgSampler_Rho_toa_sym

module = SmartgSampler_Rho_toa_sym(
    atmo_config=atmo,
    geo_config=geo,
    spectral_config=spectral,
    remove_rayleigh=False,
    nr=80,          # radial PSF samples
    n_ph=int(1e6),  # photons per Smart-G run
)
scene = module(scene)  # requires rho_s
rho_toa = scene[S2Band.B02]["rho_toa"]  # dims: (y, x, aot, wl, ...)
```

### Smart-G auxiliary data

Smart-G requires auxiliary data files that are not bundled with `adjeff`:

```bash
export SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata
```

---

## 11. PSF models and PSFDict

### Analytical PSF models

All analytical models are `torch.nn.Module` subclasses with constrained trainable parameters (positivity, bounded range enforced via `ConstrainedParameter`).

| Class | Shape | Parameters |
|---|---|---|
| `GaussPSF` | Gaussian | `sigma` |
| `GaussGeneralPSF` | Anisotropic Gaussian | `sigma_x`, `sigma_y`, `theta` |
| `VoigtPSF` | Voigt (Gauss + Lorentz) | `sigma`, `gamma` |
| `KingPSF` | King profile | `r_c`, `alpha` |
| `MoffatGeneralizedPSF` | Generalized Moffat | `alpha`, `beta`, `eta` |

```python
from adjeff.core import GaussPSF, PSFGrid

psf  = GaussPSF(sigma=0.3)      # sigma in km
grid = PSFGrid(res_km=0.01, n=101)
kernel = psf(grid)              # xr.DataArray, dims: (y, x)
```

`NonAnalyticalPSF` wraps a fixed numpy kernel (non-trainable) for applying a pre-computed PSF directly.

### PSFDict

`PSFDict` maps `SensorBand` → PSF kernel, in either trainable or frozen mode:

```python
from adjeff.core import PSFDict

# Trainable (for optimization)
psf_dict = PSFDict.from_modules({
    S2Band.B02: GaussPSF(sigma=0.3),
    S2Band.B03: GaussPSF(sigma=0.25),
})

# Frozen (export after training, or from pre-computed kernels)
psf_dict_frozen = psf_dict.to_frozen(grid)
kernel_b02 = psf_dict_frozen[S2Band.B02]  # xr.DataArray
```

A `PSFDict` can carry extra dimensions (e.g. `aot`, `rh`) to represent PSFs that vary with atmospheric state.

---

## 12. Atmospheric correction (5S model)

### Forward model

The 5S formula applied by `Unif2Toa` (no adjacency) and `SmartgSampler_Rho_toa_sym` (with PSF convolution):

$$\rho_{toa} = \rho_{atm} + (T^\uparrow_{dir} + T^\uparrow_{dif}) \cdot (T^\downarrow_{dir} + T^\downarrow_{dif}) \cdot \frac{\rho_{unif}}{1 - s \cdot \rho_{unif}}$$

### Inversion — Toa2Unif

`Toa2Unif` inverts the formula analytically to produce the *equivalent uniform reflectance* $\rho_{unif}$ — the reflectance the pixel would have if the surface were spatially uniform:

```python
from adjeff.modules.classic import Toa2Unif

scene = Toa2Unif()(scene)  # requires rho_toa + all 6 radiative quantities
rho_unif = scene[S2Band.B02]["rho_unif"]
```

### Surface recovery — Unif2Surface

`Unif2Surface` deconvolves $\rho_{unif}$ with a PSF to recover the actual surface $\rho_s$:

```python
from adjeff.modules.models import Unif2Surface

scene = Unif2Surface(psf_dict=psf_dict_frozen)(scene)
rho_s_recovered = scene[S2Band.B02]["rho_s"]
```

---

## 13. PSF optimization

The optimizer learns PSF parameters that best match a set of reference `(rho_s, rho_toa)` image pairs. It runs one independent L-BFGS optimization per atmospheric state combination and assembles the results into a multi-dimensional `PSFDict`.

```python
from adjeff.optim import LBFGSOptimizer, LBFGSConfig, Loss, TrainingImages

train_images = TrainingImages([scene_1, scene_2, scene_3])

optimizer = LBFGSOptimizer(
    train_images=train_images,
    config=LBFGSConfig(
        min_steps=5,
        max_steps=50,
        loss_relative_tolerance=1e-4,
        loss=Loss("MSE_RAD"),
    ),
)

psf_dict = optimizer.run(model)
```

<details>
<summary>Available loss functions</summary>

| Loss | Description |
|---|---|
| `MSE` | Mean squared error |
| `RMSE` | Root mean squared error |
| `MAE` | Mean absolute error |
| `MSE_RAD` | MSE weighted by radial distance |
| `RMSE_RAD` | RMSE weighted by radial distance |
| `MAE_RAD` | MAE weighted by radial distance |

Radial-weighted losses emphasise the wings of the PSF, which carry the adjacency signal.

</details>

---

## 14. xarray accessor and caching

### xarray accessor

All `DataArray` objects produced by `adjeff` can be analysed via the `.adjeff` accessor:

```python
rho_s = scene[S2Band.B02]["rho_s"]

profile = rho_s.adjeff.radial()       # azimuthal mean vs radius
cdf     = rho_s.adjeff.radial_cdf()   # area-weighted CDF
field   = profile.adjeff.to_field(ds) # reconstruct 2D from radial profile
```

### Caching

Every `SceneModule` accepts a `CacheStore` that persists results as Zarr arrays on disk. Cache keys are derived from the module type, its configuration, and a hash of the inputs — rerunning with identical inputs is a no-op.

```python
from adjeff.utils import CacheStore

cache = CacheStore(path="./adjeff_cache")

pipeline = RadiativePipeline(
    atmo_config=atmo, geo_config=geo, spectral_config=spectral,
    remove_rayleigh=False, cache=cache,
)
scene = pipeline(scene)   # computed and cached on first run
scene = pipeline(scene)   # loaded from cache, no GPU call
```

---

## 15. Installation

### Prerequisites

- Python 3.11 or 3.12
- [pixi](https://pixi.sh) ≥ 0.40
- A CUDA 12.6-compatible GPU and driver (for GPU environments)
- Smart-G auxiliary data (see below)

### Clone

```bash
git clone https://github.com/walcark/Adjeff.git
cd Adjeff
```

### Environments

| Environment | GPU | Dev tools | Command |
|---|---|---|---|
| `cpu` | No | No | `pixi install -e cpu` |
| `gpu` | Yes | No | `pixi install -e gpu` |
| `dev` | No | Yes (ruff, mypy, pytest) | `pixi install -e dev` |
| `dev-gpu` | Yes | Yes | `pixi install -e dev-gpu` |
| `notebooks` | No | JupyterLab | `pixi install -e notebooks` |
| `notebooks-gpu` | Yes | JupyterLab | `pixi install -e notebooks-gpu` |

The library is installed in editable mode automatically from `pyproject.toml`.

### Verify

```bash
pixi run -e cpu python -c "import adjeff; print('adjeff OK')"
```

### Smart-G auxiliary data

Smart-G requires auxiliary data files (absorption databases, aerosol models) that are not bundled with `adjeff`. Download them separately and set:

```bash
export SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata
```

Add this to `~/.bashrc` or `~/.zshrc` to make it permanent. The variable must be set before any Smart-G simulation is launched.

### Development workflow

```bash
export SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata

pixi run -e dev fmt          # ruff format
pixi run -e dev lint         # ruff check
pixi run -e dev type-check   # mypy strict
pixi run -e dev test         # pytest (CPU, fast subset)
pixi run -e dev all          # fmt + lint + type-check + test
```

### Notebooks

```bash
# CPU — notebooks 01 and 02
pixi run -e notebooks jupyter lab notebooks/

# GPU — required for notebooks 03 and 04 (Smart-G simulations)
SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata \
pixi run -e notebooks-gpu jupyter lab notebooks/
```

| Notebook | GPU | Content |
|---|---|---|
| `01-create-and-display-image` | No | `ImageDict`, analytical surfaces, radial profiles |
| `02-atmospheric-configuration` | No | `AtmoConfig`, `GeoConfig`, sweeps |
| `03-compute-radiative-quantities` | Yes | `RadiativePipeline`, caching |
| `04-simulate-rho-toa` | Yes | `SmartgSampler_Rho_toa_sym` |

---

## 16. Roadmap

- **Output provenance** — a signing mechanism so every `DataArray` carries a record of the module and parameters that produced it; the `.adjeff` accessor would expose this lineage.
- **Partial cache reuse** — when only a subset of bands or parameter combinations is missing from the cache, recompute only the missing entries rather than the full set.
- **Extended notebook tutorials** — loading MAJA L2A products, building custom `SceneModule` subclasses, end-to-end correction workflow.
