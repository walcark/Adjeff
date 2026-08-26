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
9. [SweepSampler, parameter sweeps and deduplication](#9-sweepsampler--parameter-sweeps-and-deduplication)
10. [Smart-G radiative samplers](#10-smart-g-radiative-samplers)
11. [PSF models and the PSF tree](#11-psf-models-and-the-psf-tree)
12. [Atmospheric correction (5S model)](#12-atmospheric-correction-5s-model)
13. [PSF optimization](#13-psf-optimization)
14. [High-level API](#14-high-level-api)
15. [xarray accessor and caching](#15-xarray-accessor-and-caching)
16. [Installation](#16-installation)
17. [Roadmap](#17-roadmap)

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
| `vza` | ° | Viewing zenith angle |
| `vaa` | ° | Viewing azimuth angle |
| `sza` | ° | Solar zenith angle |
| `saa` | ° | Solar azimuth angle |

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
| `TdirDownSampler` | — | `tdir_down` | Direct solar transmittance ↓ |
| `TdirUpSampler` | — | `tdir_up` | Direct transmittance ↑ |
| `TdifDownSampler` | — | `tdif_down` | Diffuse transmittance ↓ |
| `TdifUpSampler` | — | `tdif_up` | Diffuse transmittance ↑ |
| `RhoAtmSampler` | — | `rho_atm` | Path reflectance |
| `SphAlbSampler` | — | `sph_alb` | Spherical albedo |
| `WuPsfSampler` | — | `psf_atm` | Sampled atmospheric PSF, Wu et al. 2024, in `adjeff.reference` |
| `RhoToaSampler` | `rho_s` | `rho_toa` | TOA simulation (GPU) |
| `RhoToaSymSampler` | `rho_s` | `rho_toa` | TOA simulation, azimuthal symmetry (GPU) |
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
from pathlib import Path
from adjeff.modules.loaders import MajaLoader

loader = MajaLoader(
    product_path=Path("/data/MAJA_L2A/"),
    bands=bands,
    res=0.12,
    mnt_path=Path("/data/mnt/"),
)

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
    TdirDownSampler, TdirUpSampler,
    RhoAtmSampler,   SphAlbSampler,
)

pipeline = Pipeline([
    TdirDownSampler(atmo_config=atmo, geo_config=geo,
                    spectral_config=spectral, remove_rayleigh=False),
    TdirUpSampler(atmo_config=atmo, geo_config=geo,
                  spectral_config=spectral, remove_rayleigh=False),
    RhoAtmSampler(atmo_config=atmo, geo_config=geo,
                  spectral_config=spectral, remove_rayleigh=False),
    SphAlbSampler(atmo_config=atmo, geo_config=geo,
                  spectral_config=spectral, remove_rayleigh=False),
])

scene = pipeline(scene)
```

If a dependency is missing, a `ConfigurationError` is raised at construction time — not at runtime.

---

## 9. SweepSampler — parameter sweeps and deduplication

A sampler that must call Smart-G once per parameter combination declares a
[xsweep](https://github.com/walcark/xsweep) contract instead of writing the
loop:

```python
class TdirDownSampler(SweepSampler):
    contract = "batch(aot, rh, h, href, sza) vec(wl) -> tdir_down(wl)"
    point_fn = staticmethod(tdir_down)
```

The contract names three roles. `loop` variables are handed one value per
call, `batch` variables are grouped so that one call carries several
states, and `vec` variables arrive whole. Smart-G amortises the
atmospheric profile over a batch, which is why the six radiative samplers
declare `batch` rather than `loop`: calling once per state costs three
times more.

The three samplers that rebuild the sensor grid per geometry cannot batch
those axes and declare `loop(sza, vza)` instead.

<details>
<summary>Spatial deduplication</summary>

When atmospheric parameters vary spatially, for instance `aot(x, y)` read
from a MAJA product, a large image usually holds few distinct values.
`dedup=True` collapses them before the GPU call and restores the full map
afterwards:

```python
sampler = TdirDownSampler(
    atmo_config=atmo_spatial,      # aot has dims ["x", "y"]
    geo_config=geo_spatial,
    spectral_config=spectral,
    remove_rayleigh=False,
    dedup=True,                    # 1000x1000 image -> N distinct states
)
scene = sampler(scene)
# tdir_down has dims (wl, x, y): the full map, computed on N points
```

It is a flag rather than a list of dimensions: xsweep collapses repeated
states wherever they occur, and guarantees the result is unchanged.

</details>

<details>
<summary>Bounding the GPU memory of one call</summary>

`batch_size` caps how many states travel in a single Smart-G call:

```python
sampler = TdirDownSampler(..., batch_size=32)
```

</details>

<details>
<summary>Choosing where a result is written</summary>

A module declares the *roles* it reads and writes; an instance binds them
to names in the Dataset. This is what lets two estimates of the same
quantity sit beside the truth they estimate:

```python
king = Unif2Surface(kernels=tree_king, rename={"rho_s": "rho_s_king"})
gauss = Unif2Surface(kernels=tree_gauss, rename={"rho_s": "rho_s_gauss"})
scene = gauss(king(scene))
```

The cache is keyed by role, so both instances share one entry: where a
result is written changes nothing to what is computed.

</details>

---

## 10. Smart-G radiative samplers

All radiative samplers are `SweepSampler` subclasses. They delegate to [Smart-G](https://github.com/hygeos/smartg), a GPU Monte Carlo radiative transfer code, and require CUDA 12.6.

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

`RhoToaSymSampler` combines all radiative quantities with a PSF convolution (under the azimuthal symmetry assumption) to produce $\rho_{toa}$ directly from a surface image:

```python
from adjeff.modules.samplers import RhoToaSymSampler

module = RhoToaSymSampler(
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

## 11. PSF models and the PSF tree

### Analytical PSF models

All analytical models are `torch.nn.Module` subclasses with constrained trainable parameters (positivity, bounded range enforced via `ConstrainedParameter`).

| Class | Shape | Parameters |
|---|---|---|
| `GaussPSF` | Gaussian | `sigma` |
| `GeneralizedGaussianPSF` | Generalised Gaussian exp(-(r/σ)ⁿ) | `sigma`, `n` |
| `VoigtPSF` | Pseudo-Voigt (Gauss + Lorentz) | `sigma`, `gamma` |
| `KingPSF` | King profile | `sigma`, `gamma` |
| `MoffatGeneralizedPSF` | Generalised Moffat | `alpha`, `beta`, `gamma` |

```python
from adjeff.core import GaussPSF, PSFGrid, S2Band

grid   = PSFGrid(res=0.01, n=101)                         # 101×101 grid, 10 m pixels
psf    = GaussPSF(grid=grid, band=S2Band.B02, sigma=0.3)  # sigma in km
kernel = psf.forward()       # torch.Tensor, shape (101, 101)
da     = psf.to_dataarray()  # xr.DataArray, dims (y_psf, x_psf)
```

`NonAnalyticalPSF` wraps a fixed numpy kernel (non-trainable) for applying a pre-computed PSF directly.

### The PSF tree

Frozen PSFs live in an `xarray.DataTree`, one group per band. Bands keep
their own grid, so a 101x101 kernel at 10 m and a 51x51 one at 20 m coexist
in the same tree, and the whole tree round-trips through zarr:

```python
from adjeff.core import (
    GaussPSF, PSFGrid, S2Band, freeze, psf_kernel, psf_params,
)

modules = {
    b: GaussPSF(PSFGrid(res=0.01, n=101), b, sigma=0.3)
    for b in (S2Band.B02, S2Band.B03)
}
tree = freeze(modules)                  # xr.DataTree, one group per band

kernel = psf_kernel(tree, S2Band.B02)   # xr.DataArray, dims (y_psf, x_psf)
params = psf_params(tree, S2Band.B02)   # {"sigma": xr.DataArray}

tree.to_zarr("psf.zarr", mode="w")
```

A tree produced by `fit` carries the sweep dimensions of the training set
(e.g. `aot`, `rh`) on both the kernel and each fitted parameter.

---

## 12. Atmospheric correction (5S model)

### Forward model

The 5S formula applied by `Unif2Toa` (no adjacency) and `RhoToaSymSampler` (with PSF convolution):

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

scene = Unif2Surface(kernels=tree)(scene)
rho_s_recovered = scene[S2Band.B02]["rho_s"]
```

---

## 13. PSF optimization

`fit` learns the PSF parameters that best match a set of reference `(rho_s, rho_toa)` image pairs. It runs one independent optimisation per `(atmospheric state, band)` pair and assembles the results into a frozen PSF tree.

```python
from adjeff.optim import Loss, Metric, TrainingImages, fit

train_images = TrainingImages(
    images=[scene_1, scene_2, scene_3],
    weights=[1.0, 1.0, 1.0],
)

# Default stages: Adam warm-up, then L-BFGS refinement.
tree = fit(model, train_images, loss=Loss(Metric.RMSE_RAD))
```

Pass `stages=` to control the schedule, and `store=` to stream each band's
kernels to zarr instead of keeping them in RAM:

```python
from adjeff.optim import AdamConfig, LBFGSConfig

tree = fit(
    model,
    train_images,
    stages=[
        AdamConfig(min_steps=5, max_steps=20, loss=Loss(Metric.MSE_RAD)),
        LBFGSConfig(min_steps=5, max_steps=50, loss=Loss(Metric.MSE_RAD)),
    ],
    store="psf.zarr",
)
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

## 14. High-level API

`adjeff.api` provides convenience functions that combine multiple building blocks into single calls:

```python
from adjeff.api import load_maja, load_config, fit_psf, apply_psf

# 1. Load a MAJA product (species persisted in attrs)
scene = load_maja(product_path=Path("..."), bands=bands, res=0.12, mnt_path=Path("..."))

# 2. Build FullConfig from the scene (with optional spatial aggregation)
cfg = load_config(scene, band=S2Band.B03, aggregate=True)

# 3. Fit a PSF end-to-end (radiatives + training scenes + optimizer)
tree = fit_psf(
    scene, bands=[S2Band.B03],
    psf_type=KingPSF,
    init_parameters={"sigma": 0.5, "gamma": 2.0},
)

# 4. Apply the frozen PSF to recover rho_s
scene_corrected = apply_psf(scene, tree, band=S2Band.B03)
```

| Function | Description |
|---|---|
| `load_scene(loader)` | Generic loader with species persistence |
| `load_maja(...)` | `load_scene` pre-wired for `MajaLoader` |
| `load_config(scene, band)` | `FullConfig` from a scene (with aggregation / digitisation) |
| `make_full_config(bands, ...)` | `FullConfig` from raw scalars |
| `run_radiatives_from_scene(scene)` | The six radiative quantities, config read from the scene |
| `run_forward_pipeline(scene, **cfg)` | Radiatives → rho_toa → rho_unif |
| `make_model(model_cls, psf_type, ...)` | One live PSF module per band, wrapped in a model |
| `fit_psf(scene, bands, psf_type, ...)` | End-to-end PSF fitting in one call |
| `apply_psf(scene, psf_tree, band)` | Apply a frozen PSF tree to a scene |
| `sample_psf_atm(bands, ...)` | Atmospheric PSF from explicit config |
| `sample_psf_atm_from_scene(scene, band)` | Atmospheric PSF inferred from a scene |

---

## 15. xarray accessor and caching

### xarray accessor

All `DataArray` objects produced by `adjeff` can be analysed via the `.adjeff` accessor:

```python
rho_s = scene[S2Band.B02]["rho_s"]

profile = rho_s.adjeff.radial()               # azimuthal mean vs radius
cdf     = rho_s.adjeff.radial("cdf")          # area-weighted CDF
both    = rho_s.adjeff.radial(symmetric=True) # mirrored, for a full transect
field   = profile.adjeff.to_field(ds)         # reconstruct 2D from a profile
```

A scalar in a configuration is stored as an array of length one, so an
output carries an `aot`, `rh`, `h` and `href` dimension even when a single
atmospheric state was simulated. `tidy` turns those into scalar
coordinates: the dimensions go, the values stay.

```python
rho_toa.dims                       # ('sza', 'vza', 'aot', 'rh', 'href', 'h', 'y', 'x')
tidied = rho_toa.adjeff.tidy()
tidied.dims                        # ('y', 'x')
float(tidied.aot)                  # 0.4, the state that produced it

tidied.adjeff.untidy()             # back to dimensions, before a merge
```

A tidied array recombines with `xr.concat`, which promotes the coordinate
back to a dimension. It does not recombine with `xr.merge` or
`xr.combine_by_coords`, which align on dimensions: call `untidy` first.

### Analysing a PSF

The accessor spells what `adjeff.analysis` implements, and that package
holds what quantifies a kernel:

```python
from adjeff.analysis import encircled_radius, fwhm, mtf, rmse

encircled_radius(kernel, 0.5)      # radius holding half the energy [km]
fwhm(kernel)                       # full width at half maximum [km]
mtf(kernel)                        # contrast against spatial frequency

rmse(estimated, truth, mask=15.0, radial=True)   # over a 15 km disc
```

One rule decides what belongs there: does it quantify a PSF, or a 2-D
field of adjacency effect? A generic 1-D FFT, a power spectral density, a
Wiener filter do not.

### Caching

Every `SceneModule` accepts a `CacheStore` that persists results as Zarr arrays on disk. Cache keys are derived from the module type, its configuration, and a hash of the inputs — rerunning with identical inputs is a no-op.

```python
from adjeff.utils import CacheStore

cache = CacheStore(cache_dir="./adjeff_cache")

pipeline = RadiativePipeline(
    atmo_config=atmo, geo_config=geo, spectral_config=spectral,
    remove_rayleigh=False, cache=cache,
)
scene = pipeline(scene)   # computed and cached on first run
scene = pipeline(scene)   # loaded from cache, no GPU call
```

---

## 16. Installation

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
| `04-simulate-rho-toa` | Yes | `RhoToaSymSampler` |
| `05-compute-2d-radiative-from-maja-output` | Yes | `load_maja`, `run_radiatives_from_scene`, spatial maps |
| `06-learn-psf` | Yes | `fit`, `sample_psf_atm`, PSF learning end-to-end |

---

## 17. Roadmap

- **Integrating the PSF over the central pixels, rather than sampling it there.** A tap of a discrete convolution is the integral of the profile over one pixel: writing the continuous convolution over an image that is constant per pixel gives `rho_env(x_i) = sum_j rho_unif(x_j) * integral over cell j of P(x_i - x')`. Sampling the profile at the pixel's centre stands in for that integral, and stands in well from the first neighbour outwards, within 0.4 % for a generalised Gaussian at `n = 0.2`. It does not at the centre pixel, where a kernel sharp against the grid varies by orders of magnitude across one cell: measured against the true cell average, the centre tap is 8 % too high for a Gaussian of `sigma = 0.33 km` on a 0.1 km grid and 550 % too high for a King profile of `sigma = 0.01 km` on the same one. The criterion is not the family of the kernel but how much of its energy falls inside one pixel.

  Sub-sampling a 33x33 patch at 16x16 and the centre pixel alone at 256x256 puts every tap within 0.5 % of its cell average even in the sharpest regime, and measures at under 2 % of a training step. It is nonetheless **not** done, for a reason worth keeping in view: the fitted parameters absorb the bias, so a kernel truer to the continuous physics does not automatically fit this discrete problem better. What would settle it is a measurement rather than an argument — fit at two resolutions with and without, then compare the loss reached and how far `(sigma, n)` move between grids. The details and the numbers are in `PSFModule`'s docstring.

  The gradient, which is a separate matter, is fixed: see `radial_power`.

- **Provenance of the parameters, not only of the module.** Every output `DataArray` already carries `_adjeff_provenance` with the module that produced it and its cache key, which is what lets a downstream hash address it. What it does not carry is the parameters that module ran with, and the `.adjeff` accessor exposes none of it. Reading a cached result back therefore says where it came from but not under what atmosphere.

- **Partial cache reuse.** A cache entry is addressed by one key covering the whole call, and `load_vars` returns nothing unless every band and every variable is present. Adding one band or one atmospheric combination changes the key and recomputes the rest along with it.

- **A notebook on writing a `SceneModule`.** The six notebooks cover creating scenes, configuring an atmosphere, computing radiative quantities, simulating `rho_toa`, reading a MAJA L2A product and learning a PSF. Extending the package with a module of one's own is documented in section 6 but nowhere worked through end to end.
