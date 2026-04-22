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
  A Python library for simulating adjacency effects and improving atmospheric correction of satellite imagery (Sentinel-2).
</p>

---

## Table of contents

1. [What is an adjacency effect?](#1-what-is-an-adjacency-effect)
2. [Atmospheric correction with adjeff](#2-atmospheric-correction-with-adjeff)
3. [Computation philosophy](#3-computation-philosophy)
4. [Installation](#4-installation)
5. [Quick start](#5-quick-start)
6. [Core concepts](#6-core-concepts)
7. [Configuring Smart-G](#7-configuring-smart-g)
8. [Running the notebooks](#8-running-the-notebooks)
9. [Roadmap](#9-roadmap)

---

## 1. What is an adjacency effect?

When a satellite sensor observes the Earth, it does not receive light from a single ground pixel alone. The atmosphere scatters photons laterally: some light that was reflected by *neighbouring* pixels is redirected towards the sensor, mixing with the signal from the target pixel. This contamination is called the **adjacency effect**.

For a pixel of surface reflectance $\rho_s(x, y)$, the observed top-of-atmosphere (TOA) reflectance $\rho_{toa}$ can be written as:

$$\rho_{toa}(x, y) = T_{dir\uparrow} \cdot (T_{dir\downarrow} + T_{dif\downarrow}) \cdot \rho_s + \rho_{atm} + \int \rho_s(x', y') \cdot \text{PSF}(x-x', y-y') \, dx' dy'$$

where:

| Symbol | Meaning |
|---|---|
| $T_{dir\downarrow}$ | Direct solar transmittance (Sun → surface) |
| $T_{dir\uparrow}$ | Direct upward transmittance (surface → sensor) |
| $T_{dif\downarrow}$ | Diffuse downward transmittance |
| $T_{dif\uparrow}$ | Diffuse upward transmittance |
| $\rho_{atm}$ | Intrinsic atmospheric reflectance (path radiance) |
| $s$ | Spherical albedo (accounts for multiple surface–atmosphere bounces) |
| PSF | Point spread function encoding the lateral redistribution of energy |

The PSF shape and width depend on the aerosol optical thickness (AOT), the geometry (SZA, VZA), and the wavelength. Correcting adjacency effects therefore requires an accurate simulation of all six radiative quantities above.

---

## 2. Atmospheric correction with adjeff

`adjeff` models the forward problem: given a known surface $\rho_s$ and an atmospheric state, what does the satellite observe?

The library provides:

- **Radiative quantity samplers** — six `SceneModule` subclasses that call [Smart-G](https://github.com/hygeos/smartg) (a GPU Monte Carlo radiative transfer code) to compute $T_{dir\downarrow}$, $T_{dir\uparrow}$, $T_{dif\downarrow}$, $T_{dif\uparrow}$, $\rho_{atm}$, and $s$.
- **TOA simulation** — `SmartgSampler_Rho_toa_sym` combines the radiative quantities with a convolution under the PSF symmetry assumption to produce $\rho_{toa}$.
- **PSF models** — analytical (Gaussian) and non-analytical PSF representations, usable independently for convolution experiments.
- **Parameter sweep engine** — efficient vectorised sweeps over atmospheric and angular parameter grids, with automatic spatial deduplication to avoid recomputing identical configurations.

The intended workflow for **inverse atmospheric correction** (i.e. recovering $\rho_s$ from $\rho_{toa}$) is to build a lookup table or a learned model on top of the forward simulations provided by this library.

---

## 3. Computation philosophy

### Smart-G: always on GPU

All Monte Carlo radiative transfer simulations are delegated to Smart-G, which runs exclusively on GPU via CUDA. There is no CPU fallback for this step. Smart-G handles the full photon transport, including Rayleigh scattering, aerosol scattering, and surface–atmosphere coupling.

The `adjeff` parameter sweep engine (`ConfigBundle` / `SceneModuleSweep`) is designed to maximise GPU occupancy: it deduplicates spatial parameter combinations and batches all unique atmospheric states into a single Smart-G call, avoiding redundant GPU launches.

### Convolutions: CPU or GPU

The convolution step (applying the PSF to a surface image) is implemented via FFT and runs on PyTorch. It can execute on either CPU or GPU depending on the `device` argument passed to `fft_convolve_2D`. For large images, GPU convolution is strongly recommended.

### Environment management: pixi

The project uses [pixi](https://pixi.sh) for reproducible environment management. Two feature sets are available:

| Environment | Command | Notes |
|---|---|---|
| `cpu` | `pixi run --environment cpu` | No GPU, PyTorch CPU only |
| `gpu` | `pixi run --environment gpu` | Requires CUDA 12.6 |
| `dev` | `pixi run --environment dev` | CPU + dev tools (ruff, mypy, pytest) |
| `dev-gpu` | `pixi run --environment dev-gpu` | GPU + dev tools |

---

## 4. Installation

### Prerequisites

- Python 3.11 or 3.12
- [pixi](https://pixi.sh) package manager
- A CUDA 12.6-compatible GPU and driver (for GPU environments)
- Smart-G auxiliary data (see [§7](#7-configuring-smart-g))

### Clone and install

```bash
git clone https://github.com/walcark/Adjeff.git
cd Adjeff
```

Install the CPU-only environment (no GPU required):

```bash
pixi install --environment cpu
```

Or the GPU environment:

```bash
pixi install --environment gpu
```

The library is installed in editable mode automatically via the `pyproject.toml` configuration.

### Verify the installation

```bash
pixi run --environment cpu python -c "import adjeff; print('adjeff OK')"
```

### Development environment

```bash
pixi install --environment dev-gpu   # or dev for CPU

# Run the full quality suite
pixi run --environment dev all       # fmt + lint + type-check + tests
```

---

## 5. Quick start

### Compute all radiative quantities for a single atmospheric state

```python
import xarray as xr
from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import ImageDict, S2Band
from adjeff.modules.samplers import RadiativePipeline

# Define the bands to compute
bands = [S2Band.B02, S2Band.B03, S2Band.B04]

# Atmospheric state
atmo = AtmoConfig(
    aot=xr.DataArray([0.1], dims=["aot"]),       # Aerosol optical thickness
    h=xr.DataArray([0.0], dims=["h"]),            # Ground elevation [km]
    rh=xr.DataArray([50.0], dims=["rh"]),         # Relative humidity [%]
    href=xr.DataArray([2.0], dims=["href"]),       # Aerosol scale height [km]
    species={"sulphate": 1.0},                    # Aerosol species mix
)

# Sun/satellite geometry
geo = GeoConfig(
    sza=xr.DataArray([30.0], dims=["sza"]),       # Sun zenith angle [°]
    vza=xr.DataArray([10.0], dims=["vza"]),       # View zenith angle [°]
    saa=xr.DataArray([120.0], dims=["saa"]),      # Sun azimuth [°]
    vaa=xr.DataArray([120.0], dims=["vaa"]),      # View azimuth [°]
)

# Spectral configuration derived from band definitions
spectral = SpectralConfig.from_bands(bands)

# Empty scene (no surface yet — radiative quantities are atmosphere-only)
scene = ImageDict({band: xr.Dataset() for band in bands})

# Run all six radiative samplers in sequence
pipeline = RadiativePipeline(
    atmo_config=atmo,
    geo_config=geo,
    spectral_config=spectral,
    remove_rayleigh=False,
)
scene = pipeline(scene)

# Access results
print(scene[S2Band.B02]["tdir_down"])   # dims: (wl, aot)
print(scene[S2Band.B02]["rho_atm"])     # dims: (wl, aot)
```

### Simulate TOA reflectance from a surface image

```python
from adjeff.core import gaussian_image_dict
from adjeff.modules.samplers import SmartgSampler_Rho_toa_sym

# Create an analytical Gaussian surface (sigma=0.5 km, 10 m resolution)
scene = gaussian_image_dict(
    sigma=0.5,
    res_km=0.01,
    rho_min=0.05,
    rho_max=0.6,
    bands=bands,
    n=101,
)

# Simulate TOA (requires GPU)
module = SmartgSampler_Rho_toa_sym(
    atmo_config=atmo,
    geo_config=geo,
    spectral_config=spectral,
    remove_rayleigh=False,
    nr=80,
    n_ph=int(1e6),
)
scene = module(scene)

rho_toa = scene[S2Band.B02]["rho_toa"]
print(rho_toa)   # dims: (y, x)
```

### Parameter sweep over AOT and SZA

All config objects accept xarray `DataArray` inputs with named dimensions. The library automatically builds the outer product and batches the computation:

```python
atmo_sweep = AtmoConfig(
    aot=xr.DataArray([0.05, 0.10, 0.20, 0.40], dims=["aot"]),
    h=xr.DataArray([0.0], dims=["h"]),
    rh=xr.DataArray([50.0], dims=["rh"]),
    href=xr.DataArray([2.0], dims=["href"]),
    species={"sulphate": 1.0},
)

geo_sweep = GeoConfig(
    sza=xr.DataArray([20.0, 30.0, 45.0], dims=["sza"]),
    vza=xr.DataArray([0.0], dims=["vza"]),
    saa=xr.DataArray([120.0], dims=["saa"]),
    vaa=xr.DataArray([120.0], dims=["vaa"]),
)

# The output will carry dims (wl, aot, sza)
pipeline_sweep = RadiativePipeline(
    atmo_config=atmo_sweep,
    geo_config=geo_sweep,
    spectral_config=spectral,
    remove_rayleigh=False,
)
scene = pipeline_sweep(ImageDict({band: xr.Dataset() for band in bands}))

tdir_down = scene[S2Band.B02]["tdir_down"]
print(tdir_down.dims)   # ('wl', 'aot', 'sza') or similar
```

---

## 6. Core concepts

### ImageDict

`ImageDict` is the central data structure. It is a dictionary keyed by `SensorBand` instances, each mapping to an `xr.Dataset` that accumulates variables as modules are applied.

```python
from adjeff.core import ImageDict, S2Band
import xarray as xr

scene = ImageDict({
    S2Band.B02: xr.Dataset({"rho_s": rho_s_b02}),
    S2Band.B03: xr.Dataset({"rho_s": rho_s_b03}),
})

# Access a band dataset
ds_b02 = scene[S2Band.B02]
```

Variables accumulate in-place as modules are applied. Each module declares which variables it `required_vars` and which it adds via `output_vars`, enabling compile-time validation of pipeline chains.

### SceneModule and Pipeline

`SceneModule` (a `torch.nn.Module` subclass) is the base class for all operations on an `ImageDict`. Modules are chained into a `Pipeline`, which validates the variable dependency graph at construction time:

```python
from adjeff.modules import Pipeline
from adjeff.modules.samplers import SmartgSampler_Tdir_down, SmartgSampler_Tdir_up

pipeline = Pipeline([
    SmartgSampler_Tdir_down(atmo_config=atmo, geo_config=geo, spectral_config=spectral, remove_rayleigh=False),
    SmartgSampler_Tdir_up(atmo_config=atmo, geo_config=geo, spectral_config=spectral, remove_rayleigh=False),
])
scene = pipeline(scene)
```

### Available radiative samplers

| Class | Output variable | Description |
|---|---|---|
| `SmartgSampler_Tdir_down` | `tdir_down` | Direct solar transmittance (Sun → surface) |
| `SmartgSampler_Tdir_up` | `tdir_up` | Direct upward transmittance (surface → sensor) |
| `SmartgSampler_Tdif_down` | `tdif_down` | Diffuse downward transmittance |
| `SmartgSampler_Tdif_up` | `tdif_up` | Diffuse upward transmittance |
| `SmartgSampler_Rho_atm` | `rho_atm` | Atmospheric path reflectance |
| `SmartgSampler_Sph_alb` | `sph_alb` | Spherical albedo |
| `SmartgSampler_Rho_toa_sym` | `rho_toa` | TOA reflectance (requires `rho_s`) |
| `RadiativePipeline` | all six above | Convenience class chaining all samplers |

### xarray accessor

All DataArrays produced or consumed by adjeff can be accessed via the `.adjeff` accessor:

```python
# Radial analysis
profile = rho_s.adjeff.radial()          # azimuthal mean vs radius
cdf = rho_s.adjeff.radial_cdf()          # area-weighted CDF
adaptive = profile.adjeff.radial_adaptive(n=50, max_gap=0.5)

# Reconstruct a 2D field from a radial profile
field_2d = profile.adjeff.to_field(target_dataset)

# Metadata
print(rho_s.adjeff.kind())     # "analytical" or "arbitrary"
print(rho_s.adjeff.model())    # "gaussian", "disk", or None
print(rho_s.adjeff.res)        # pixel size [km]
```

### Caching

All modules support result caching via `CacheStore`. A shared cache can be passed to a `RadiativePipeline` to avoid recomputing identical inputs across runs:

```python
from adjeff.utils import CacheStore

cache = CacheStore(path="./adjeff_cache")
pipeline = RadiativePipeline(..., cache=cache)
```

Cache keys are built from the module type, its configuration, and a hash of the input arrays, so re-running with the same inputs is a no-op.

### Spatial deduplication

For large images where atmospheric parameters vary spatially (e.g. `aot(x, y)`), the `deduplicate_dims` argument collapses all unique parameter combinations into a compact index before the GPU call, then reconstructs the full spatial result:

```python
# 1000×1000 image with only 20 unique (aot, sza) pairs
# → Smart-G runs on 20 points, not 1,000,000
sampler = SmartgSampler_Tdir_down(
    atmo_config=atmo_spatial,   # aot has dims ["x", "y"]
    geo_config=geo_spatial,     # sza has dims ["x", "y"]
    spectral_config=spectral,
    remove_rayleigh=False,
    deduplicate_dims=["x", "y"],
)
```

---

## 7. Configuring Smart-G

Smart-G requires auxiliary data files (molecular absorption databases, aerosol models, etc.) that are **not bundled** with the package. You must download them separately and point Smart-G to their location via an environment variable.

### Setting SMARTG_DIR_AUXDATA

```bash
export SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata
```

This variable must be set before any Smart-G simulation is run. A good place to put it is your shell configuration file (`~/.bashrc` or `~/.zshrc`) or in a `.env` file loaded at the start of your session.

When using pixi, you can also define it in the `pixi.toml` (not recommended for sensitive paths) or set it in your shell before calling `pixi run`.

### Verifying the setup

```python
import os
from smartg.smartg import Smartg

assert "SMARTG_DIR_AUXDATA" in os.environ, "SMARTG_DIR_AUXDATA is not set"
smartg = Smartg(autoinit=False)   # should not raise
```

### AFGL atmosphere profiles

All radiative samplers accept an `afgl_type` parameter that selects the standard AFGL background atmosphere. The default is `"afgl_exp_h8km"`, which uses an exponential aerosol vertical profile with an 8 km scale height. The available profiles depend on the auxiliary data version installed.

---

## 8. Running the notebooks

The `notebooks/` directory contains step-by-step tutorials. Two dedicated pixi environments bundle JupyterLab, Matplotlib, and ipywidgets alongside the library:

| Environment | Command | Notes |
|---|---|---|
| `notebooks` | `pixi run --environment notebooks jupyter lab` | CPU only, no Smart-G simulations |
| `notebooks-gpu` | `pixi run --environment notebooks-gpu jupyter lab` | Full GPU support, all notebooks |

### Install the notebook environment

```bash
pixi install --environment notebooks       # CPU
# or
pixi install --environment notebooks-gpu   # GPU (requires CUDA 12.6)
```

### Launch JupyterLab

```bash
# CPU — suitable for notebooks 01 and 02
pixi run --environment notebooks jupyter lab notebooks/

# GPU — required for notebooks 03 and 04 (Smart-G simulations)
SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata \
pixi run --environment notebooks-gpu jupyter lab notebooks/
```

JupyterLab opens in your browser at `http://localhost:8888`. The `notebooks/` folder contains the tutorials in suggested reading order:

| Notebook | GPU required | Description |
|---|---|---|
| `01-create-and-display-image.ipynb` | No | `ImageDict`, analytical surfaces, radial profiles |
| `02-atmospheric-configuration.ipynb` | No | `AtmoConfig`, `GeoConfig`, `SpectralConfig`, parameter sweeps |

### Tip: persistent kernel

If you run multiple notebooks in a session and do not want to re-run Smart-G simulations from scratch, pass a shared `CacheStore` at the top of each notebook:

```python
from adjeff.utils import CacheStore
cache = CacheStore(path="./adjeff_cache")
```

Results are keyed on module type + configuration + input hash and are reused automatically on subsequent runs.

---

## 9. Roadmap

The following features are planned for upcoming releases:

### (i) Dask integration and lazy loading

For very large satellite images (tens of thousands × tens of thousands of pixels), loading all bands into memory at once is impractical. The planned approach is to integrate Dask into `ImageDict` so that:
- Arrays are loaded lazily from disk (e.g. from Cloud-Optimized GeoTIFFs or Zarr stores);
- Module operations are expressed as Dask task graphs and executed on demand;
- Memory pressure is bounded regardless of image size.

### (ii) Jupyter notebook documentation

Step-by-step tutorials covering:
- Loading MAJA L2A processor output data and running module on the MAJA output attributes.
- Building custom `SceneModule` subclasses

### (iii) Ensure that each module signs the output

The adjeff accessor relies on the attrs of the xr.DataArray instances. It is therefore important to keep track on all the operations performed on the scene fields. For instance, a sampling module that produces `rho_s` should specify that `rho_s` originates from an estimation process. Another module that loads `rho_s` from a MAJA output folder should specify that it originates from an external source. 

Therefore, some effort should be put into designing specific denominations for signs output from methods. This could be performed by a specific class or from enumations to ensure consistent naming.
