# Adjeff user guide

1. [The adjacency effect](#1-the-adjacency-effect)
2. [Scenes](#2-scenes)
3. [Atmosphere and geometry](#3-atmosphere-and-geometry)
4. [Scene modules and pipelines](#4-scene-modules-and-pipelines)
5. [Radiative samplers](#5-radiative-samplers)
6. [PSF models](#6-psf-models)
7. [Correction](#7-correction)
8. [Fitting a PSF](#8-fitting-a-psf)
9. [High-level API](#9-high-level-api)
10. [Analysis and the `.adjeff` accessor](#10-analysis-and-the-adjeff-accessor)
11. [Cache](#11-cache)
12. [Logging](#12-logging)
13. [Installation details](#13-installation-details)

---

## 1. The adjacency effect

The atmosphere scatters light from the neighbours of a pixel into the
sensor. For a surface reflectance $\rho_s$, the top-of-atmosphere
reflectance is

$$\rho_{toa} = \rho_{atm} + T^\downarrow \frac{T^\uparrow_{dir}\,\rho_s + T^\uparrow_{dif}\,(\rho_s \ast P)}{1 - s\,(\rho_s \ast P)}$$

| Symbol | Variable | Meaning |
|---|---|---|
| $T^\downarrow = T^\downarrow_{dir} + T^\downarrow_{dif}$ | `tdir_down`, `tdif_down` | Sun to surface |
| $T^\uparrow_{dir}$, $T^\uparrow_{dif}$ | `tdir_up`, `tdif_up` | Surface to sensor |
| $\rho_{atm}$ | `rho_atm` | Path reflectance |
| $s$ | `sph_alb` | Spherical albedo |
| $P$ | | Point spread function |

$P$ depends on the aerosols (`aot`, `rh`, `href`, species), the
geometry (`sza`, `vza`, `saa`, `vaa`), the wavelength and the ground
elevation `h`. Adjeff simulates $\rho_{toa}$, fits $P$ as an analytical
kernel, and inverts the formula to recover $\rho_s$.

## 2. Scenes

An `ImageDict` holds one `xr.Dataset` per band, since bands may differ
in resolution. Modules add variables to it; extra dimensions such as
`aot` appear as xarray dimensions.

```python
from adjeff.core import ImageDict, S2Band

scene = ImageDict({S2Band.B02: xr.Dataset({"rho_s": rho_s_b02})})
scene[S2Band.B02]["rho_s"]
```

Synthetic scenes take a pixel size and either `n` or `extent_km`:

```python
from adjeff.core import disk_image_dict, gaussian_image_dict, random_image_dict

bands = [S2Band.B02, S2Band.B03]
gauss = gaussian_image_dict(sigma=0.5, res_km=0.01, rho_min=0.05, rho_max=0.6, bands=bands, n=101)
disk = disk_image_dict(radius=1.0, res_km=0.01, rho_min=0.05, rho_max=0.6, bands=bands, n=101)
noise = random_image_dict(bands, ["rho_s"], res_km=0.01, seed=0, n=101)
```

Gaussian and disk scenes record their model and parameters, so that
`RhoToaSymSampler` can redraw them exactly.

## 3. Atmosphere and geometry

Three pydantic configs describe a simulation. Each field takes a
scalar, a list (swept) or a DataArray, e.g. a map on `(y, x)`; every
dimension propagates to the outputs.

```python
from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig

atmo = AtmoConfig(aot=[0.05, 0.1, 0.2], h=0.0, rh=50.0, href=2.0, species={"sulphate": 1.0})
geo = GeoConfig(sza=30.0, vza=10.0, saa=120.0, vaa=120.0)
spectral = SpectralConfig.from_bands(bands)
```

| Field | Unit | |
|---|---|---|
| `aot` | | Aerosol optical thickness at 560 nm |
| `h` | km | Ground elevation, in [0, 10] |
| `rh` | % | Relative humidity |
| `href` | km | Aerosol scale height |
| `species` | | OPAC species fractions, summing to 1 |
| `sza`, `vza` | ° | Sun and viewing zenith angles |
| `saa`, `vaa` | ° | Sun and viewing azimuth angles |
| `sat_height` | km | Satellite altitude, 700 by default |

`adjeff.make_full_config(bands, aot=..., sza=...)` builds the three at
once, as a dict to unpack with `**cfg`.

## 4. Scene modules and pipelines

Every step is a `SceneModule`: it reads `required_vars`, writes
`output_vars`, and leaves the rest of the scene untouched.

```python
from adjeff.modules import SceneModule

class Double(SceneModule):
    _required_vars = ["rho_s"]
    _output_vars = ["rho_2s"]

    def _compute(self, scene):
        for band in scene.bands:
            scene[band]["rho_2s"] = 2 * scene[band]["rho_s"]
        return scene
```

Calling a module checks its inputs, reads the cache or computes, refuses
NaN outputs, and records provenance. `rename` binds a role to another
variable, to keep two estimates side by side:

```python
king = Unif2Surface(kernels=tree_king, rename={"rho_s": "rho_s_king"})
```

A `SceneSource`, such as a loader, creates a scene and needs no input.
`MajaLoader` reads a MAJA L2A product with its aerosols, geometry and
DEM:

```python
from adjeff.modules.loaders import MajaLoader

scene = MajaLoader(product_path, bands=bands, res=0.12, mnt_path=dem_path)()
```

A `Pipeline` chains modules and checks at construction that each input
is produced upstream. `stream_dims={"aot": 3}` runs it on chunks, to
bound memory.

## 5. Radiative samplers

The samplers run Smart-G (CUDA GPU) over every atmospheric state of the
configs, through an [xsweep](https://github.com/walcark/xsweep) sweep.

| Sampler | Writes | Method |
|---|---|---|
| `TdirDownSampler`, `TdirUpSampler` | `tdir_down`, `tdir_up` | From the optical depth |
| `TdifDownSampler`, `TdifUpSampler` | `tdif_down`, `tdif_up` | Monte Carlo |
| `RhoAtmSampler` | `rho_atm` | Monte Carlo |
| `SphAlbSampler` | `sph_alb` | Monte Carlo, no geometry |
| `TdifUpBrdfSampler`, `SphAlbBrdfSampler` | `tdif_up`, `sph_alb` | Over an RTLS surface |
| `RhoToaSymSampler` | `rho_toa` | Radially symmetric analytical surface |
| `RhoToaSampler` | `rho_toa` | Arbitrary surface, on a pixel sub-grid |
| `WuPsfSampler` (`adjeff.reference`) | `psf_atm` | Wu et al. (2024), for comparison |

`RadiativePipeline` runs the six terms in order; with
`rtls=(k0, k1p, k2p)` it uses the RTLS variants:

```python
from adjeff.modules.samplers import RadiativePipeline, RhoToaSymSampler

scene = RadiativePipeline(atmo, geo, spectral, remove_rayleigh=False)(scene)
scene = RhoToaSymSampler(atmo, geo, remove_rayleigh=False, nr=80, n_ph=int(1e6))(scene)
```

Every sampler takes `batch_size` (states per Smart-G call: cost only)
and `dedup` (merge repeated states: worth it when parameters are
spatial maps). `SMARTG_DIR_AUXDATA` must point at the Smart-G auxiliary
data, or Smart-G returns NaN, which the modules turn into an error.

## 6. PSF models

Analytical PSFs are `torch.nn.Module`s with bounded parameters, sampled
on an odd, square `PSFGrid`.

| Class | Profile | Parameters |
|---|---|---|
| `GaussPSF` | $e^{-r^2/2\sigma^2}$ | `sigma` |
| `GeneralizedGaussianPSF` | $e^{-(r/\sigma)^n}$ | `sigma`, `n` |
| `VoigtPSF` | Gaussian and Lorentzian mix | `sigma`, `gamma` |
| `KingPSF` | $(1 + r^2/2\sigma^2\gamma)^{-\gamma}$ | `sigma`, `gamma` |
| `MoffatGeneralizedPSF` | $(1 + (r/\alpha)^{2\beta})^{-\gamma}$ | `alpha`, `beta`, `gamma` |

`NonAnalyticalPSF` holds a fixed kernel.

```python
from adjeff.core import GaussPSF, PSFGrid, freeze, psf_kernel, psf_params

psf = GaussPSF(PSFGrid(res=0.01, n=101), S2Band.B02, sigma=0.3)
psf.forward()          # torch.Tensor (101, 101), sums to 1
psf.to_dataarray()     # dims (y_psf, x_psf), with model and parameters in attrs
```

Frozen PSFs live in an `xr.DataTree`, one group per band, each with
its own grid and the sweep dims of its fit. It round-trips through zarr.

```python
tree = freeze({S2Band.B02: psf})
psf_kernel(tree, S2Band.B02)      # kernel
psf_params(tree, S2Band.B02)      # {"sigma": DataArray}
tree.to_zarr("psf.zarr", mode="w")
```

## 7. Correction

For a uniform surface, `Unif2Toa` applies the 5S formula and `Toa2Unif`
inverts it, giving `rho_unif`, the reflectance a uniform surface would
need to produce `rho_toa`:

```python
from adjeff.modules.classic import Toa2Unif

scene = Toa2Unif()(scene)    # needs rho_toa and the six radiative terms
```

`Unif2Surface` then convolves `rho_unif` with the PSF into $\rho_{env}$
and solves the formula for `rho_s`:

```python
from adjeff.modules.models import Unif2Surface

scene = Unif2Surface(kernels=tree)(scene)
```

## 8. Fitting a PSF

`fit` optimises the PSF of a model over reference scenes, once per
band and atmospheric state, and returns the frozen tree.

```python
from adjeff import fit, make_model
from adjeff.core import KingPSF
from adjeff.modules.models import Unif2Surface
from adjeff.optim import AdamConfig, LBFGSConfig, Loss, Metric, TrainingImages

model = make_model(Unif2Surface, KingPSF, bands, res_km=0.12, n=1999,
                   init_parameters={"sigma": 0.1, "gamma": 2.0})
tree = fit(model, TrainingImages(images=[scene_1, scene_2]))
```

The default schedule is an Adam warm-up, then L-BFGS, on
`Loss(Metric.RMSE_RAD)`. To change it:

```python
loss = Loss(Metric.MSE_RAD, mask_on=15.0)      # within 15 km of the centre
tree = fit(model, images, stages=[
    AdamConfig(min_steps=5, max_steps=20, loss_relative_tolerance=1e-4, loss=loss),
    LBFGSConfig(min_steps=5, max_steps=50, loss_relative_tolerance=1e-6, loss=loss),
], store="psf.zarr")                            # kernels streamed to zarr
```

| Metric | |
|---|---|
| `MAE`, `MSE`, `RMSE` | Uniform over the masked pixels |
| `MAE_RAD`, `MSE_RAD`, `RMSE_RAD` | Every radius weighted the same |

Residuals are divided by the reference's maximum. `mask_on` is a
variable name (pixels within its 99 % radial energy, `"rho_unif"` by
default), a radius in km, or `None`.

`loss_landscape` and `energy_radius_landscape` evaluate the loss and the
encircled-energy radii over a list of PSFs, e.g. a parameter grid.

## 9. High-level API

`adjeff` exports one-call functions for the common paths:

| Function | |
|---|---|
| `load_scene(loader)`, `load_maja(...)` | Load a product, with its aerosol species |
| `make_full_config(bands, ...)`, `load_config(scene, band)` | Configs from scalars, or from a scene |
| `run_radiatives_from_scene(scene)` | The six radiative terms, configured from the scene |
| `run_forward_pipeline(scene, **cfg)` | Radiative terms, `rho_toa`, then `rho_unif` |
| `make_model(...)`, `fit_psf(...)`, `apply_psf(...)` | Build, fit on simulated disks, and apply a PSF |
| `sample_psf_atm(...)`, `sample_psf_atm_from_scene(...)` | PSF of Wu et al. (2024) |

`fit_psf` reads the mean atmosphere and geometry of the scene with
`load_config(..., aggregate=True)`, so the scene must carry `aot`, `h`,
`rh`, `href`, `sza`, `vza`, `saa` and `vaa`, as a loaded product does.

## 10. Analysis and the `.adjeff` accessor

`adjeff.analysis` quantifies a PSF or a 2-D field:

```python
from adjeff.analysis import bias, encircled_radius, fwhm, mtf, rmse

encircled_radius(kernel, 0.5)                        # radius holding half the energy
encircled_radius(kernel, 0.5, normalize="plane")     # of the plane energy instead
fwhm(kernel)
mtf(kernel)                                          # on dim "f", cycles per km
rmse(estimated, truth, mask=15.0, radial=True)       # within a 15 km disk
```

Every DataArray carries the `.adjeff` accessor:

```python
da.adjeff.radial()                  # azimuthal mean against radius
da.adjeff.radial("cdf")             # cumulated energy
profile.adjeff.to_field(ds)         # 2-D field from a profile
da.adjeff.transect(45.0)            # values along a line
da.adjeff.res, da.adjeff.n          # pixel size, pixels per side
da.adjeff.tidy()                    # length-one dims to scalar coords
da.adjeff.untidy()                  # and back, before xr.merge
```

## 11. Cache

Every module takes a `CacheStore`. Results are stored in Zarr, keyed by
the module, its configuration and the provenance of its inputs; an
identical call reads them back instead of computing, and outputs are
held as lazy views.

```python
from adjeff.utils import CacheStore

cache = CacheStore("./adjeff_cache")
pipeline = RadiativePipeline(atmo, geo, spectral, remove_rayleigh=False, cache=cache)
scene = pipeline(scene)   # computed, then cached
scene = pipeline(scene)   # read back, no GPU call
```

Without a cache, every output stays in RAM.

## 12. Logging

Adjeff prints nothing until asked:

```python
adjeff.setup_logging(level="info")               # console
adjeff.setup_logging(level="debug", json=True)   # one JSON object per line
```

`xsweep` follows the same level; chatty dependencies (zarr, matplotlib…)
are capped at `warning`, and Python warnings join the stream. Lines are
`event key=value`:

```
[info ] module.start   module=TdirDownSampler bands=1 key=18e7e4dd stage=1/3
[info ] sweep.plan     states=1 bands=1 n_ph=10000 batch_size=64 dedup=False
[info ] module.done    module=TdirDownSampler cached=False duration_s=2.204
```

Everything is an ordinary `logging` record under `adjeff`, so an
application's own handlers receive it.

## 13. Installation details

| Environment | GPU | Contents |
|---|---|---|
| `cpu` | | The library |
| `gpu` | ✓ | The library |
| `dev` | | With ruff, mypy, pytest |
| `dev-gpu` | ✓ | With ruff, mypy, pytest |
| `notebooks` | | With JupyterLab |
| `notebooks-gpu` | ✓ | With JupyterLab |

```bash
pixi install -e dev-gpu
export SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata   # e.g. in ~/.bashrc
pixi run -e notebooks-gpu jupyter lab notebooks/
```

| Notebook | GPU | |
|---|---|---|
| `01-create-and-display-image` | | Scenes, radial profiles |
| `02-atmospheric-configuration` | | Configs, sweeps |
| `03-compute-radiative-quantities` | ✓ | `RadiativePipeline`, cache |
| `04-simulate-rho-toa` | ✓ | `RhoToaSymSampler` |
| `05-compute-2d-radiative-from-maja-output` | ✓ | MAJA product, spatial maps |
| `06-learn-psf` | ✓ | PSF fitting end to end |
