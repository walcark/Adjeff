# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **The forward pipeline gave a different scene on a warm cache.**
  `rho_atm` is a single Monte-Carlo number, and `RhoToaSym` drew its own
  instead of reusing the one `RadiativePipeline` had already put in the
  scene.  On a cold run that second draw overwrote the first and fed
  `Toa2Unif`; on a warm run the sampler came from the cache, its
  `_compute` never ran, and `Toa2Unif` saw the first draw instead.
  `rho_toa` was then built with one draw and inverted with another,
  leaving a constant bias of around `6e-5` on the whole `rho_unif` field.
  Both `RhoToaSym` and `RhoToa` now reuse the scene's `rho_atm` when it
  is there, and declare it in the new `optional_vars` so that two
  different path reflectances cannot share one cache entry.

  The bias was small in absolute terms but decided the outcome of any
  radial energy mask: a 99% mask on `rho_unif` kept 4% of the grid
  without it and 95% with it.

  Breaking: `RhoToaSym` and `RhoToa` cache entries are invalidated, since
  `rho_atm` now takes part in their key.

- **`SceneModule.optional_vars`** declares inputs a module consumes when
  the scene carries them and computes itself otherwise.  They enter the
  cache key only when present, which `required_vars` could not express:
  declaring them there would forbid the standalone call that produces
  them.

## [0.8.0]

### Changed

- **One `fit()` replaces five optimiser entry points.** `_Optimizer`,
  `SingleStageOptimizer`, `OptimizerPipeline`, `AdamOptimizer` and
  `LBFGSOptimizer` were one algorithm behind five names: build an
  object, call `.run(model)`. There is now a function,
  `fit(model, train_images, loss=..., stages=..., store=...)`, and the
  stages are named by their configuration (`AdamConfig`, `LBFGSConfig`)
  rather than by a wrapper class the caller had to import.
  `api.optimize_adam_lbfgs` is gone; `fit` is its replacement and is
  exported from `adjeff`.

  `fit` also stops replaying parameter snapshots. It captures each
  kernel when its combo finishes, which is when the model already holds
  it, instead of restoring the parameters afterwards to rebuild what was
  there.

- **`PSFConvModule.psf_params(band)`** returns the parameters a fitted
  PSF currently holds. Reading them meant walking `model.modules()`
  looking for anything answering `param_dict()`.

- **A much smaller public surface.** `adjeff.api` publishes 12 names
  instead of 52: it had no `__all__`, so `Path`, `np`, `xr`, `cast`,
  `TypeVar` and 29 re-exported classes were part of it by accident.
  `adjeff.utils` publishes 6 instead of 23, the ones a caller outside
  the package needs (`CacheStore`, the two `fft_convolve_2D`, and the
  three building blocks of a custom PSF). The rest is internal plumbing
  and stays reachable through its own submodule, e.g.
  `from adjeff.utils.radial import bin_radial`.

  Breaking: `config_from_scene` is gone, `load_config` does everything
  it did and more; `make_atmo_config` and `make_geo_config` are private,
  `make_full_config` builds both; `apply_psf(psf_dict=)` becomes
  `apply_psf(tree=)`; `TrainingSet` and `extend_analytical` leave the
  public listing, the latter deleted since nothing called it.

### Changed

- **Every sampler runs on [xsweep](https://github.com/walcark/xsweep).**
  The three that loop over geometry — `RhoToaSampler`, `RhoToaSymSampler`
  and `WuPsfSampler` — declare `loop(sza, vza) vec(...)`: the sensor grid
  is rebuilt per angle, so a call carries one geometry and those axes
  cannot be batched.

- **The six radiative samplers run on xsweep's batch clause.**
  They declare a contract instead of a sweep: `batch(aot, rh, h, href,
  sza) vec(wl) -> tdir_down(wl)`. The atmospheric states stay sweep axes,
  so dedup and resumption keep working, but Smart-G still receives a
  whole group per call — calling it once per state costs 3x, since it
  amortises the atmospheric profile over the batch.

  `sweep_chunks` and `deduplicate_dims` become `batch_size` and `dedup`,
  in the samplers, in `RadiativePipeline` and in `api`. `dedup` is now a
  flag rather than a dim list: xsweep collapses repeated states wherever
  they are, and guarantees the result is unchanged.

- **`PSFDict` is gone.** It was two types under one name: a mode flag
  governed six methods, `to_dataarray()` raised in one mode and
  `get_module()` in the other, and `params()` had two competing storage
  strategies. Frozen kernels are now an `xarray.DataTree`, one group per
  band; live PSFs are a plain `dict[SensorBand, PSFModule]`.
  `PSFConvModule` takes `psfs=` or `kernels=`, exactly one of the two,
  so the mode is visible at the call site instead of hidden in the
  object. 362 lines become 164, fully covered.
- `da.adjeff.band` becomes `da.adjeff.band_id` and returns the id string.
- **Published methods move to `adjeff.reference`.** `PsfAtmSampler`
  implements the sampled PSF of Wu et al. (2024), the baseline this work
  is compared against, but sat unlabelled among adjeff's own samplers
  with no citation. It is now `adjeff.reference.WuPsfSampler`, in a
  package where one module means one paper. Reading
  `adjeff.modules.samplers` tells you what adjeff computes; reading
  `adjeff.reference` tells you what it is measured against.

  Breaking: `adjeff.modules.samplers.PsfAtmSampler` is gone, with no
  alias. `api.sample_psf_atm` is unchanged. Cached atmospheric PSFs are
  invalidated, since the module name enters the cache key.

### Fixed

- **The cache refused to store an array it had just handed back.**
  `forward` swaps its outputs for lazy zarr-backed views, so they carry
  the file's own `encoding["chunks"]`; `to_zarr` honours that encoding
  over the chunking the cache asks for and raises rather than write. It
  only showed up once a swept dimension was longer than one, since the
  two chunkings agree otherwise. The cache and `write_band` now drop the
  encoding they did not choose.

- **A batched Smart-G call mixed up the angles between its points.**
  `tdif_down`, `tdif_up` and `rho_atm` ask Smart-G for one direction per
  point, but the engine evaluates every direction for every atmosphere
  it is handed, so the return is the cross product of the two. Only the
  diagonal is meaningful: point `i` asked for angle `i`. Sweeping two or
  more atmospheric states at once raised a shape error from xsweep, and
  a case where it did not raise would have returned a value computed for
  another point's geometry. Found by the article's `table_aot` figure,
  which sweeps three AOT values in one call.

- **`saa` and `vaa` were declared as arrays and used as scalars.** Every
  `_smartg` function read `float(saa.flat[0])` from what its signature
  called an `np.ndarray`. The loop samplers already passed `.item()`;
  only the radiative ones passed an array. Found by xsweep refusing to
  fingerprint an ndarray static, which is the check the cache needed.

- **A frozen PSF could not be written to zarr.** `to_dataarray()` stored
  the `SensorBand` enum in `attrs`, which is not JSON serialisable, so
  `PSFDict.to_zarr()` raised on any tree built from `to_frozen()`. It
  went unnoticed because the optimiser's own path dropped attrs on the
  way. The attribute now holds the band id, and the round-trip is
  tested, heterogeneous per-band grids included.

## [0.7.0]

Correctness release. Five defects were found by an audit of the cache and
the streaming pipeline; none of them were visible to `ruff` or to
`mypy --strict`, and each is now covered by a regression test.

### Fixed

- **Cache keys now reject parameters they cannot see.** `_config_dict()`
  reads `__init__` parameters off same-named attributes, so a parameter
  stored under a private name was silently dropped from the key and two
  configurations collided on one entry. It now raises
  `ConfigurationError` instead of guessing.
- **`deduplicate_dims` and `sweep_chunks` reach the cache key.** Both
  change the shape of a sampler's output; they were stored privately on
  `SceneModuleSweep` and therefore invisible to the key.
- **`ProductLoader.res` reaches the cache key.** The same MAJA product
  loaded at two target resolutions returned the first one's tiles.
- **A truncated cache entry now reads as a miss.** `CacheStore.load_vars`
  returned whichever variables happened to be present, reported as a hit,
  handing back a scene silently short of an output.
- **`Pipeline(stream_dims=...)` over several dimensions.** The Cartesian
  product of chunks was concatenated along a single axis, stacking
  `n0 * n1` pieces instead of rebuilding the grid.
- **Streaming preserves band `attrs`.** The aerosol species written by
  `load_scene()` was dropped, so `load_config()` fell back to sulphate
  without a word.

- **`PsfAtmSampler` enriches the scene instead of replacing it.** It
  returned a fresh `ImageDict`, dropping every input variable, which
  broke the `SceneModule` contract and made it unusable anywhere but
  first in a pipeline. `api.sample_psf_atm` never noticed because it
  builds a throwaway scene and reads only `psf_atm`.

### Added

- **Integration tests** (`tests/test_integration.py`, marked
  `integration`). The unit suite left `_smartg.py` at 16% coverage and
  `api.py` at 29%: it never ran the physics. These drive the real chain
  on a tiny grid and found the two defects above on their first run.
  Excluded from the default run by `addopts` and skipped without CUDA,
  so CI never attempts them. Run with
  `pixi run -e dev-gpu test-integration`.

- `adjeff.sweep` is gone, and with it `SweepBundle`, `UniqueIndex` and
  `SceneModuleSweep`: 617 lines that xsweep now provides, with a store,
  resumption and a plan on top. `ParamBatch` stays, no longer pretending
  to be a sweep engine: it is Smart-G's own flattening of the `vec` axes.

### Removed

- `SweepBundle.from_configs`, `CacheStore.clear_function` (which looked
  entries up by module name, though they are keyed by hash),
  `PSFDict._cache_dict`, and `_Config.unique` / `.iter` / `.run`. None had
  a caller in the package.
- `adjeff.modules.TestModule`, a test double that also triggered a pytest
  collection warning on every run. It now lives in `tests/`.

### Changed

- `geoclide` is pinned below 4. smartg 1.2.0 calls its `get_rotateX_tf`,
  renamed to `get_rotate_x_tf` in geoclide 4.0.0, which made
  `PsfAtmSampler` raise `AttributeError`. Only the PSF sampler reaches
  that code path, so the break was invisible until it ran. Lift the
  bound once smartg ships a release built against geoclide 4.
- The GPU environment moves to CUDA 12.9. `nvcc` only learns `sm_120`
  (Blackwell, RTX 50xx) from 12.8 on, and Smart-G compiles its kernels
  for the local compute capability.

### Upgrading

Sampler cache entries written by 0.6.0 are invalidated: `sweep_chunks`
and `deduplicate_dims` now take part in the hash. The first run after
upgrading recomputes them.
