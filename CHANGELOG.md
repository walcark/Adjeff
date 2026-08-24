# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

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
