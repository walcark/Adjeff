# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

### Removed

- `SweepBundle.from_configs`, `CacheStore.clear_function` (which looked
  entries up by module name, though they are keyed by hash),
  `PSFDict._cache_dict`, and `_Config.unique` / `.iter` / `.run`. None had
  a caller in the package.
- `adjeff.modules.TestModule`, a test double that also triggered a pytest
  collection warning on every run. It now lives in `tests/`.

### Changed

- The GPU environment moves to CUDA 12.9. `nvcc` only learns `sm_120`
  (Blackwell, RTX 50xx) from 12.8 on, and Smart-G compiles its kernels
  for the local compute capability.

### Upgrading

Sampler cache entries written by 0.6.0 are invalidated: `sweep_chunks`
and `deduplicate_dims` now take part in the hash. The first run after
upgrading recomputes them.
