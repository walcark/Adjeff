# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.11.0]

### Added

- **`adjeff.analysis`**, for what quantifies a PSF or a two-dimensional
  field of adjacency effect.  Nothing here is new behaviour: the radial
  profile lived in the `.adjeff` accessor, the encircled energy was
  written a second time inside `optim/landscape.py` for speed, and the
  error metrics on DataArrays lived in the article's own repository
  because the package had none.

  - `radial_profile`, `transect`, `to_field`, moved out of the accessor,
    which now spells them rather than implementing them.
  - `radial_profile(symmetric=True)` mirrors the profile around `r = 0`,
    so that a transect can be drawn across its full width.  The values
    are unchanged: a radial profile is symmetric by construction, this
    only writes the other half down.
  - `encircled_energy`, `encircled_radius` and the batched
    `encircled_radii`, which `energy_radius_landscape` now delegates to,
    dropping 55 lines of duplicated binning to 8.  Checked against the
    closed form of a Gaussian, which neither previous implementation
    was.
  - `fwhm` and `mtf`.  The modulation transfer function is how the
    instrument literature reports a PSF, and the package had no way to
    produce one.
  - `rmse`, `mae` and `bias` on DataArrays, with a mask given as a field
    or as a radius in km, and an optional radial weighting.  A shape
    mismatch raises instead of broadcasting into a number that means
    nothing.

  One rule decides what belongs there: does it quantify a PSF, or a 2-D
  field of adjacency effect?  A generic 1-D FFT, a power spectral
  density, a Wiener filter do not.  `scipy.signal` exists.

## [0.10.0]

### Added

- **A module chooses the slots it reads and writes.** A class declares
  canonical *roles* (`_required_vars`, `_output_vars`, `_optional_vars`)
  and an instance binds them to Dataset names through `rename=`:

  ```python
  king = Unif2Surface(kernels=tree, rename={"rho_s": "rho_s_king"})
  gauss = Unif2Surface(kernels=other, rename={"rho_s": "rho_s_gauss"})
  scene = gauss(king(scene))      # truth and both estimates, side by side
  ```

  It answers a collision the article kept working around: `Unif2Surface`
  writes `rho_s`, which is also the name of the ground truth, so a scene
  carrying both lost one of them.  Renaming the module's output alone
  would only move the collision, since `RhoToaSampler` then asks for
  `rho_s` in turn; binding roles to slots is what lets each estimate
  travel under its own name and still be read by the next module.

  The cache is keyed and filled by role, so two instances differing only
  by their slots share one entry: where a result is written changes
  nothing to what is computed.

- **`SensorBand.from_wl(560.0)`** returns the band centred on a
  wavelength, which every figure naming a band on its command line was
  building a table for.  An approximate value is an error rather than a
  silent neighbour.

- **`adjeff.modules.samplers.RADIATIVE_VARS`** publishes the six
  quantities of the 5S formula.  Each sampler declared its own; nothing
  named the set.

- **`da.adjeff.tidy()` and `da.adjeff.untidy()`.** A scalar in a
  configuration is coerced to an array of length one, so an output
  carries an `aot`, `rh`, `h` and `href` dimension even when one
  atmospheric state was simulated, and every caller peeled them off by
  hand.  `tidy` turns those into scalar coordinates: the dimensions go,
  the values stay, so the array still says which state produced it.

  Selecting one value of a *swept* parameter needs neither: `sel` already
  removes the dimension and keeps the coordinate.

  A tidied array recombines with `concat`, which promotes the coordinate
  back to a dimension.  It does not recombine with `merge` or
  `combine_by_coords`, which align on dimensions: `untidy` first.  The
  modules themselves accept either form, since they align on values.

- **`ARTICLE_TRAIN_RADII_KM` and `ARTICLE_TRAIN_SIZE`** name the
  manuscript's training choices, which `fit_psf` used to impose as
  unnamed literals.  They remain its defaults, and a reader can now tell
  a decision of the article from a decision of the library.

### Changed

- **`TrainingImages(weights=...)` is optional**, and defaults to uniform
  weights.  Every caller wrote `[1.0] * len(images)` by hand.  A wrong
  count now raises instead of failing later on a mismatched zip.

  Breaking: `required_vars`, `output_vars` and `optional_vars` are
  properties resolved per instance.  A subclass declaring them as class
  attributes silently shadows the resolution, so the class-level
  declarations were renamed with a leading underscore, and reading them
  off a class (`SomeModule.output_vars`) now returns a property object.
  Use `SomeModule._output_vars` for the roles, or an instance for the
  slots.

## [0.9.0]

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

### Added

- **`SceneModule.optional_vars`** declares inputs a module consumes when
  the scene carries them and computes itself otherwise.  They enter the
  cache key only when present, which `required_vars` could not express:
  declaring them there would forbid the standalone call that produces
  them.

### Removed

- `GeoConfig.sun_le`, `sat_le`, `sun_sensor` and `sat_sensor`, plus
  `satellite_relative_position`.  None had a caller outside the tests;
  `_smartg.py` builds those dictionaries and sensors locally.  They were
  not merely redundant: `satellite_relative_position` computes its
  `x` offset from `180 - vaa` where the production path uses `vaa`, so
  the two disagree on the sign. Deleting is safer than unifying, since
  only the used path is validated by the article's figures.

### Changed

- **`import adjeff` no longer needs `SMARTG_DIR_AUXDATA`.** Two
  module-level Smart-G imports remained in `atmosphere/atmo_factory.py`
  and `atmosphere/surface.py`, where 23 others are already deferred into
  function bodies.  Anyone who only wants the CPU half of the library,
  the accessor, the image generators, the PSF models, the radial
  analysis, can now import it.  Covered by `test_import_without_smartg`,
  which runs in a subprocess with the variable removed.
- A degenerated L-BFGS line search is logged at `warning` instead of
  `info`.  The public `OptimizationWarning` is unchanged: Python shows a
  warning once per call site, which hides how often the case occurs over
  a sweep, so the log line is the one that counts them.
- The 88 column limit of the project guidelines replaces 79, and the
  tests join the lint and format scope.

## [0.8.1]

### Fixed

- **The plain metrics masked on the residual they were measuring.**
  `mae`, `mse` and `rmse` ignored the domain the caller passed and built
  their own from the 99% radial energy of `|eps|`.  That closes a
  feedback loop: the optimiser can lower the loss by shrinking its own
  mask rather than by fitting better.  Measured on the manuscript's
  landscapes, it collapses the fit, the King core width falling from
  0.19 km to 0.0065 km, its 50% energy radius to zero, and the
  generalisation error growing by a factor 2.6.  The formula published
  in the article, which masks on `eps` squared, is worse still at a
  factor 4.4.  All six metrics now take their domain from the caller.

- **The residual was scaled on the prediction.** It was divided by
  `max(|pred|, |truth|)`, a normalisation the manuscript does not
  mention, which made the metric depend on the very quantity being
  optimised.  With a mask it became plainly wrong: a prediction going
  astray far outside the mask raised the scale for every pixel, so the
  error measured inside the mask, on pixels that had not moved, fell by
  two orders of magnitude.  The reference alone now sets the scale.

  On the three landscapes of the manuscript the fitted parameters do not
  move, since all three peak at 1.0 and the scale is therefore 1.

### Changed

- **`KingPSF` confines its power-law index to `[1, 5]`.** On a plane the
  radial integral of the profile converges only for `gamma > 1`; below
  that the kernel has no scale of its own and the grid sets its
  normalisation, with half of the energy sitting beyond 79 km on a
  141 km grid at `gamma = 0.4`.  That region is also where the loss
  surface turns concave, which is where L-BFGS stalled: one start out of
  sixteen without the Adam warm-up, none with the bound in place, over
  32 runs.  `VoigtPSF` and `MoffatGeneralizedPSF` document their own
  integrability conditions, which a per-parameter bound cannot express.

- **`Loss(mask_on=...)` accepts any variable name, or a radius in km.**
  It was restricted to `"rho_unif"` or `None` by a hand-written check.
  A radius keeps a disc, and unlike an energy mask it does not move when
  the prediction changes, which is what a quasi-Newton step assumes.

- **`loss_landscape` takes any model and any loss.** It hardcoded the
  convolution of `Unif2Surface`, the variable names it reads, and
  reached inside the loss object for its metric and its mask, so the
  promise of its own docstring held for exactly one pair.  A new
  `kernel=` argument on `forward_band` is what lets a candidate be
  evaluated without installing it in the model.  Coverage of the module
  goes from 19% to 100%.

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
