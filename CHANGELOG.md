# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.15.0]

Smart-G 2.0, which renamed most of what adjeff calls.

Smart-G 2.0 is a rewrite of the public API rather than a version bump:
`Smartg.run` returns an `xarray.Dataset` instead of an `MLUT`, its
keywords moved from capitals to snake_case, and the scene classes left
`smartg.smartg` for `smartg.sensor`, `smartg.surface`, `smartg.albedo`
and `smartg.objects3d`. It is built against geoclide 4, which lifts the
bound 1.2.0 forced on adjeff and with it the break that only
`PsfAtmSampler` ever reached.

Five defects were found during the port, each hidden by the previous
one, and the last of them is the reason this release ships two test
files rather than one. `docs/plan-smartg-2.0-0.15.0.md` records the
correspondence table and how each was found.

### Changed

- **adjeff calls Smart-G 2.0.** Every import, keyword and return type
  follows the new API. `Smartg.run` already returns a `Dataset`, so the
  `MLUT` conversion that opened `adapt_smartg_output` is gone;
  the normalisation it performs stays, the variable and dimension names
  being unchanged.

- **The sensor zenith is explicit** in `tdif_up` and `tdif_down`.
  `Sensor.th_deg` defaulted to 0, the zenith, and now defaults to 180,
  the nadir: two flux collectors silently turned over and returned
  zeros. Nothing in the diff showed it, and no signature check could:
  the parameter kept its name. The 6S reciprocity
  `tdif_up(θ) ≈ tdif_down(θ)` is what caught it.

- **pycuda comes from conda in every environment.** Smart-G depends on
  it hard, including on the cpu side, and the PyPI sdist compiles
  against `cudaProfiler.h` and g++.

### Added

- **`tests/test_smartg_api.py`** reads the AST of `src/` and checks
  every keyword adjeff passes to the fourteen Smart-G callables against
  `inspect.signature`. Two seconds, against the two and a half minutes
  of a GPU run, and it is what found `Entity(TC=)` and
  `Smartg(obj3D=)` after a first rename pass had missed them.

- **`tests/test_smartg_defaults.py`** freezes the thirty-one defaults
  adjeff relies on by not passing them. A default that moves is
  invisible in a diff and silent at runtime; this turns the next one
  into a one second failure naming the parameter.

- **The `physics` marker** on the five tests that read a physical
  identity rather than a code path. `pytest -m physics` runs them in
  fifty seconds, apart from the rest of the integration suite.

### Upgrading

Smart-G 2.0 rejects scattering angles that are not strictly increasing,
which 1.x accepted. Any auxiliary table produced by MOPSMAP has to be
sorted before it can be read; see the plan document for the eight CAMS
species this affected.

## [0.14.0]

What the scalar terms of the 5S model become when the ground is not
Lambertian, and one sensor that had been pointed the wrong way.

The 5S formalism uses a single upward diffuse transmittance and a single
spherical albedo, which is exact only over a Lambertian surface. Of the six
quantities a forward run produces, exactly two depend on the surface model:
`tdif_up` and `sph_alb`. This release computes those two over a Ross-Li
surface, which is what quantifying the cost of the Lambertian assumption
requires.

### Added

- **`TdifUpBrdfSampler` and `SphAlbBrdfSampler`** sample `tdif_up` and
  `sph_alb` over an RTLS surface given its `(k0, k1p, k2p)` weights. They
  write the slot their Lambertian counterpart writes, so nothing downstream
  learns which one ran.

- **`RadiativePipeline(rtls=...)` and `run_forward_pipeline(rtls=...)`**
  swap the two surface-dependent samplers. `None`, the default, keeps the
  Lambertian ones. The surface the scalar terms assume is independent of the
  surface `rho_toa` is simulated over, which is what lets the two be varied
  one at a time.

- **`collect_batched` and `pair_angles_with_points`**, public in
  `adjeff.utils`. Four Smart-G kernels had each grown their own tail to pull
  a batched output apart; they now share one that places results by label
  rather than by the layout Smart-G happened to return.

### Fixed

- **`rho_atm`'s satellite sensor points where the caller asked.** The
  azimuth was passed through unchanged where Smart-G expects the direction
  the sensor looks along, not the direction it looks from. Measured against
  the Rayleigh phase function across relative azimuth, `vaa + 180` is the
  one that reproduces it. At nadir the correction is worth 0.07 %, so the
  article's geometry is unaffected; at `theta_v = 30` it is worth 12.5 %.

- **`RadiativePipeline(rename=...)` is no longer accepted and ignored.**
  It is now split per module: each sampler receives the part of the mapping
  naming a role it declares.

- **`RTLSSurface` is built through the keyword triple that works.** Smart-G
  1.1 raises on the per-argument form.

### Upgrading

Anything computed before this release with `theta_v != 0` was simulated at
the wrong azimuth. Clear the caches of those runs; `_cache_key` hashes the
module configuration, not the code, so a stale result is served silently.

## [0.13.3]

A fit that started on a singularity of its own model.

`KingPSF` takes its initial `gamma` from the caller, and every figure script
passed `1.0`, which is exactly `GAMMA_BOUNDS`' lower end. Two things go wrong
at once there. The plane integral of `(1 + r^2/2*sigma^2*gamma)^-gamma`
diverges at `gamma = 1`, so the initial kernel is defined only by its
truncation at the edge of the grid: the loss starts at `0.35` instead of the
`0.02` to `0.09` a usable start gives. And the sigmoid that maps the raw
parameter onto the interval is flat there, `dgamma/dp` around `1e-3` against
`0.75` at the middle, so Adam moves `gamma` by `2e-6` over its twenty steps
and L-BFGS is left to find the basin in one jump. Whether it lands well is
luck: on three Sentinel-2 bands fitted with identical settings, one stalled at
`0.080` while the others reached `0.049` and `0.037`.

### Fixed

- **`GAMMA_BOUNDS` starts above the integrability threshold**, `(1.02, 5.0)`
  rather than `(1.0, 5.0)`. The class docstring already explained why
  `gamma > 1` is required; the bound had been placed on the threshold itself,
  where `project()` sends the parameter back to a value the model does not
  admit and where the gradient is dead.

### Upgrading

Kernels fitted with an initial `gamma` of `1.0` are not comparable to those
fitted after this release: the optimiser starts from a different point. Clear
any PSF cache before refitting. Callers should also move their initial `gamma`
to the middle of the interval, `2.0` rather than `1.0`.

## [0.13.2]

The other half of moving the model.

0.13.1 moved the model to the device the fit runs on. `.to` carries
parameters and buffers, and `ConstrainedParameter` kept its bounds as
plain tensors, so the parameter left for the GPU and its bounds stayed
on the host.

### Fixed

- **A parameter's bounds move with the parameter.** `p_min` and `p_max`
  become non-persistent buffers. `forward` clamps against them and
  `project` does so in place, so leaving them behind put a mixed-device
  pair into every optimisation step. A test now walks the PSF module
  tree and rejects any tensor that is neither a parameter nor a buffer,
  which catches the next one without needing a GPU to run on.

### Upgrading

Nothing to change. 0.13.1 is superseded rather than broken: the bounds
are zero-dimensional, which torch is willing to treat as scalars, so the
mixed-device pair may well have gone unnoticed.

## [0.13.1]

The PSF was built on the CPU, once per training landscape.

`PSFGrid.meshgrid` allocated its tensors on whatever device torch
defaults to, which is the CPU, and `PSFModule.forward` called it afresh
on every evaluation. `fit` moved the training data to the requested
device but never the model, so the kernel was assembled on the host and
copied across the bus, gradient included, once per sample per step.

Measured on a CNES GPU node, at the manuscript's `n = 3999`: building a
King kernel takes **268 ms** on the one core the scheduler allocates,
against **1 ms** on the Tesla V100 of the same node, where the
convolution the fit is supposedly bound by takes 23 ms. With three
training landscapes, an Adam step spent about 2.4 s on the kernel and
0.2 s on everything the GPU was there for.

### Fixed

- **The radial grid is a buffer, built once.** `PSFModule` registers
  `_r2` at construction and exposes `r` and `r2`; the five analytical
  kernels read them instead of calling `meshgrid`. It is non-persistent,
  being derived from `grid` rather than learned, so it stays out of a
  checkpoint. `PSFGrid.meshgrid` gains a `device` argument.

- **`fit` moves the model to its device.** The training data already went
  there; the model, and with it the parameters and the radial grid, did
  not.

- **The kernel is evaluated once per step, not once per landscape.**
  `_ComboStage._total_loss` builds it and passes it through the `kernel`
  argument `forward_band` already accepted for the loss-landscape scan.

- `float(loss_t)` in the Adam loop becomes `float(loss_t.detach())`,
  which is what the surrounding code meant and what torch was warning
  about on every step.

### Upgrading

Nothing to change. `PSFGrid.meshgrid()` keeps its signature, with
`device` optional, and the kernels are numerically unchanged.

## [0.13.0]

Observability, and what an encircled-energy curve is a fraction of.

adjeff had 21 log calls in 11 371 lines, of which the nine at `info` all
said `"done"`, and it discarded everything its own dependencies were
saying. On a measured forward run of 13.8 s, 100 % of the wait was
unannounced: nothing said what was running, what it would cost, or how
long it had taken.

The second subject was planned as a release of its own and folded in
here, nothing having been published in between. It is one option and the
provenance needed to honour it.

### Changed

- **adjeff logs through the standard library.** structlog stays the
  writing API and stops being the transport. Left to its defaults it was
  both, and the defaults are `PrintLoggerFactory`, which writes to stdout
  outside `logging` altogether, and `BoundLoggerFilteringAtNotset`, which
  filters nothing. Two consequences: raising adjeff's level did not quiet
  `zarr` and quieting `zarr` did not raise `xsweep`, and with no level to
  set at all, all six notebooks began by redirecting structlog to
  `/dev/null`.

  Everything the package emits is now an ordinary `logging` record under
  the `adjeff` namespace. An application that already configures logging
  receives adjeff's lines in its own handlers, in its own format, without
  calling anything. `structlog.configure` is left alone: configuring the
  process is the application's decision.

- **Every log message is `object.action`**, lowercase, dotted, never
  interpolated: `module.done`, `cache.miss`, `sweep.plan`, `fit.step`.
  Three were f-strings, and those mattered: `fit` and both optimisers
  pre-formatted the step and the loss into the message, which undoes
  structlog entirely. `fit.step` now carries `optimizer`, `step`, `of`,
  `loss` and `delta_pct` as data, and `_loss_delta` returns a number
  rather than the string `Δ=+1.4%`.

- `maja.rh_defaulted` becomes a warning; `wu_psf.done` drops to debug.

### Added


- **`encircled_energy(..., normalize="plane")`**, and the same option on
  `encircled_radius`. Grid normalisation sends every curve to one at the
  edge of the domain, which is right for the operator, since that is the
  kernel the convolution applies. It hides how much each kernel left
  outside on the way: on the manuscript's aerosol sweep a King fitted at
  an optical thickness of 0.1 holds 95.6 % of its plane energy inside the
  240 km domain, against 99.4 % at 0.7. Under grid normalisation the four
  curves converge at the edge and their ordering vanishes exactly where
  the question is asked.

  The plane total is the integral of the fitted profile over the whole
  plane, so it exists only for an analytical kernel carrying its model
  and parameters, and only where that integral converges: a King needs
  `gamma > 1`, a generalised Moffat `gamma * beta > 1`, and a Voigt never
  qualifies, its Lorentzian part integrating as `log r`. Each case raises
  rather than guesses, since a silent fallback would rescale a published
  curve without saying so. `encircled_radius` returns `nan` for a
  fraction the grid never reached.

- **`adjeff.setup_logging()`**, one line to turn logging on:

  ```python
  adjeff.setup_logging(level="info")
  adjeff.setup_logging(level="debug", json=True)
  ```

  `xsweep` follows adjeff's level, because it reports on the same work:
  over one run it emits the point counts, the cache decisions and the
  elapsed time of every sweep, all of which used to be invisible at any
  setting. `zarr`, `numcodecs`, `matplotlib`, `asyncio`, `h5py`,
  `trimesh`, `PIL` and `fsspec` are capped at warning, having produced
  170 of the 205 records measured over that run. `captureWarnings` routes
  Python's warnings into the same stream.

- **Durations.** `SceneModule.forward` brackets its work with
  `module.start` and `module.done`, the second carrying `duration_s` and
  `cached`. A failure emits `module.failed` with the duration and the
  exception type, so a run that dies mid-way says where and how long it
  got. No log in this package carried a duration before.

- **Run context.** `fit` binds a run id, and each optimisation its band
  and combo; `Pipeline` binds the stage. Every line below carries them
  without a caller passing anything down.

- **The three files that had no logging at all** now have some, and they
  are where the hours go. `sweep.plan` announces the states, bands,
  photon count and batch size *before* the first Smart-G call.
  `psf.convolve` names the shape and device before the FFT.
  `landscape.scan` brackets the scan over kernels.

- **The warning level, which was two calls in the whole package.**
  `profile.extrapolated` fires when a radial profile is reconstructed
  well past its last radius, which a Pchip does by continuing the slope
  it ended on. It follows how *far* the extrapolation reaches rather than
  whether it happens: a profile binned from a square grid stops at the
  centre of its outermost annulus, so the four corner pixels always sit
  half a bin beyond, and a warning that fires on the ordinary case
  teaches the reader to ignore it. Below 5 % overshoot it is a debug
  line. `parameter.clamped` fires when an initial value falls outside a
  `ConstrainedParameter`'s bounds, which used to happen in silence.

- **Every line names where it came from.** The logger is rendered
  alongside the message, and `SceneModule.forward` binds the module into
  the context rather than only onto its own logger, so a line raised by a
  helper three frames down carries it too. Those are the lines whose
  origin is hardest to guess.

### Fixed

- **A log line names where it came from, and no longer cries wolf.**
  `profile.extrapolated` carried `stage=2/3` and nothing else: the module
  was bound onto `SceneModule`'s own logger instead of into the context,
  so helpers three frames below inherited nothing. The module now goes
  into the context and the logger name is rendered on every line. The
  level also follows how far the extrapolation reaches rather than
  whether it happens, 5 % overshoot separating debug from warning: the
  reported case was four pixels out of forty thousand reaching 0.36 %
  past the last knot, which is how radial binning works and not a defect.

- **`fit.done` carries the run it closes.** It was emitted outside the
  `run_context` block, so the one line saying a fit had finished had no
  `run_id` to match against its `fit.start`, and no duration on the
  longest operation in the package. Found by auditing what a real fit
  emits: of 117 records, four could not be located and this was the only
  one where that was wrong.

- **A kernel read back from a PSF tree keeps its model.** `_stack` drops
  every attribute when it combines per-combo kernels, rightly for
  `adjeff:params`, which differs between combos, but not for the model
  name, which does not. `psf_kernel` restores the parameters too when the
  tree holds a single combo. Without this, `normalize="plane"` refused
  every kernel that had been through a fit, which is every kernel a
  figure draws.

### Removed

- `MultilineConsoleRenderer`, 72 lines configured nowhere in src, in the
  notebooks or in the article's figures.

### Measured

Same run, before and after:

| | 0.12.0 | 0.13.0 |
| --- | --- | --- |
| unannounced wait, at `info` | 100 % | 0 % |
| unannounced wait, all levels | 95 % | 0 % |
| lines at `info` | 8 | 48 |
| distinct events at `info` | 1 | 6 |
| `xsweep` records visible | 0 of 21 | 21 of 21 |

The wait itself did not shorten and could not: the run is seven blocking
GPU calls of about 1.8 s each, and nothing speaks from inside a CUDA
kernel. What changed is that it is announced and costed before it is
served. The plan's original criterion, time spent in silence, was the
wrong measure and is corrected in `docs/plan-observabilite-0.13.0.md`.

### Upgrading

adjeff prints nothing until `setup_logging()` is called. Anything that
relied on structlog's output appearing by itself, which is what it did
before, needs that one line. The article repository's `makefig` is the
known case.

## [0.12.0]

### Fixed

- **The encircled energy was read at the wrong radius.** `radius_at`
  cumulated the energy through a bin, which is the energy inside that
  bin's *outer edge*, then returned the bin's *centre*, and snapped to
  it rather than interpolating. Against the closed form of the King
  profile the published radii were off by up to 7 % on EE10 % of a
  65-pixel grid and around 1 % on EE50 % and EE90 %; every case is now
  under 0.2 %. `encircled_energy` therefore returns one point per bin
  edge, starting at zero. The `cdf` statistic of the radial profile
  keeps its abscissa and no longer ends at exactly one: on a constant
  field the energy enclosed by the last bin centre is the area ratio,
  and claiming one there was the defect. On the article's 1999-pixel
  grid the curve ends within 1e-4 of one.

- **A non-finite result is no longer cached.** Smart-G returns NaN
  rather than raising when it cannot allocate on the GPU, and the six
  radiative samplers of a run then wrote six NaN entries.  Cached, that
  is permanent: every later run read them back and failed two hundred
  lines away, inside a Pchip interpolation, with nothing pointing at a
  simulation that had run minutes earlier.  A module now checks its
  outputs before writing and raises `ComputationError`, naming the
  module, the variable and the band, and leaving no entry behind.

- **A parameter that overshot its bound stayed dead.**
  `ConstrainedParameter.forward` clamped the raw parameter, which bounds
  the physical value but leaves the raw one outside, where the
  derivative of `clamp` is zero. The gradient died and no later step
  could bring it back, while the value on display stayed perfectly
  plausible. `project()` now puts the raw parameter *on* the boundary
  after every optimiser step, where the transform is still
  differentiable, and both optimisers call it.

- **`api.py`'s docstrings disagreed with its signatures.** Forty-four
  parameters carried a default their type line did not call `optional`.
  `cache` was described eight ways across eight functions and `n_bins`
  four ways across four.

- **A PSF whose profile raises the radius to a power below one made
  every parameter NaN.** Differentiating `(r/sigma)**p` with respect to
  `sigma` brings out `(r/sigma)**(p-1)`, which is `inf` at the origin for
  `p < 1`, times `r/sigma**2`, which is zero: IEEE 754 answers NaN. One
  pixel is enough, the NaN reaching every parameter through the sum of
  the gradient on the next optimiser step. `GeneralizedGaussianPSF` is
  always exposed, its shape exponent being constrained to `[0.1, 0.4]`,
  and `MoffatGeneralizedPSF` whenever `beta < 0.5`.

  Whether it fired was decided by floating-point rounding: `linspace`
  lands exactly on zero for a 401-pixel grid at 0.5 km and misses it by
  5e-08 for a 1999-pixel grid at 0.1 km. The article's figures were
  spared; the smoke run's grid was not, and had been fitting a NaN model
  and drawing it since long before the guard above made it visible.

  `radial_power` evaluates the origin on a stand-in radius and discards
  it, giving the gradient there the value of its limit, zero. Discarding
  the *result* would not have been enough: the gradient of an overwritten
  value is zero, and zero times NaN is still NaN. Every kernel is
  unchanged, bit for bit, on twenty-one cases across three grids.

- Reading a trained parameter into a `float` no longer warns. Ten sites
  did it, and torch is right to complain: that is where a value leaves
  the autograd graph, and doing it by accident inside a training loop is
  a real mistake. `ConstrainedParameter.scalar` says so on purpose.

- The message raised on a non-finite result named a busy GPU as the only
  cause. Auxiliary data Smart-G cannot find produces the same NaN, which
  is what an integration run without `SMARTG_DIR_AUXDATA` actually hits.

### Added

- `ComputationError`, for a result that cannot be used.  It is raised
  before anything reaches the cache, so a bad result fails where it was
  produced rather than becoming a value later runs keep reading.

- **`py.typed`.** The package had no PEP 561 marker, so every consumer
  saw `Any` for every import and none of the `--strict` clean annotations
  reached anyone, including the article repository.

- `ConstrainedParameter.scalar`, the current value as a plain number,
  detached on purpose.

- `PSFModule` documents what its sampling assumes. A discrete convolution
  needs each tap to carry the integral of the profile over one pixel;
  sampling the profile at the pixel's centre stands in for that, within
  0.4 % from the first neighbour outwards and 550 % out at the centre for
  a kernel sharp against the grid. Integrating the central pixels instead
  is measured at under 2 % of a training step, and is deliberately not
  done: the fitted parameters absorb the bias, so a kernel truer to the
  continuous physics is not automatically a better fit to this discrete
  problem. Settling it needs a measurement, and the note says which.

- `RadialBinning`, in `adjeff.utils.radial`: pixels grouped by radius
  once, values reduced many times through `sum`, `mean`, `std`, `cdf`
  and `radius_at`. With `annulus_areas` and `cumulate` beside it, this
  is the single radial binning the package has; `_RadialGrid` and
  `bin_radial` were two, and `mtf` had a third written in numpy.

### Changed

- `ConstrainedParameter` accepts any transform that is finite and
  strictly increasing over the parameter's bounds, which is the property
  its bounds need, instead of a fixed list of two blessed classes. The
  transforms themselves now compose `torch.distributions.transforms`;
  `Transform` is a Protocol and `IdentityTransform`, never constructed
  anywhere, is gone.

- The six radiative samplers share `AtmoSampler`, declaring their photon
  budget and the geometry arguments Smart-G takes as constants instead
  of repeating twenty lines of construction each. Contracts, photon
  counts, static arguments and config tuples are unchanged, object by
  object.

- `run_forward_pipeline` and `run_radiatives_from_scene` state the
  relation between what they take and what they return with one
  constrained `TypeVar` instead of two `@overload` stubs each.

- The image generators share one body. `gaussian_image_dict` and
  `disk_image_dict` were identical for forty-six lines apiece.

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
