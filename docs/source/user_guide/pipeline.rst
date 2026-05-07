Pipeline
========

Adjeff is built around a chain of :class:`~adjeff.modules.scene_module.SceneModule`
objects that transform :class:`~adjeff.core.image_dict.ImageDict` datasets.

SceneModule
-----------

:class:`~adjeff.modules.scene_module.SceneModule` is a plain Python class
(not a :class:`torch.nn.Module`).  Every transformation subclass declares
two class-level contracts:

- ``required_vars`` — variables that must be present in every band Dataset
  of the input :class:`~adjeff.core.image_dict.ImageDict`.
- ``output_vars`` — variables written by this module into the output scene.

``forward()`` validates inputs, checks an optional Zarr cache, delegates to
``_compute()``, stamps provenance metadata, and saves results.

.. note::

   :class:`~adjeff.modules.scene_module.TrainableSceneModule` is a separate
   subclass that *does* inherit from both ``SceneModule`` and
   :class:`torch.nn.Module`.  It is used for differentiable PSF modules
   that participate in gradient-based optimisation.

Pipeline
--------

:class:`~adjeff.modules.pipeline.Pipeline` chains multiple
:class:`~adjeff.modules.scene_module.SceneModule` objects and validates
dependency edges at construction time: every ``required_vars`` must be
satisfied by a prior module or the initial image.

.. code-block:: python

   from adjeff.modules.pipeline import Pipeline

   pipeline = Pipeline([module_a, module_b, module_c])
   result = pipeline(image)

SceneModuleSweep and dimension expansion
-----------------------------------------

:class:`~adjeff.modules.scene_module_sweep.SceneModuleSweep` extends
:class:`~adjeff.modules.scene_module.SceneModule` for modules that must
be called over a space of atmospheric parameters.  The key behaviour:
**output DataArrays carry the swept parameter values as named
coordinates**, so the scene remembers which atmospheric state produced
which result, and downstream modules broadcast over those extra
dimensions automatically via xarray semantics.

Two kinds of swept dimensions are distinguished:

- ``scalar_dims`` — iterated one-by-one as a Cartesian product (one
  Smart-G call per combination, e.g. one call per ``(sza, vza)`` pair).
- ``vector_dims`` — passed whole to the physics function in a single
  batched call (e.g. all wavelengths at once).

The three examples below cover the three regimes you will encounter.

**Example 1 — pure atmospheric quantity, no spatial dependency**

``tdir_down`` (direct downward transmittance) depends only on the
atmosphere, not on the surface.  Its sampler declares
``required_vars = []`` and puts everything in ``vector_dims``, so a
single Smart-G call produces the result for all parameter combinations:

.. code-block:: python

   import adjeff
   from adjeff.core import S2Band

   cfg = adjeff.make_full_config(
       bands=[S2Band.B04],
       aot=[0.05, 0.1, 0.3],
       rh=[50.0, 80.0],
   )
   scene = adjeff.run_forward_pipeline(disk, **cfg)

   da = scene[S2Band.B04]["tdir_down"]
   da.dims   # ("aot", "rh")  — no x, no y
   da.coords["aot"].values   # [0.05, 0.1, 0.3]

**Example 2 — spatial atmospheric maps and** ``deduplicate_dims``

When ``aot`` and ``h`` come from a per-pixel MAJA product they are
full 2-D arrays.  Without deduplication, Smart-G would be called once
per pixel (potentially hundreds of thousands of calls).
``deduplicate_dims`` collapses the grid to its unique
``(aot, h)`` rows, runs only as many calls as there are distinct
combinations, then re-expands the result to the original spatial grid:

.. code-block:: python

   # scene loaded from MAJA: aot.shape == (y=512, x=512)
   scene = adjeff.run_radiatives_from_scene(
       scene,
       n_bins=10,                    # quantise to ≤10 unique values
       deduplicate_dims=["x", "y"],  # collapse spatial dims
   )

   da = scene[S2Band.B04]["tdir_down"]
   da.dims   # ("y", "x")  — spatial grid restored, no aot/rh coords
              # because the sweep was absorbed by deduplication

**Example 3 — spatial quantity swept over atmospheric parameters**

``rho_toa`` (TOA reflectance) depends on the surface reflectance map
``rho_s(y, x)`` *and* on the atmosphere.  The sampler declares
``required_vars = ["rho_s"]`` and sweeps over ``sza``, ``vza``
(``scalar_dims``) and ``aot``, ``rh`` (``vector_dims``).  The output
therefore combines spatial and atmospheric dimensions:

.. code-block:: python

   cfg = adjeff.make_full_config(
       bands=[S2Band.B04],
       aot=[0.05, 0.1, 0.3],
       rh=[50.0, 80.0],
   )
   scene = adjeff.run_forward_pipeline(disk, **cfg)

   da = scene[S2Band.B04]["rho_toa"]
   da.dims   # ("aot", "rh", "y", "x")

   # Slice a single atmospheric condition:
   da.sel(aot=0.1, rh=50.0)   # shape (y, x)
