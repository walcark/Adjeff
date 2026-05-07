Quickstart
==========

This page walks through Adjeff's main concepts one at a time, from the
simplest building blocks to a full end-to-end PSF fitting workflow.

Images
------

The central data structure is :class:`~adjeff.core.ImageDict` — a dict that
maps a :class:`~adjeff.core.SensorBand` to an :class:`xarray.Dataset`.
Datasets are progressively enriched as they pass through a pipeline.

Adjeff ships three synthetic image generators that return ready-to-use
:class:`~adjeff.core.ImageDict` objects, no GPU needed:

.. code-block:: python

   from adjeff.core import gaussian_image_dict, disk_image_dict, S2Band

   # Gaussian reflectance profile, 256×256 pixels, 120 m resolution
   image = gaussian_image_dict(
       sigma=1.0,          # Gaussian radius [km]
       res_km=0.12,
       bands=[S2Band.B04],
   )

   # Disk (step function) — typical training scene for PSF fitting
   disk = disk_image_dict(
       radius=5.0,         # disk radius [km]
       res_km=0.12,
       bands=[S2Band.B04],
   )

Each band holds a ``rho_s`` variable (surface reflectance):

.. code-block:: python

   ds = image[S2Band.B04]   # xr.Dataset
   ds["rho_s"]              # DataArray with dims (y, x)

PSF models
----------

A PSF model is an :class:`~adjeff.core._psf.PSFModule` — a
:class:`torch.nn.Module` with learnable parameters.  All analytical models
ensure physically valid values via constrained parameters.

.. code-block:: python

   from adjeff.core import GaussPSF, KingPSF, PSFGrid

   grid = PSFGrid(res=0.12, n=999)   # 999×999 pixels, 120 m

   psf = GaussPSF(grid=grid, sigma=0.5)
   kernel = psf()                     # (999, 999) tensor, sums to 1

The :attr:`~adjeff.accessor.AdjeffDataArrayAccessor.adjeff` accessor gives
direct access to radial analysis:

.. code-block:: python

   da = psf().detach().numpy()
   # radial profile via the xarray accessor
   import xarray as xr
   da_xr = xr.DataArray(da, dims=["y", "x"])
   profile = da_xr.adjeff.radial(res_km=0.12)   # mean per radial bin

:class:`~adjeff.core.PSFDict` maps bands to PSF modules and supports extra
atmospheric dimensions (``aot``, ``rh``) for atmosphere-dependent PSFs:

.. code-block:: python

   from adjeff.core import PSFDict

   psf_dict = PSFDict({S2Band.B04: GaussPSF(grid=grid, sigma=0.5)})

Configuration
-------------

Three Pydantic config objects describe the atmospheric and geometric state.
:func:`~adjeff.make_full_config` is the single entry point:

.. code-block:: python

   import adjeff

   cfg = adjeff.make_full_config(
       bands=[S2Band.B04],
       aot=0.1,            # aerosol optical thickness
       rh=50.0,            # relative humidity [%]
       sza=30.0,           # sun zenith angle [°]
       vza=0.0,            # view zenith angle [°]
   )
   # cfg is a FullConfig TypedDict:
   # {"atmo_config": ..., "geo_config": ..., "spectral_config": ...}

For parametric sweeps over multiple atmospheric states pass lists or
:class:`xarray.DataArray` objects:

.. code-block:: python

   cfg_sweep = adjeff.make_full_config(
       bands=[S2Band.B04],
       aot=[0.05, 0.1, 0.2, 0.3],   # 4 AOT values → swept independently
   )

Forward pipeline (GPU)
----------------------

The forward pipeline runs Smart-G Monte Carlo simulations to compute the six
radiative quantities (transmittances, path radiance, spherical albedo) and
the TOA reflectance:

.. code-block:: python

   import adjeff
   from adjeff.core import disk_image_dict, S2Band

   scene = disk_image_dict(radius=5.0, res_km=0.12, bands=[S2Band.B04])
   cfg   = adjeff.make_full_config(bands=[S2Band.B04], aot=0.1)

   scene = adjeff.run_forward_pipeline(scene, **cfg)
   # scene[S2Band.B04] now contains rho_toa, rho_unif,
   # tdir_down, tdif_down, tdir_up, tdif_up, rho_atm, sph_alb

Pass a list to process multiple scenes with one set of radiative simulations:

.. code-block:: python

   scenes = adjeff.run_forward_pipeline([disk1, disk2, disk3], **cfg)

See :doc:`/examples/03-compute-radiative-quantities` for a full walkthrough.

End-to-end PSF fitting (GPU)
----------------------------

:func:`~adjeff.fit_psf` wraps the entire pipeline — training scenes,
forward simulation, model instantiation, and optimisation — in one call:

.. code-block:: python

   import adjeff
   from adjeff.core import KingPSF, S2Band

   # scene must already contain atmospheric / geometric fields
   # (loaded with load_maja or run_radiatives_from_scene)
   psf_dict = adjeff.fit_psf(
       scene=scene,
       bands=[S2Band.B04],
       psf_type=KingPSF,
       init_parameters={"sigma": 0.3, "gamma": 1.0},
   )
   # psf_dict[S2Band.B04] is a frozen kernel DataArray

Apply the fitted PSF to a new scene:

.. code-block:: python

   result = adjeff.apply_psf(scene, psf_dict, band=S2Band.B04)
   result[S2Band.B04]["rho_s"]   # estimated surface reflectance

See :doc:`/examples/06-learn-psf` for a complete optimisation example.

Going further
-------------

For more control over individual steps:

- :doc:`pipeline` — how ``SceneModule``, ``Pipeline``, and sweeps work
- :doc:`atmosphere` — building configs manually and parametric sweeps
- :doc:`psf` — PSF models, ``PSFDict``, and the optimisation loop
- :ref:`api/index:API Reference` — full API documentation
