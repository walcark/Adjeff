PSF models
==========

The Point Spread Function (PSF) describes the spatial blurring introduced by
the atmosphere and sensor optics.

PSFGrid and PSFModule
---------------------

:class:`~adjeff.core._psf.PSFGrid` is the base data class that holds a
2-D kernel on a spatial grid.  :class:`~adjeff.core._psf.PSFModule` wraps
it as a :class:`torch.nn.Module` with learnable parameters.

Analytical models
-----------------

All analytical PSF models inherit from :class:`~adjeff.core._psf.PSFModule`
and use :class:`~adjeff.utils.torchutils.ConstrainedParameter` to ensure
physically valid parameter values:

- :class:`~adjeff.core.analytical_psf.GaussPSF` — Gaussian
- :class:`~adjeff.core.analytical_psf.VoigtPSF` — Voigt profile (Gaussian × Lorentzian)
- :class:`~adjeff.core.analytical_psf.KingPSF` — King profile
- :class:`~adjeff.core.analytical_psf.MoffatGeneralizedPSF` — Generalised Moffat

Non-analytical PSF
------------------

:class:`~adjeff.core.non_analytical_psf.NonAnalyticalPSF` holds a fixed
kernel (not optimisable).  Useful to inject a known PSF into the pipeline.

PSFDict
-------

:class:`~adjeff.core.psf_dict.PSFDict` maps
:class:`~adjeff.core.bands.SensorBand` to a PSF module or a pre-computed
:class:`xarray.DataArray`.  It supports extra dimensions (e.g. ``aot``,
``rh``) for atmosphere-dependent PSFs.

Optimisation
------------

See :doc:`/api/optim` and the :doc:`/examples/06-learn-psf` notebook.
