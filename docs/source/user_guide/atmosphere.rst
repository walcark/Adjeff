Atmospheric configuration
=========================

Adjeff uses three Pydantic configuration classes to describe the atmospheric
and geometric state.

AtmoConfig
----------

:class:`~adjeff.atmosphere.atmo_config.AtmoConfig` groups optical properties:

- ``aot`` — aerosol optical thickness at 550 nm
- ``h``, ``rh`` — aerosol scale height and relative humidity
- ``href`` — reference altitude

GeoConfig
---------

:class:`~adjeff.atmosphere.geo_config.GeoConfig` groups sun/sensor angles:

- ``sza``, ``vza`` — solar and view zenith angles (degrees)
- ``saa``, ``vaa`` — solar and view azimuth angles (degrees)
- ``sat_height`` — satellite altitude (km)

SpectralConfig
--------------

:class:`~adjeff.atmosphere.spectral_config.SpectralConfig` defines the
spectral channels used in a simulation.

Parametric sweeps
-----------------

:class:`~adjeff.sweep.bundle.ConfigBundle` combines these three configs
into a multi-dimensional parameter space.  Dimensions marked as
``scalar_dims`` are expanded (one Smart-G call per value); those marked as
``vector_dims`` are batched into a single call.

.. code-block:: python

   import numpy as np
   from adjeff.sweep.bundle import ConfigBundle
   from adjeff.atmosphere.atmo_config import AtmoConfig

   bundle = ConfigBundle(
       atmo=AtmoConfig(aot=np.array([0.05, 0.1, 0.3])),
       scalar_dims=["aot"],
   )
