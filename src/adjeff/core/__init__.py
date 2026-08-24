"""Core data structures, sensor bands, and PSF models for adjeff.

**Image representation**

- :class:`ImageDict` — multi-band scene container
  (``dict[SensorBand → xr.Dataset]``).

**Sensor bands**

- :class:`SensorBand` — abstract base for band enumerations.
- :class:`S2Band` — Sentinel-2 spectral bands.

**PSF models**

Analytical (trainable) PSFs inherit from :class:`PSFGrid`:
:class:`GaussPSF`, :class:`VoigtPSF`, :class:`KingPSF`,
:class:`MoffatGeneralizedPSF`, :class:`GeneralizedGaussianPSF`.

Fixed-kernel (non-trainable): :class:`NonAnalyticalPSF`.

Frozen kernels live in an :class:`xarray.DataTree`, one group per band:
:func:`psf_tree`, :func:`freeze`, :func:`psf_kernel`, :func:`psf_params`.
Live, gradient-tracked PSFs are a plain ``dict[SensorBand, PSFModule]``.

**Image generation**

:func:`gaussian_image_dict`, :func:`disk_image_dict`,
:func:`random_image_dict`, :func:`extend_analytical`.
"""

from ._psf import PSFGrid
from .analytical_psf import (
    GaussPSF,
    GeneralizedGaussianPSF,
    KingPSF,
    MoffatGeneralizedPSF,
    VoigtPSF,
)
from .bands import S2Band, SensorBand
from .image_dict import ImageDict
from .image_generator import (
    disk_image_dict,
    extend_analytical,
    gaussian_image_dict,
    random_image_dict,
)
from .non_analytical_psf import NonAnalyticalPSF
from .psf_tree import freeze, psf_kernel, psf_params, psf_tree

__all__ = [
    # Image representation
    "ImageDict",
    # Sensor bands
    "SensorBand",
    "S2Band",
    # PSF models
    "PSFGrid",
    "GaussPSF",
    "VoigtPSF",
    "KingPSF",
    "MoffatGeneralizedPSF",
    "GeneralizedGaussianPSF",
    "NonAnalyticalPSF",
    "psf_tree",
    "freeze",
    "psf_kernel",
    "psf_params",
    # Image generation
    "disk_image_dict",
    "gaussian_image_dict",
    "random_image_dict",
    "extend_analytical",
]
