r"""Shared utilities for adjeff internals.

**Configuration** (:mod:`._config`)

- :class:`_Config` — Pydantic base for atmospheric/geometric configs.
- :class:`ConfigProtocol` — structural protocol for sweep machinery.
- :func:`to_arr` — ``BeforeValidator`` converting scalars to DataArrays.
- ``Parameter``, ``Module`` — type aliases.

**Caching** (:mod:`.cache_store`)

- :class:`CacheStore` — Zarr content-hash cache for SceneModule outputs.

**Convolution** (:mod:`.convolve`)

- :func:`fft_convolve_2D` — xarray wrapper (extra dims via
  ``apply_ufunc``).
- :func:`fft_convolve_2D_torch` — low-level GPU FFT convolution.

**Radial analysis** (:mod:`.radial`)

- :func:`radial_distances`, :func:`natural_npix`, :func:`bin_radial`.

**PyTorch utilities** (:mod:`.torchutils`)

- :class:`ConstrainedParameter`, :class:`ExpTransform`,
  :class:`SigmoidTransform`.
- :func:`radial_weights`, :func:`radial_mask`.

**xarray utilities** (:mod:`.xrutils`)

- :class:`ParamBatch` — flattens/restores atmospheric parameter arrays.
- :func:`square_grid`, :func:`grid`.

**Smart-G utilities** (:mod:`.smartgutils`)

- :func:`make_sensors`, :func:`compute_optical_depth`,
  :func:`adapt_smartg_output`.

**Logging** (:mod:`.logger`)

- :class:`MultilineConsoleRenderer` — structlog multi-line renderer.
"""

from ._config import ConfigProtocol, Module, Parameter, _Config, to_arr
from .cache_store import CacheStore
from .convolve import fft_convolve_2D, fft_convolve_2D_torch
from .logger import MultilineConsoleRenderer
from .radial import bin_radial, natural_npix, radial_distances
from .smartgutils import (
    adapt_smartg_output,
    compute_optical_depth,
    make_sensors,
)
from .torchutils import (
    ConstrainedParameter,
    ExpTransform,
    SigmoidTransform,
    radial_mask,
    radial_weights,
)
from .xrutils import ParamBatch, grid, square_grid

__all__ = [
    # Configuration
    "_Config",
    "ConfigProtocol",
    "Module",
    "Parameter",
    "to_arr",
    # Caching
    "CacheStore",
    # Convolution
    "fft_convolve_2D",
    "fft_convolve_2D_torch",
    # Radial analysis
    "radial_distances",
    "natural_npix",
    "bin_radial",
    # PyTorch utilities
    "ConstrainedParameter",
    "ExpTransform",
    "SigmoidTransform",
    "radial_weights",
    "radial_mask",
    # xarray utilities
    "ParamBatch",
    "square_grid",
    "grid",
    # Smart-G utilities
    "make_sensors",
    "compute_optical_depth",
    "adapt_smartg_output",
    # Logging
    "MultilineConsoleRenderer",
]
