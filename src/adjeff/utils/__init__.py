r"""Shared utilities for adjeff internals.

Only a handful of these names are meant to be imported from outside the
package (see ``__all__``): the cache, the two convolution entry points,
and the pieces needed to write a custom
:class:`~adjeff.core._psf.PSFModule`.  Everything else is plumbing and
stays reachable through its own submodule, e.g.
``from adjeff.utils.radial import bin_radial``.

**Public** (importable from ``adjeff.utils``)

- :class:`CacheStore`: Zarr content-hash cache for SceneModule outputs.
- :func:`fft_convolve_2D`: xarray wrapper (extra dims via
  ``apply_ufunc``); :func:`fft_convolve_2D_torch`: low-level GPU FFT.
- :class:`ConstrainedParameter`, :class:`ExpTransform`,
  :class:`SigmoidTransform`: constrained parameters for custom PSFs.

**Internal** (import from the submodule)

- :mod:`._config`: :class:`_Config`, :class:`ConfigProtocol`,
  :func:`to_arr`, and the ``Parameter`` / ``Module`` type aliases.
- :mod:`.radial`: :func:`radial_distances`, :func:`natural_npix`,
  :func:`bin_radial`.
- :mod:`.torchutils`: :func:`radial_weights`, :func:`radial_mask`.
- :mod:`.xrutils`: :class:`ParamBatch`, :func:`square_grid`,
  :func:`grid`.
- :mod:`.smartgutils`: :func:`make_sensors`,
  :func:`compute_optical_depth`, :func:`adapt_smartg_output`.
- :mod:`.logger`: :class:`MultilineConsoleRenderer`.
"""

from .cache_store import CacheStore
from .convolve import fft_convolve_2D, fft_convolve_2D_torch
from .torchutils import (
    ConstrainedParameter,
    ExpTransform,
    SigmoidTransform,
)

__all__ = [
    # Caching
    "CacheStore",
    # Convolution
    "fft_convolve_2D",
    "fft_convolve_2D_torch",
    # Building blocks for custom PSF modules
    "ConstrainedParameter",
    "ExpTransform",
    "SigmoidTransform",
]
