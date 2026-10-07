"""Shared utilities.

Only the names below are public; the rest is reached through its
submodule, e.g. ``from adjeff.utils.radial import RadialBinning``.

Classes
-------
    CacheStore
        Zarr cache of SceneModule outputs.
    ConstrainedParameter, ExpTransform, SigmoidTransform
        Bounded trainable parameters, for custom PSFs.

Functions
---------
    fft_convolve_2D
        FFT convolution of DataArrays, over any extra dims.
    fft_convolve_2D_torch
        FFT convolution of 2-D tensors.
"""

from .cache_store import CacheStore
from .convolve import fft_convolve_2D, fft_convolve_2D_torch
from .torchutils import (
    ConstrainedParameter,
    ExpTransform,
    SigmoidTransform,
)

__all__ = [
    "CacheStore",
    "fft_convolve_2D",
    "fft_convolve_2D_torch",
    "ConstrainedParameter",
    "ExpTransform",
    "SigmoidTransform",
]
