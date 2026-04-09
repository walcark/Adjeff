"""Define all utilities not specific to the project core implementations.

Those utilities are generally writen to simplify usage of third-party packages.
"""

from ._config import ConfigProtocol, Module, Parameter, _Config, to_arr
from .cache_store import CacheStore
from .config_bundle import ConfigBundle
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
)
from .xrutils import ParamBatch, grid, square_grid

__all__ = [
    "ConfigProtocol",
    "Module",
    "Parameter",
    "_Config",
    "to_arr",
    "CacheStore",
    "ConfigBundle",
    "fft_convolve_2D",
    "fft_convolve_2D_torch",
    "MultilineConsoleRenderer",
    "ConstrainedParameter",
    "ExpTransform",
    "SigmoidTransform",
    "radial_mask",
    "square_grid",
    "grid",
    "radial_distances",
    "natural_npix",
    "bin_radial",
    "make_sensors",
    "compute_optical_depth",
    "adapt_smartg_output",
    "ParamBatch",
]
