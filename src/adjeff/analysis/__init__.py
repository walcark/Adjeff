"""Quantify a PSF, or a two-dimensional field of adjacency effect.

The functions here were scattered: the radial profile lived in the
``.adjeff`` accessor, the encircled energy was written a second time in
``optim/landscape.py`` for speed, and the error metrics on DataArrays
lived in the article's own repository because the package had none.

One rule decides what belongs here:

    does it quantify a PSF, or a 2-D field of adjacency effect?

A generic 1-D FFT, a power spectral density, a Wiener filter do not.
``scipy.signal`` exists.
"""

from ._energy import (
    encircled_energy,
    encircled_radii,
    encircled_radius,
    fwhm,
    mtf,
)
from ._error import bias, mae, rmse
from ._radial import radial_profile, resolution, to_field, transect

__all__ = [
    "bias",
    "encircled_energy",
    "encircled_radii",
    "encircled_radius",
    "fwhm",
    "mae",
    "mtf",
    "rmse",
    "radial_profile",
    "resolution",
    "to_field",
    "transect",
]
