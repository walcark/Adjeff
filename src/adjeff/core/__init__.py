"""Scenes, sensor bands and PSF models.

Classes
-------
    ImageDict
        One ``xr.Dataset`` per sensor band.
    SensorBand, S2Band
        Band enumerations: generic base and Sentinel-2.
    PSFGrid
        Square grid a PSF is sampled on.
    GaussPSF, GeneralizedGaussianPSF, VoigtPSF, KingPSF, MoffatGeneralizedPSF
        Trainable analytical PSFs.
    NonAnalyticalPSF
        Fixed, non-trainable kernel.

Functions
---------
    psf_tree
        Frozen PSFs as an ``xr.DataTree``, one group per band.
    freeze
        Current kernels of live PSF modules, as a tree.
    psf_kernel, psf_params
        Kernel and fitted parameters of one band in a tree.
    gaussian_image_dict, disk_image_dict, random_image_dict
        Synthetic scenes.
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
]
