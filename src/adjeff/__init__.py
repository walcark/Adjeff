"""Adjeff: a package for modelling of adjacency effects.

The package provides utilities to simulate environment effects and optimize
models for adjacency effects.
"""

from ._logging import setup_logging
from .accessor import AdjeffDataArrayAccessor
from .api import (
    FullConfig,
    apply_psf,
    fit_psf,
    load_config,
    load_maja,
    load_scene,
    make_full_config,
    make_model,
    run_forward_pipeline,
    run_radiatives_from_scene,
    sample_psf_atm,
    sample_psf_atm_from_scene,
)
from .exceptions import (
    AdjeffAccessorError,
    AdjeffError,
    ComputationError,
    ConfigurationError,
    ImageIOError,
    MissingVariableError,
    OptimizationWarning,
)
from .optim import fit

__all__ = [
    "setup_logging",
    "AdjeffDataArrayAccessor",
    "AdjeffAccessorError",
    "AdjeffError",
    "ComputationError",
    "ConfigurationError",
    "ImageIOError",
    "OptimizationWarning",
    "FullConfig",
    "MissingVariableError",
    # API — loaders
    "load_scene",
    "load_maja",
    # API — config
    "load_config",
    "make_full_config",
    # API — model
    "make_model",
    "fit",
    "fit_psf",
    "apply_psf",
    # API — pipelines
    "run_forward_pipeline",
    "run_radiatives_from_scene",
    # API — PSF sampling
    "sample_psf_atm",
    "sample_psf_atm_from_scene",
]
