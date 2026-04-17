"""Adjeff: a package for modelling of adjacency effects.

The package provides utilities to simulate environment effects and optimize
models for adjacency effects.
"""

from .accessor import AdjeffDataArrayAccessor
from .api import (
    FullConfig,
    config_from_scene,
    load_maja,
    make_full_config,
    make_model,
    optimize_adam_lbfgs,
    run_forward_pipeline,
    run_radiatives_from_scene,
    sample_psf_atm,
)
from .exceptions import (
    AdjeffAccessorError,
    AdjeffError,
    ConfigurationError,
    MissingVariableError,
)

__all__ = [
    "AdjeffDataArrayAccessor",
    "AdjeffAccessorError",
    "AdjeffError",
    "ConfigurationError",
    "FullConfig",
    "MissingVariableError",
    "config_from_scene",
    "load_maja",
    "make_full_config",
    "make_model",
    "optimize_adam_lbfgs",
    "run_forward_pipeline",
    "run_radiatives_from_scene",
    "sample_psf_atm",
]
