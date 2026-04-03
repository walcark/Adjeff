"""Adjeff: a package for modelling of adjacency effects.

The package provides utilities to simulate environment effects and optimize
models for adjacency effects.
"""

from .accessor import AdjeffDataArrayAccessor
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
    "MissingVariableError",
]
