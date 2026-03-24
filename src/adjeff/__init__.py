"""Adjeff: a package for modelisation of adjacency effects.

The package provides utilities to simulate environment effets and optimize
models for adjacency effects.
"""

from .accessor import AdjeffAccessor
from .exceptions import AdjeffAccessorError, AdjeffError, MissingVariableError

__all__ = [
    "AdjeffAccessor",
    "AdjeffAccessorError",
    "AdjeffError",
    "MissingVariableError",
]
