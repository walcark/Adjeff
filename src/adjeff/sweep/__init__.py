"""Sweep utilities for iterating over atmospheric parameter combinations.

This module provides the two building blocks used by
:class:`~adjeff.modules.SceneModuleSweep` to drive Smart-G sweeps:

- :class:`SweepBundle` — runs a callable over the Cartesian product of
  scalar parameters, passing vector parameters whole (or in chunks) at
  each step.  The results are stacked into a multi-dimensional
  :class:`xr.DataArray`.

- :class:`UniqueIndex` — collapses a multi-dimensional parameter grid
  (e.g. ``aot(x, y)``, ``rh(x, y)``) to its unique row combinations
  before the sweep, then expands the result back to the original grid.
  This avoids redundant Smart-G calls when many pixels share the same
  atmospheric state.

Typical use is through :class:`~adjeff.modules.SceneModuleSweep`, which
wires both classes together automatically.  Direct use is only needed
for custom sweep loops outside the module pipeline.
"""

from ._dedup import UniqueIndex
from .bundle import SweepBundle

__all__ = ["SweepBundle", "UniqueIndex"]
