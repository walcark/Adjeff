"""Operate transformation of xr.Dataset and xr.DataArray objects.

Contains:
---------

- ParamsBatch: used to handle iteration on DataArrays values and restore
back dimensions after calculations.

- grid / square_grid: instanciate square of rectangle (x,y) grid for 2D
image coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import xarray as xr


@dataclass
class ParamBatch:
    """Flattened atmospheric parameters ready for a SmartG batch call.

    Produced by :func:`atmo_flatten`.  Exposes the flat parameter dict for
    :func:`~adjeff.atmosphere.create_atmosphere` and a :meth:`unstack`
    method that reconstructs the full dimensional structure from SmartG
    output — including handling the deduplication ``_index_tmp`` rename
    transparently.
    """

    _DEDUP_TMP: ClassVar[str] = "_index_tmp"
    _flat: dict[str, xr.DataArray]
    _index_coord: xr.DataArray  # MultiIndex coord for unstack

    @classmethod
    def from_dataarrays(cls, **arrs: xr.DataArray) -> Self:
        """Broadcast and flatten DataArrays into a common dimension.

        This common dimension name is ``index``. Handles the deduplication
        case where some args already carry a dim named ``"index"`` produced
        by the bundle's deduplication machinery. In that case the existing
        index is temporarily renamed to avoid a name conflict with the new
        flat index, and the :meth:`AtmoBatch.unstack` method renames it back
        transparently after the computation.

        Parameters
        ----------
        **args:
            Named DataArrays (``wl``, ``aot``, ``rh``, ``h``, etc.). Each must
            be 1-D along its own named dim — or already have been deduplicated
            onto a single ``"index"`` dim by the bundle.
        """
        # Rename any existing "index" dimension
        renamed: dict[str, xr.DataArray] = {}

        for name, da in arrs.items():
            dims_to_rename = {}
            for d in da.dims:
                if d == "index":
                    dims_to_rename[d] = cls._DEDUP_TMP

            renamed[name] = da.rename(dims_to_rename)

        # Assign coords so unstack restores actual parameter values. For
        # _DEDUP_TMP (the dedup index), integer positions are used so that
        # all arrays share identical coords along that dim and xr.broadcast
        # succeeds.
        assigned: dict[str, xr.DataArray] = {}

        for name, da in renamed.items():
            coords = {}

            for d in da.dims:
                if d == cls._DEDUP_TMP:
                    # Int coordinates so all arrays align for broadcast
                    coords[d] = np.arange(da.sizes[d])
                elif d in da.coords:
                    coords[d] = da.coords[d]
                else:
                    coords[d] = da.values

            assigned[name] = da.assign_coords(coords)

        broadcasted = xr.broadcast(*assigned.values())
        dims = broadcasted[0].dims
        stacked = [arr.stack(index=dims) for arr in broadcasted]
        flat_arrs = {name: arr for name, arr in zip(arrs.keys(), stacked)}

        return cls(_flat=flat_arrs, _index_coord=stacked[0]["index"])

    def as_dict(self) -> dict[str, xr.DataArray]:
        """Return flat parameter dict suitable for ``create_atmosphere``."""
        return dict(self._flat)

    @property
    def index_coord(self) -> xr.DataArray:
        """MultiIndex coordinate to use when constructing result DataArrays."""
        return self._index_coord

    def unstack(self, res: xr.DataArray) -> xr.DataArray:
        """Unstack the ``"index"`` dim and restore the dedup index if present.

        Parameters
        ----------
        res:
            DataArray with ``dim="index"`` already set (with ``index_coord``
            as coordinate).  May also have leading angular dims like ``"vza"``
            or ``"sza"``.
        """
        res = res.unstack("index")
        if self._DEDUP_TMP in res.dims:
            res = res.rename({self._DEDUP_TMP: "index"})
        return res


def square_grid(n: int, res: float) -> xr.Coordinates:
    """Create the coordinates for a 2D square grid.

    The grid is assumed to have the same number of pixel for each dimensions
    and is centered on (0, 0).

    Parameters
    ----------
    n : int
        Number of pixels per dimension.
    res : float
        Grid resolution

    Returns
    -------
    xr.Coordinates
        The xarray coordinates associated to the grid.

    """
    return grid(nx=n, ny=n, res=res)


def grid(nx: int, ny: int, res: float) -> xr.Coordinates:
    """Create the coordinates for a 2D rectangular grid.

    The grid can have a different number of pixels on each dimensions and is
    centered on (0, 0).

    Parameters
    ----------
    nx : int
        Number of pixels on the `x` dimension.
    ny : int
        Number of pixels on the `y` dimension.
    res : float
        Grid resolution

    Returns
    -------
    xr.Coordinates
        The xarray coordinates associated to the grid.

    """
    halfx = nx * res * 0.5
    halfy = ny * res * 0.5
    x = np.linspace(-halfx + res * 0.5, halfx - res * 0.5, nx)
    y = np.linspace(-halfy + res * 0.5, halfy - res * 0.5, ny)
    coords = xr.Coordinates(dict(x=x, y=y))
    return coords
