"""xarray utilities for adjeff internal use.

- :class:`ParamBatch` — broadcasts and flattens atmospheric parameter
  DataArrays for a Smart-G batch call, then restores the original
  dimensional structure after the simulation.
- :func:`square_grid`, :func:`grid` — build centered ``(x, y)``
  coordinate objects for 2D scene grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import xarray as xr


@dataclass
class ParamBatch:
    """Flattened atmospheric parameters ready for a Smart-G batch call.

    Built by :meth:`from_dataarrays`.  Exposes the flat parameter dict
    for :func:`~adjeff.atmosphere.create_atmosphere` and an
    :meth:`unstack` method that reconstructs the full dimensional
    structure from Smart-G output — including transparent handling of the
    ``"index"`` rename introduced by deduplication.
    """

    _DEDUP_TMP: ClassVar[str] = "_index_tmp"
    #: Dims whose entries are aligned rather than swept: several variables
    #: vary together along them, so they carry integer positions instead of
    #: their own values.  Using the values would build a duplicate index
    #: (``rh = [50, 50, 50]``) and break the broadcast.
    _POSITIONAL: ClassVar[tuple[str, ...]] = (_DEDUP_TMP, "point")
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
        **arrs : xr.DataArray
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

        # Assign coords so unstack restores actual parameter values. Dims
        # listed in _POSITIONAL carry aligned entries rather than a swept
        # axis, so they get integer positions instead: that is what lets all
        # arrays share identical coords along them and xr.broadcast succeed.
        assigned: dict[str, xr.DataArray] = {}

        for name, da in renamed.items():
            coords = {}

            for d in da.dims:
                if d in cls._POSITIONAL:
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
        res : xr.DataArray
            DataArray with ``dim="index"`` already set (with ``index_coord``
            as coordinate).  May also have leading angular dims like ``"vza"``
            or ``"sza"``.

        Returns
        -------
        xr.DataArray
            Array with ``"index"`` unstacked back to the original parameter
            dimensions. If deduplication introduced a ``_index_tmp`` dimension,
            it is renamed back to ``"index"``.
        """
        res = res.unstack("index")
        if self._DEDUP_TMP in res.dims:
            res = res.rename({self._DEDUP_TMP: "index"})
        return res


def square_grid(n: int, res: float) -> xr.Coordinates:
    """Create centered ``(x, y)`` coordinates for a square grid.

    Parameters
    ----------
    n : int
        Number of pixels per dimension.
    res : float
        Pixel size in coordinate units (km for adjeff scenes).

    Returns
    -------
    xr.Coordinates
        Centered ``x`` and ``y`` coordinate arrays.
    """
    return grid(nx=n, ny=n, res=res)


def grid(nx: int, ny: int, res: float) -> xr.Coordinates:
    """Create centered ``(x, y)`` coordinates for a rectangular grid.

    Parameters
    ----------
    nx : int
        Number of pixels on the ``x`` dimension.
    ny : int
        Number of pixels on the ``y`` dimension.
    res : float
        Pixel size in coordinate units (km for adjeff scenes).

    Returns
    -------
    xr.Coordinates
        Centered ``x`` and ``y`` coordinate arrays.
    """
    halfx = nx * res * 0.5
    halfy = ny * res * 0.5
    x = np.linspace(-halfx + res * 0.5, halfx - res * 0.5, nx)
    y = np.linspace(-halfy + res * 0.5, halfy - res * 0.5, ny)
    coords = xr.Coordinates(dict(x=x, y=y))
    return coords
