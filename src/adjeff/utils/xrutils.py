"""xarray helpers.

Classes
-------
    ParamBatch
        Atmospheric parameters flattened for one Smart-G call, and restored.

Functions
---------
    square_grid
        Centred ``(x, y)`` coordinates of a square grid.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import numpy as np
import xarray as xr


@dataclass
class ParamBatch:
    """Atmospheric parameters broadcast and flattened onto one ``index`` dim.

    Built by :meth:`from_dataarrays`; :meth:`unstack` restores the dims
    on a Smart-G output.
    """

    _DEDUP_TMP: ClassVar[str] = "_index_tmp"
    #: Dim a batched xsweep call stacks its points along, as
    #: ``xsweep.delivery.GROUP_DIM``.
    GROUP_DIM: ClassVar[str] = "point"
    #: Dims along which arrays vary together: labelled by position, since
    #: their values may repeat and break the broadcast.
    _POSITIONAL: ClassVar[tuple[str, ...]] = (_DEDUP_TMP, GROUP_DIM)
    _flat: dict[str, xr.DataArray]
    _index_coord: xr.DataArray  # MultiIndex coord for unstack

    @classmethod
    def from_dataarrays(cls, **arrs: xr.DataArray) -> Self:
        """Broadcast *arrs* together and stack them onto ``index``.

        An existing ``index`` dim (from deduplication) is set aside and
        restored by :meth:`unstack`.
        """
        # Rename any existing "index" dimension
        renamed: dict[str, xr.DataArray] = {}

        for name, da in arrs.items():
            dims_to_rename = {}
            for d in da.dims:
                if d == "index":
                    dims_to_rename[d] = cls._DEDUP_TMP

            renamed[name] = da.rename(dims_to_rename)

        assigned: dict[str, xr.DataArray] = {}

        for name, da in renamed.items():
            coords: dict[Hashable, Any] = {}

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
        """Unstack ``index`` of *res* back onto the parameter dims."""
        res = res.unstack("index")
        if self._DEDUP_TMP in res.dims:
            res = res.rename({self._DEDUP_TMP: "index"})
        return res


def square_grid(n: int, res: float) -> xr.Coordinates:
    """Return centred coordinates of an ``n × n`` grid of pixel *res*."""
    return grid(nx=n, ny=n, res=res)


def grid(nx: int, ny: int, res: float) -> xr.Coordinates:
    """Create centered ``(x, y)`` coordinates for a rectangular grid."""
    halfx = nx * res * 0.5
    halfy = ny * res * 0.5
    x = np.linspace(-halfx + res * 0.5, halfx - res * 0.5, nx)
    y = np.linspace(-halfy + res * 0.5, halfy - res * 0.5, ny)
    coords = xr.Coordinates(dict(x=x, y=y))
    return coords
