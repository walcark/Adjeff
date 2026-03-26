"""Operate transformation of xr.Dataset and xr.DataArray objects.

Also implements method to with LUT / MLUT and the luts package, that is
an older version of xarray used in Smart-G. See: https://github.com/hygeos/luts.

"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _normalize_da(
    name: str,
    da: xr.DataArray,
    deduplicate_dims: list[str] | None,
) -> xr.DataArray:
    """Ensure *da* has a coordinate along its dim and is at least 1-D.

    Scalar DataArrays are promoted to 1-D.  1-D DataArrays without a
    coordinate on their sweep dimension get one assigned from their values.
    Multi-dimensional DataArrays are only valid when *deduplicate_dims* is set.

    Raises
    ------
    ValueError
        If *da* has more than one dimension and *deduplicate_dims* is ``None``.
    """
    if da.ndim == 0:
        v = float(da)
        return xr.DataArray([v], dims=[name], coords={name: [v]})
    if da.ndim == 1 and da.dims[0] == name and name not in da.coords:
        return da.assign_coords({name: da.values})
    if da.ndim > 1 and deduplicate_dims is None:
        raise ValueError(
            f"Parameter '{name}' has {da.ndim} dimensions "
            f"{list(da.dims)}. Multi-dimensional config "
            "parameters require deduplicate_dims to be set."
        )
    return da


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
