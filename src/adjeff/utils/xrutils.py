"""Operate transformation of xr.Dataset and xr.DataArray objects.

Also implements method to with LUT / MLUT and the luts package, that is
an older version of xarray used in Smart-G. See: https://github.com/hygeos/luts.

"""

import numpy as np
import xarray as xr


def square_grid(n: int, res: float) -> xr.Coordinates:
    """Create the coordinates for a 2D square grid.

    The grid is assumed to have the same number of pixel for each dimensions.

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

    The grid can have a different number of pixels on each dimensions.

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
