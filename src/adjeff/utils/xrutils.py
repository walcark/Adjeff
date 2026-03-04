"""
Methods to operate transformation of xr.Dataset and xr.DataArray objects.
Also implements method to with LUT / MLUT and the luts package, that is
an older version of xarray used in Smart-G.A

See: https://github.com/hygeos/luts.
"""

from collections.abc import Sequence
from typing import Any, Union
import xarray as xr
import pandas as pd
import numpy as np
import math


def square_grid(n: int, res: float) -> xr.Coordinates:
    """
    Creates the coordinates for a 2D square grid. The grid
    is assumed to have the same number of pixel for each
    dimensions.

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
    """
    Creates the coordinates for a 2D symmetric grid. The grid
    can have a different number of pixels on each dimensions.

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


if __name__ == "__main__":
    g = square_grid(n=3, res=5.0)
    print(g)
    gp = xr.Coordinates(dict(x=[-5.0, 0.0, 5.0], y=[-5.0, 0.0, 5.0]))
    xr.testing.assert_equal(g, gp)
