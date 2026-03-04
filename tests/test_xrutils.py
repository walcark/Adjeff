import xarray as xr
import pytest

from adjeff.utils import square_grid, grid


@pytest.mark.parametrize(
    "n,res,x",
    [
        (5, 5.5, [-11.0, -5.5, 0.0, 5.5, 11.0]),
        (4, 5.5, [-8.25, -2.75, 2.75, 8.25]),
        (7, 0.120, [-0.36, -0.24, -0.12, 0.0, 0.12, 0.24, 0.36]),
        (4, 0.120, [-0.18, -0.06, 0.06, 0.18]),
    ],
)
def test_grid_coordinates(n, res, x):
    g = square_grid(n=n, res=res)
    g_test = xr.Coordinates(dict(x=x, y=x))
    xr.testing.assert_allclose(g, g_test, rtol=1e-5)
