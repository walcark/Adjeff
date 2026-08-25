import xarray as xr
import pytest

from adjeff.utils.xrutils import grid, square_grid


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


# ---------------------------------------------------------------------------
# Angle/point pairing in a batched Smart-G call
# ---------------------------------------------------------------------------


def test_pairing_keeps_the_angle_each_point_asked_for():
    """Smart-G returns every angle for every point; keep the diagonal.

    A batched call hands Smart-G one atmosphere and one angle per point,
    but the engine evaluates the full cross product.  Point ``i`` must
    keep angle ``i``; anything else silently mixes two states.
    """
    import numpy as np

    from adjeff.modules.samplers._smartg import _pair_angles_with_points

    cross = xr.DataArray(
        np.arange(9).reshape(3, 3),
        dims=["vza", "point"],
        coords={"vza": [0.0, 30.0, 60.0]},
    )

    paired = _pair_angles_with_points(cross, "vza")

    assert paired.dims == ("point",)
    np.testing.assert_array_equal(paired.values, [0, 4, 8])


def test_pairing_leaves_an_unbatched_call_alone():
    """Outside a batch the angle axis is a genuine sweep axis."""
    import numpy as np

    from adjeff.modules.samplers._smartg import _pair_angles_with_points

    swept = xr.DataArray(
        np.arange(6).reshape(3, 2),
        dims=["vza", "aot"],
        coords={"vza": [0.0, 30.0, 60.0]},
    )

    assert _pair_angles_with_points(swept, "vza").identical(swept)
