"""Tests for the Smart-G output collector.

:func:`~adjeff.utils.smartgutils.collect_batched` places every axis of a
Smart-G return by its label.  The idiom it replaces transposed the array
and rebuilt it from ``.values``, which was correct only as long as the
final diagonal pick happened to hide a wrong axis order.

These tests feed synthetic returns whose values encode ``(batch index,
angle)``, so any mispairing produces a wrong number rather than a wrong
shape.  They run without a GPU: no atmosphere is ever built, only the
:class:`~adjeff.utils.xrutils.ParamBatch` the collector needs.
"""

import numpy as np
import pytest
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import ImageDict, S2Band
from adjeff.modules.samplers._atmo_sampler import AtmoSampler
from adjeff.utils.smartgutils import collect_batched
from adjeff.utils.xrutils import ParamBatch

BANDS = [S2Band.B02, S2Band.B03]


# ---------------------------------------------------------------------------
# Synthetic Smart-G returns
# ---------------------------------------------------------------------------


def _batch(*, n_wl: int = 1, n_point: int | None = None, **swept: object) -> ParamBatch:
    """Build a ParamBatch the way a point function does.

    *n_point* stacks the atmospheric parameters along the group dim, as a
    batched sweep hands them over; leaving it ``None`` gives the
    unbatched case, where the parameters carry their own sweep dims.
    """
    wl = xr.DataArray(np.linspace(490.0, 490.0 + 10.0 * n_wl, n_wl), dims=["wl"])
    if n_point is not None:
        defaults: dict[str, xr.DataArray] = {
            name: xr.DataArray(np.full(n_point, value), dims=[ParamBatch.GROUP_DIM])
            for name, value in (("aot", 0.2), ("rh", 50.0), ("h", 0.0), ("href", 2.0))
        }
        defaults["aot"] = xr.DataArray(
            np.linspace(0.05, 0.6, n_point), dims=[ParamBatch.GROUP_DIM]
        )
    else:
        defaults = {
            "aot": xr.DataArray([0.1, 0.4], dims=["aot"]),
            "rh": xr.DataArray(50.0),
            "h": xr.DataArray(0.0),
            "href": xr.DataArray(2.0),
        }
    defaults.update({k: v for k, v in swept.items() if isinstance(v, xr.DataArray)})
    return ParamBatch.from_dataarrays(wl=wl, **defaults)


def _index_positions(batch: ParamBatch) -> xr.DataArray:
    """Return, for each unstacked cell, the flat position it came from."""
    n = len(batch.index_coord)
    return batch.unstack(
        xr.DataArray(np.arange(n), dims=["index"], coords={"index": batch.index_coord})
    )


def _encode(n_index: int, *angle_sizes: int) -> np.ndarray:
    """Build values that identify the ``(index, angle...)`` cell they sit in."""
    grids = np.meshgrid(
        np.arange(n_index), *(np.arange(s) for s in angle_sizes), indexing="ij"
    )
    out = grids[0] * 1000.0
    for depth, grid in enumerate(grids[1:]):
        out = out + grid * 10.0 ** (2 - depth)
    return out


def _as_return(
    values: np.ndarray,
    *dims: str,
    layout: tuple[int, ...] | None = None,
    azimuth: bool = False,
) -> xr.DataArray:
    """Wrap encoded values as Smart-G would return them.

    *layout* permutes the axes: Smart-G's dim order is not guaranteed, and
    a collector placing axes by label must not care.
    """
    res = xr.DataArray(values, dims=list(dims))
    if azimuth:
        res = res.expand_dims({"Azimuth angles": [120.0]})
    if layout is not None:
        res = res.transpose(*[res.dims[i] for i in layout])
    return res


# ---------------------------------------------------------------------------
# One angle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [(0, 1), (1, 0)], ids=["index-first", "angle-first"])
@pytest.mark.parametrize("n_wl", [1, 3], ids=["one-band", "three-bands"])
def test_one_angle_is_placed_by_label_whatever_the_layout(layout, n_wl):
    """The dim order Smart-G happens to return must not change the result.

    With a single wavelength the index and angle axes have the same
    length, and the diagonal pick that ends the collection is invariant
    by transposition: a wrong order stays invisible.  Three wavelengths
    break that coincidence, which is what makes this test worth running.
    """
    n_point = 4
    batch = _batch(n_wl=n_wl, n_point=n_point)
    n_index = len(batch.index_coord)
    vza = np.linspace(0.0, 60.0, n_point)

    raw = _as_return(
        _encode(n_index, n_point), "wavelength", "Zenith angles", layout=layout
    )
    out = collect_batched(raw, batch, angles={"Zenith angles": ("vza", vza)})

    positions = _index_positions(batch)
    expected = (
        positions * 1000.0
        + xr.DataArray(np.arange(n_point), dims=[ParamBatch.GROUP_DIM]) * 100.0
    )
    # The pick leaves the angle behind as a tag on the point axis.
    assert "vza" in out.coords and "vza" not in out.dims
    xr.testing.assert_allclose(
        out.reset_coords(drop=True), expected.transpose(*out.dims)
    )


def test_an_azimuth_axis_of_size_one_is_dropped():
    """Smart-G keeps a singleton azimuth axis the caller never asked for."""
    batch = _batch(n_point=3)
    n_index = len(batch.index_coord)

    raw = _as_return(_encode(n_index, 3), "wavelength", "Zenith angles", azimuth=True)
    out = collect_batched(
        raw,
        batch,
        angles={"Zenith angles": ("vza", np.array([0.0, 30.0, 60.0]))},
        drop=["Azimuth angles"],
    )

    assert "Azimuth angles" not in out.dims


# ---------------------------------------------------------------------------
# Two angles, the rho_atm shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout", [(0, 1, 2), (2, 0, 1), (1, 2, 0)], ids=["abc", "cab", "bca"]
)
def test_two_angles_each_keep_their_own_point(layout):
    """``rho_atm`` pairs both ``vza`` and ``sza`` against the same points."""
    n_point = 3
    batch = _batch(n_wl=2, n_point=n_point)
    n_index = len(batch.index_coord)
    vza = np.linspace(0.0, 60.0, n_point)
    sza = np.linspace(10.0, 50.0, n_point)

    raw = _as_return(
        _encode(n_index, n_point, n_point),
        "wavelength",
        "sensor index",
        "Zenith angles",
        layout=layout,
    )
    out = collect_batched(
        raw,
        batch,
        angles={"sensor index": ("vza", vza), "Zenith angles": ("sza", sza)},
    )

    assert "vza" not in out.dims and "sza" not in out.dims
    pos = np.arange(n_point)
    expected = (
        _index_positions(batch) * 1000.0
        + xr.DataArray(pos, dims=[ParamBatch.GROUP_DIM]) * 100.0
        + xr.DataArray(pos, dims=[ParamBatch.GROUP_DIM]) * 10.0
    )
    xr.testing.assert_allclose(
        out.reset_coords(drop=True), expected.transpose(*out.dims)
    )


# ---------------------------------------------------------------------------
# No angle, and the axis Smart-G collapses away
# ---------------------------------------------------------------------------


def test_no_angle_round_trips_the_parameter_dims():
    """``sph_alb`` returns one value per state and no angle at all."""
    batch = _batch(n_wl=2, n_point=3)
    n_index = len(batch.index_coord)

    raw = xr.DataArray(np.arange(n_index) * 1000.0, dims=["wavelength"])
    out = collect_batched(raw, batch)

    xr.testing.assert_allclose(
        out.reset_coords(drop=True),
        (_index_positions(batch) * 1000.0).transpose(*out.dims),
    )


def test_a_single_state_leaves_smartg_no_axis_to_return():
    """One wavelength and one point collapse the axis; it must be rebuilt."""
    batch = _batch(n_wl=1, n_point=1)

    out = collect_batched(xr.DataArray(np.array(0.42)), batch)

    assert set(out.dims) == {"wl", ParamBatch.GROUP_DIM}
    assert float(out.values.ravel()[0]) == pytest.approx(0.42)


def test_a_collapsed_angle_axis_is_expanded_back():
    """A single angle is returned without its axis, and must regain it."""
    batch = _batch(n_wl=2, n_point=1)
    n_index = len(batch.index_coord)

    raw = xr.DataArray(np.arange(n_index) * 1000.0, dims=["wavelength"])
    out = collect_batched(
        raw, batch, angles={"Zenith angles": ("vza", np.array([30.0]))}
    )

    assert "vza" not in out.dims
    xr.testing.assert_allclose(
        out.reset_coords(drop=True),
        (_index_positions(batch) * 1000.0).transpose(*out.dims),
    )


# ---------------------------------------------------------------------------
# Outside a batch, the angle is a genuine sweep axis
# ---------------------------------------------------------------------------


def test_an_unbatched_call_keeps_the_angle_as_a_sweep_axis():
    """With no point dim there is no diagonal to take.

    This is the case the old positional idiom could corrupt silently: no
    pick hides a wrong axis order, so the layout carried meaning.
    """
    batch = _batch(n_wl=1)  # aot swept on its own dim, no point dim
    n_index = len(batch.index_coord)
    vza = np.array([0.0, 30.0, 60.0])

    raw = _as_return(
        _encode(n_index, len(vza)), "wavelength", "Zenith angles", layout=(1, 0)
    )
    out = collect_batched(raw, batch, angles={"Zenith angles": ("vza", vza)})

    assert "vza" in out.dims
    assert out.sizes["vza"] == len(vza)
    expected = _index_positions(batch) * 1000.0 + xr.DataArray(
        np.arange(len(vza)) * 100.0, dims=["vza"], coords={"vza": vza}
    )
    xr.testing.assert_allclose(
        out.reset_coords(drop=True), expected.transpose(*out.dims)
    )


# ---------------------------------------------------------------------------
# End to end through a sampler, over every shape a user can configure
# ---------------------------------------------------------------------------


def _signal(aot: np.ndarray, vza: np.ndarray) -> np.ndarray:
    """Return a value that identifies the ``(aot, vza)`` pair, and no other."""
    return 100.0 * aot + vza


def _fake_tdif_up(
    wl: xr.DataArray,
    aot: xr.DataArray,
    rh: xr.DataArray,
    h: xr.DataArray,
    href: xr.DataArray,
    vza: xr.DataArray,
    species: dict[str, float],
    afgl_type: str,
    remove_rayleigh: bool,
    n_ph: int,
    saa: float,
) -> xr.DataArray:
    """Stand in for a Smart-G kernel with the same return shape.

    Smart-G evaluates every requested angle for every atmosphere, so the
    return is the full cross product of the flattened states with the
    angles.  The value of a cell is :func:`_signal` of the pair it sits
    on, which makes a wrong pairing a wrong number.
    """
    batch = ParamBatch.from_dataarrays(wl=wl, aot=aot, rh=rh, href=href, h=h)
    angles = np.atleast_1d(np.squeeze(vza.values))
    flat = np.asarray(batch.as_dict()["aot"].values, dtype=float)
    raw = _as_return(
        _signal(flat[:, None], angles[None, :]),
        "wavelength",
        "Zenith angles",
        layout=(1, 0),  # the layout the replaced idiom had to transpose away
    )
    return collect_batched(raw, batch, angles={"Zenith angles": ("vza", angles)})


class _FakeSampler(AtmoSampler):
    """A radiative sampler whose physics is replaced by an identity tag."""

    _output_vars = ["fake"]
    contract = "batch(aot, rh, h, href, vza) vec(wl) -> fake(wl)"
    point_fn = staticmethod(_fake_tdif_up)
    default_n_ph = 1
    geo_statics = ("saa",)


def _build(aot, vza, bands, **kwargs):
    """Return a fake sampler configured the way a user would."""
    return _FakeSampler(
        atmo_config=AtmoConfig(
            aot=aot, rh=50.0, h=0.0, href=2.0, species={"sulphate": 1.0}
        ),
        geo_config=GeoConfig(sza=30.0, vza=vza, saa=120.0, vaa=120.0),
        spectral_config=SpectralConfig.from_bands(bands),
        remove_rayleigh=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "aot",
    [
        0.2,
        [0.1, 0.4],
        np.linspace(0.05, 0.6, 3),
        xr.DataArray([0.1, 0.3], dims=["aot"]),
    ],
    ids=["float", "list", "ndarray", "dataarray"],
)
@pytest.mark.parametrize(
    "vza",
    [10.0, [0.0, 60.0], np.array([0.0, 20.0, 40.0])],
    ids=["float", "list", "ndarray"],
)
@pytest.mark.parametrize("bands", [BANDS[:1], BANDS], ids=["one-band", "two-bands"])
def test_every_state_gets_the_value_of_the_pair_it_asked_for(aot, vza, bands):
    """A float, a list, an array and a DataArray are all legal parameters.

    Each builds a different sweep space, and every combination of it must
    come back holding :func:`_signal` of its own ``(aot, vza)``, not of a
    neighbour's.
    """
    scene = _build(aot, vza, bands)(ImageDict({}))

    for band in bands:
        out = scene[band]["fake"]
        assert np.isfinite(out.values).all()
        expected = _signal(out["aot"], out["vza"]).broadcast_like(out)
        expected = expected.reset_coords(drop=True)
        xr.testing.assert_allclose(
            out.reset_coords(drop=True), expected.transpose(*out.dims)
        )


@pytest.mark.parametrize("bands", [BANDS[:1], BANDS], ids=["one-band", "two-bands"])
def test_a_two_dimensional_parameter_map_keeps_its_grid(bands):
    """An AOT map sweeps ``y`` and ``x`` rather than a named parameter axis."""
    aot_map = xr.DataArray(np.array([[0.1, 0.4], [0.7, 0.2]]), dims=["y", "x"])

    scene = _build(aot_map, [0.0, 60.0], bands)(ImageDict({}))

    for band in bands:
        out = scene[band]["fake"].squeeze(drop=True).transpose("y", "x", "vza")
        expected = _signal(aot_map.values[:, :, None], np.array([0.0, 60.0]))
        np.testing.assert_allclose(out.values, expected)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 64], ids=lambda n: f"batch{n}")
def test_the_batch_size_does_not_move_values_between_points(batch_size):
    """Splitting a sweep into groups must not change a single value."""
    out = _build(
        [0.1, 0.3, 0.5, 0.7], [0.0, 20.0, 40.0, 60.0], BANDS[:1], batch_size=batch_size
    )(ImageDict({}))[BANDS[0]]["fake"]

    expected = _signal(out["aot"], out["vza"]).broadcast_like(out)
    expected = expected.reset_coords(drop=True)
    xr.testing.assert_allclose(
        out.reset_coords(drop=True), expected.transpose(*out.dims)
    )


def test_deduplication_gives_every_repeated_state_the_same_value():
    """Repeated states are computed once and expanded back in place."""
    aot_map = xr.DataArray(np.array([[0.1, 0.4], [0.4, 0.1]]), dims=["y", "x"])

    out = _build(aot_map, 10.0, BANDS[:1], dedup=True)(ImageDict({}))[BANDS[0]]["fake"]

    assert {"y", "x"} <= set(out.dims)
    values = np.asarray(out.squeeze(drop=True).values, dtype=float)
    np.testing.assert_allclose(values, _signal(aot_map.values, 10.0))
