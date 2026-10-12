"""Run a two-dimensional reduction over the dimensions it does not read.

A reduction of a field to a profile is written on one ``(y, x)`` slice:
`radial_profile`, `encircled_energy`, `mtf` and `transect` all are.  A
kernel fitted over several atmospheric states carries those states as
extra dimensions, and xarray's own convention is that a reduction keeps
them, so that ``profile.plot(hue="aot")`` draws one line per state.

`over_slices` is that convention, applied once rather than inside each
reduction.  The slice implementations are untouched: they keep receiving
a 2-D array with its coordinates, which is what they need to build radii
from.

Functions
---------
    spatial_dims
        The two dimensions a field is reduced over.
    over_slices
        Map a slice reduction over every other dimension.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import xarray as xr


def spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    """Return the ``(y, x)`` dimensions of *da*, kernel grid or field.

    Parameters
    ----------
    da : xr.DataArray
        A kernel, on ``y_psf``/``x_psf``, or a field on its own axes.

    Returns
    -------
    tuple of str
        The two dimension names, outer first; every dimension of an
        input that has fewer than two, which `reduce` then takes whole.
    """
    if "x_psf" in da.dims and "y_psf" in da.dims:
        return "y_psf", "x_psf"
    if da.ndim < 2:
        # `radial_profile("adaptive")` accepts a profile it already read.
        return tuple(str(d) for d in da.dims)  # type: ignore[return-value]
    return str(da.dims[-2]), str(da.dims[-1])


def over_slices(
    reduce: Callable[..., xr.DataArray],
    da: xr.DataArray,
    *args: Any,
    **kwargs: Any,
) -> xr.DataArray:
    """Return *reduce* applied to every ``(y, x)`` slice of *da*.

    A 2-D input is handed straight to *reduce*.  Anything wider is
    sliced along one extra dimension at a time, so the result carries
    them all back with their coordinates.

    Parameters
    ----------
    reduce : callable
        Takes a 2-D DataArray and returns a 1-D one.
    da : xr.DataArray
        The field or kernel to reduce.
    *args, **kwargs
        Forwarded to *reduce*.

    Returns
    -------
    xr.DataArray
        The reduced axis, plus every dimension *reduce* does not read.
    """
    extra = [d for d in da.dims if d not in spatial_dims(da)]
    if not extra:
        return reduce(da, *args, **kwargs)

    # One dimension at a time, rather than stacking them all: `concat`
    # rebuilds a plain index, so a stacked MultiIndex cannot be unstacked
    # back afterwards.
    dim = extra[0]
    slices = [
        over_slices(reduce, da.isel({dim: i}), *args, **kwargs)
        for i in range(da.sizes[dim])
    ]
    return xr.concat(slices, dim=da.coords[dim] if dim in da.coords else dim)
