"""Radial statistics of a field.

The ``.adjeff`` accessor delegates to these functions.

Functions
---------
    radial_profile
        Azimuthal mean, std, cumulative energy or adaptive sampling
        against radius.
    transect
        Values along a line through the centre at a given azimuth.
    to_field
        2-D field rebuilt from a radial profile.
    resolution
        Pixel size, from the x coordinate spacing.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import xarray as xr

from adjeff.exceptions import AdjeffAccessorError
from adjeff.utils.radial import (
    RadialBinning,
    _profile_to_field,
    _sample_radial_from_cdf,
    cumulate,
    edges_from_centres,
    natural_npix,
    radial_distances,
)

__all__ = ["radial_profile", "resolution", "to_field", "transect"]


def resolution(da: xr.DataArray) -> float:
    """Return the pixel size of *da*, from its x coordinate spacing."""
    x = da.coords["x"].values
    return float(abs(x[1] - x[0]))


def radial_profile(
    da: xr.DataArray,
    stat: Literal["mean", "cdf", "std", "adaptive"] = "mean",
    center: tuple[float, float] | None = None,
    n_bins: int | None = None,
    *,
    normalize: bool = True,
    n: int | None = None,
    max_gap: float | None = None,
    symmetric: bool = False,
) -> xr.DataArray:
    """Return a radial statistic of *da*, on dim ``"r"``.

    Parameters
    ----------
    stat : {"mean", "cdf", "std", "adaptive"}, optional
        Azimuthal mean (default), cumulated energy, azimuthal standard
        deviation, or the mean on *n* radii denser where it varies.
    center : tuple[float, float] or None, optional
        ``(cx, cy)`` origin; the coordinate mean by default.
    n_bins : int or None, optional
        Bin count, the natural one by default; above it, ``"mean"`` and
        ``"cdf"`` are interpolated.
    normalize : bool, optional
        Normalise ``"cdf"`` to 1.
    n, max_gap : optional
        Number of radii, and largest gap between them, of ``"adaptive"``.
    symmetric : bool, optional
        Mirror the profile around ``r = 0``, to draw it across a transect.

    Raises
    ------
    AdjeffAccessorError
        On an unknown *stat*, or ``"adaptive"`` without *n*.
    """
    profile = _radial_profile(
        da,
        stat,
        center,
        n_bins,
        normalize=normalize,
        n=n,
        max_gap=max_gap,
    )
    if not symmetric:
        return profile
    r = profile.coords["r"].values
    values = profile.values
    return xr.DataArray(
        np.concatenate([values[::-1], values]),
        dims=["r"],
        coords={"r": np.concatenate([-r[::-1], r])},
    )


def _radial_profile(
    da: xr.DataArray,
    stat: Literal["mean", "cdf", "std", "adaptive"] = "mean",
    center: tuple[float, float] | None = None,
    n_bins: int | None = None,
    *,
    normalize: bool = True,
    n: int | None = None,
    max_gap: float | None = None,
) -> xr.DataArray:
    """Return :func:`radial_profile` of *da*, unmirrored."""
    if stat == "mean":
        rr_np, vv_np = radial_distances(da, center=center)
        npix = natural_npix(da)
        binning = RadialBinning(torch.from_numpy(rr_np), npix)
        r_np = binning.centres.numpy()
        v_np = binning.mean(torch.from_numpy(vv_np)).numpy()
        if n_bins is not None and n_bins > npix:
            valid = ~np.isnan(v_np)
            r_np_new = np.linspace(r_np[0], r_np[-1], n_bins)
            v_np = np.interp(r_np_new, r_np[valid], v_np[valid])
            r_np = r_np_new
        return xr.DataArray(v_np, dims=["r"], coords={"r": r_np})

    if stat == "cdf":
        mean_profile = _radial_profile(da, "mean", center, n_bins)
        r = torch.from_numpy(mean_profile.coords["r"].values.astype(np.float32))
        f = torch.clamp(
            torch.from_numpy(mean_profile.values.astype(np.float32)),
            min=0.0,
        )
        # The cumulated energy belongs to the outer edge of each
        # annulus, so it is built there and read back on the profile's
        # own abscissa, which keeps the shape the other statistics have.
        edges = edges_from_centres(r)
        at_edges = cumulate(f, edges, normalize=normalize)
        return xr.DataArray(
            np.interp(r.numpy(), edges.numpy(), at_edges.numpy()).astype(np.float32),
            dims=mean_profile.dims,
            coords=mean_profile.coords,
        )

    if stat == "std":
        rr_np, vv_np = radial_distances(da, center=center)
        npix = n_bins if n_bins is not None else natural_npix(da)
        binning = RadialBinning(torch.from_numpy(rr_np), npix)
        std = binning.std(torch.from_numpy(vv_np))
        return xr.DataArray(
            std.numpy(), dims=["r"], coords={"r": binning.centres.numpy()}
        )

    if stat == "adaptive":
        if n is None:
            raise AdjeffAccessorError(
                "stat='adaptive' requires n=<int> (number of samples)."
            )
        if "r" in da.dims:
            profile = da
        else:
            profile = _radial_profile(da, "mean", center=center, n_bins=n_bins)
        r_vals = _sample_radial_from_cdf(profile, n, max_gap=max_gap)
        values = np.interp(r_vals, profile.coords["r"].values, profile.values)
        return xr.DataArray(values, dims=["r"], coords={"r": r_vals})

    raise AdjeffAccessorError(
        f"Unknown stat={stat!r}. Valid options: 'mean', 'cdf', 'std', 'adaptive'."
    )


def transect(
    da: xr.DataArray,
    angle: float,
    center: tuple[float, float] | None = None,
    n_points: int | None = None,
) -> xr.DataArray:
    """Return the values of the 2-D *da* along a line through its centre.

    The line points to *angle* [°, counterclockwise from +x] on its
    positive side; the result is on dim ``"s"``, the signed distance to
    *center* (the coordinate mean by default).  *n_points* defaults to
    one per pixel.

    Raises
    ------
    AdjeffAccessorError
        If *da* is not 2-D.
    """
    from scipy.interpolate import (  # type: ignore[import-untyped]
        RegularGridInterpolator,
    )

    if da.ndim != 2:
        raise AdjeffAccessorError(
            f"transect requires a 2-D (y, x) DataArray; "
            f"got shape {da.shape}. Call .squeeze() first."
        )

    x = da.coords["x"].values
    y = da.coords["y"].values
    cx = float(np.mean(x)) if center is None else float(center[0])
    cy = float(np.mean(y)) if center is None else float(center[1])

    angle_rad = np.deg2rad(angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))

    r_max = min(x[-1] - cx, cx - x[0], y[-1] - cy, cy - y[0])
    if n_points is None:
        n_points = max(3, int(round(2.0 * r_max / resolution(da))) + 1)

    s_vals = np.linspace(-r_max, r_max, n_points)
    xs = cx + s_vals * cos_a
    ys = cy + s_vals * sin_a

    interp = RegularGridInterpolator(
        (y, x),
        da.values,
        method="linear",
        bounds_error=False,
        fill_value=float("nan"),
    )
    sampled = interp(np.column_stack([ys, xs]))
    return xr.DataArray(sampled, dims=["s"], coords={"s": s_vals})


def to_field(da: xr.DataArray, target_ds: xr.Dataset) -> xr.DataArray:
    """Return the 2-D field of the radial profile *da*, on the grid of *target_ds*.

    Pchip-interpolated at each pixel's radius; extra dims are kept.
    """
    x = target_ds.coords["x"].values
    y = target_ds.coords["y"].values
    xx, yy = np.meshgrid(x, y)
    r = da.coords["r"].values

    def _interp(values: np.ndarray) -> np.ndarray:
        return _profile_to_field(r, values, xx, yy)

    result = xr.apply_ufunc(
        _interp,
        da,
        input_core_dims=[["r"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
    )
    return xr.DataArray(result.assign_coords(y=y, x=x))
