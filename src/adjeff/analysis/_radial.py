"""Radial statistics of a field, and the reconstruction that undoes them.

These were methods of the ``.adjeff`` accessor.  They are functions here
so that the accessor stays a thin way of spelling them, and so that the
radial binning has one implementation rather than one per caller.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import xarray as xr

from adjeff.exceptions import AdjeffAccessorError
from adjeff.utils.radial import (
    _profile_to_field,
    _sample_radial_from_cdf,
    bin_radial,
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
    """Radial profile of *da*.

    See :func:`_radial_profile` for the statistics; *symmetric* mirrors
    the result around ``r = 0`` so that it can be drawn across a full
    transect rather than on the positive half only.  The values are
    unchanged: a radial profile is symmetric by construction, this only
    writes the other half down.
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
    """Radial profile of *da*, one statistic at a time.

    Parameters
    ----------
    stat : {"mean", "cdf", "std", "adaptive"}, optional
        Statistic to compute (default ``"mean"``):

        - ``"mean"``     — azimuthal mean per radial bin.
        - ``"cdf"``      — cumulative area-weighted distribution.
        - ``"std"``      — azimuthal standard deviation per bin.
        - ``"adaptive"`` — gradient-driven sparse sampling (requires *n*).

    center : tuple[float, float] or None, optional
        ``(cx, cy)`` origin in coordinate units. Defaults to the
        coordinate mean.
    n_bins : int or None, optional
        Number of radial bins. When ``None``, natural bin count is used.
        For ``stat="mean"`` and ``"cdf"``, values above the natural count
        are upsampled; for ``stat="std"`` it directly sets the bin count.
        Not used for ``stat="adaptive"``.
    normalize : bool, optional
        Normalise the CDF output to ``[0, 1]`` (default ``True``).
        Only relevant for ``stat="cdf"``.
    n : int or None, optional
        Number of gradient-driven sample positions.
        Required when ``stat="adaptive"``.
    max_gap : float or None, optional
        Maximum allowed gap between consecutive adaptive samples.
        Only used when ``stat="adaptive"``.

    Returns
    -------
    xr.DataArray
        1-D DataArray with dim ``"r"``.

    Raises
    ------
    AdjeffAccessorError
        If *stat* is not one of the four recognised options, or if
        ``stat="adaptive"`` is requested without providing *n*.
    """
    if stat == "mean":
        rr_np, vv_np = radial_distances(da, center=center)
        npix = natural_npix(da)
        rr = torch.from_numpy(rr_np)
        vv = torch.from_numpy(vv_np)
        _, _, counts, sum_vals, r_centers = bin_radial(rr, vv, npix)
        val_mean = torch.full((npix,), float("nan"))
        mask = counts > 0
        val_mean[mask] = sum_vals[mask] / counts[mask]
        r_np = r_centers.numpy()
        v_np = val_mean.numpy()
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
        dr = r[1:] - r[:-1]
        edges = torch.empty(r.numel() + 1, dtype=r.dtype)
        edges[1:-1] = 0.5 * (r[:-1] + r[1:])
        edges[0] = r[0] - 0.5 * dr[0]
        edges[-1] = r[-1] + 0.5 * dr[-1]
        area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        cdf = torch.cumsum(f * area, dim=0)
        if normalize and cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        return xr.DataArray(
            cdf.numpy(),
            dims=mean_profile.dims,
            coords=mean_profile.coords,
        )

    if stat == "std":
        rr_np, vv_np = radial_distances(da, center=center)
        npix = n_bins if n_bins is not None else natural_npix(da)
        rr = torch.from_numpy(rr_np)
        vv = torch.from_numpy(vv_np)
        _, inds, counts, sum_vals, r_centers = bin_radial(rr, vv, npix)
        sum_sq = torch.bincount(inds, weights=vv**2, minlength=npix)
        std = torch.full((npix,), float("nan"))
        mask = counts > 0
        mean_v = sum_vals[mask] / counts[mask]
        std[mask] = torch.sqrt(
            (sum_sq[mask] / counts[mask] - mean_v**2).clamp(min=0.0)
        )
        return xr.DataArray(
            std.numpy(), dims=["r"], coords={"r": r_centers.numpy()}
        )

    if stat == "adaptive":
        if n is None:
            raise AdjeffAccessorError(
                "stat='adaptive' requires n=<int> (number of samples)."
            )
        if "r" in da.dims:
            profile = da
        else:
            profile = _radial_profile(
                da, "mean", center=center, n_bins=n_bins
            )
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
    """Sample values along a line through the centre at a given azimuth.

    Parameters
    ----------
    angle : float
        Azimuth angle [°], measured counterclockwise from the +x axis.
        The positive side of the transect points in direction *angle*;
        the negative side points in direction *angle* + 180°.
    center : tuple[float, float] or None, optional
        ``(cx, cy)`` origin in coordinate units.  Defaults to the
        coordinate mean.
    n_points : int or None, optional
        Number of sample points.  Defaults to the number of pixels
        that fit along the transect at the native resolution.

    Returns
    -------
    xr.DataArray
        1-D DataArray with dim ``"s"`` (signed distance from centre,
        same units as the spatial coordinates).

    Raises
    ------
    AdjeffAccessorError
        If the DataArray is not exactly 2-D ``(y, x)``.  Call
        ``.squeeze()`` first when extra dimensions are present.
    """
    from scipy.interpolate import (  # type: ignore[import-untyped]
        RegularGridInterpolator,
    )

    da = da
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
    """Reconstruct a field from a radial profile, by Pchip interpolation.

    Interpolates *da* (dim ``"r"``) at the radial distances of
    every pixel in *target_ds*, broadcasting over all extra dimensions
    (e.g. ``aot``, ``wavelength``).

    Parameters
    ----------
    da : xr.DataArray
        Radial profile, with dim ``"r"``.
    target_ds : xr.Dataset
        Dataset whose ``"x"`` and ``"y"`` coordinates define the output
        grid.

    Returns
    -------
    xr.DataArray
        DataArray with dims ``(..., "y", "x")`` on the target grid.
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
