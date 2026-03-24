"""Radial-analysis helper functions for 2D image arrays.

Provides three low-level building blocks used by the adjeff accessor:

1) radial_distances : compute flat (r, v) arrays from a Dataset variable.
2) natural_npix     : maximum bin count that guarantees no empty radial bins.
3) bin_radial       : bin (r, v) torch tensors into radial histograms.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import xarray as xr


def radial_distances(
    ds: xr.Dataset,
    var: str,
    center: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat float32 radial-distance and value arrays for *var*.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing *var*.
    var : str
        Variable name.
    center : tuple[float, float] or None
        ``(cx, cy)`` origin. Defaults to the coordinate mean.

    Returns
    -------
    rr : np.ndarray
        Flat float32 array of radial distances, shape ``(n_pixels,)``.
    vv : np.ndarray
        Flat float32 array of pixel values, shape ``(n_pixels,)``.
    """
    da = ds[var]
    if "x_psf" in da.dims and "y_psf" in da.dims:
        cx, cy = (0.0, 0.0) if center is None else center
        x = da.coords["x_psf"].values
        y = da.coords["y_psf"].values
    else:
        x_dim, y_dim = da.dims[-1], da.dims[-2]
        cx = float(da.coords[x_dim].mean()) if center is None else center[0]
        cy = float(da.coords[y_dim].mean()) if center is None else center[1]
        x = da.coords[x_dim].values
        y = da.coords[y_dim].values

    XX, YY = np.meshgrid(x - cx, y - cy)
    rr = np.sqrt(XX**2 + YY**2).astype(np.float32).ravel()
    vv = da.values.astype(np.float32).ravel()
    return rr, vv


def natural_npix(ds: xr.Dataset, var: str) -> int:
    """Return the maximum bin count that keeps bin width >= 1 pixel.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing *var*.
    var : str
        Variable name.

    Returns
    -------
    int
        Maximum number of radial bins with no empty bins guaranteed.
    """
    da = ds[var]
    side = min(da.shape[-2], da.shape[-1])
    return max(int((side - 1) / math.sqrt(2)) - 1, 2)


def bin_radial(
    rr: "torch.Tensor",
    vv: "torch.Tensor",
    npix: int,
) -> tuple[torch.Tensor, ...]:
    """Bin *rr*/*vv* into *npix* radial bins.

    Parameters
    ----------
    rr : torch.Tensor
        Flat float32 radial-distance array, shape ``(n_pixels,)``.
    vv : torch.Tensor
        Flat float32 value array, shape ``(n_pixels,)``.
    npix : int
        Number of radial bins.

    Returns
    -------
    bins : torch.Tensor
        Bin edges, shape ``(npix + 1,)``.
    inds : torch.Tensor
        Per-pixel bin index, shape ``(n_pixels,)``.
    counts : torch.Tensor
        Number of pixels per bin, shape ``(npix,)``.
    sum_vals : torch.Tensor
        Sum of values per bin, shape ``(npix,)``.
    r_centers : torch.Tensor
        Bin centre radii, shape ``(npix,)``, with ``r_centers[0] == 0``.
    """
    import torch

    bins = torch.linspace(0.0, rr.max(), npix + 1)
    inds = torch.bucketize(rr, bins, right=False) - 1
    inds = inds.clamp(0, npix - 1)

    counts = torch.bincount(inds, minlength=npix).float()
    sum_vals = torch.bincount(inds, weights=vv, minlength=npix)

    r_centers = (0.5 * (bins[:-1] + bins[1:])).clone()
    r_centers[0] = 0.0

    return bins, inds, counts, sum_vals, r_centers
