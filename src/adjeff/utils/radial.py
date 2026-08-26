"""Radial-analysis helper functions for 2D image arrays.

Provides the low-level building blocks the radial analysis rests on:

1) radial_distances : compute flat (r, v) arrays from a Dataset variable.
2) natural_npix     : maximum bin count that guarantees no empty radial bins.
3) RadialBinning    : group pixels by radius once, reduce values many times.
4) annulus_areas    : the area each bin stands for, and cumulate over them,
                      always read at the bin edges rather than at its centre.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import xarray as xr
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]


def _sample_radial_from_cdf(
    profile: xr.DataArray,
    n: int,
    max_gap: float | None = None,
) -> np.ndarray:
    """Return radii sampled adaptively via inverse-CDF of a radial profile.

    Generates *n* points concentrated where ``|df/dr|`` is large (pure
    gradient-based sampling, no DC bias).  If *max_gap* is provided, extra
    uniform points are inserted in any interval that exceeds that distance,
    guaranteeing a minimum spatial coverage in flat regions.

    Parameters
    ----------
    profile : xr.DataArray
        1-D DataArray with dim ``"r"`` (e.g. output of ``.adjeff.radial()``).
    n : int
        Number of gradient-driven sample radii.
    max_gap : float or None, optional
        Maximum allowed distance between two consecutive samples, in the same
        units as ``profile.coords["r"]``.  When ``None`` no gap constraint is
        applied.

    Returns
    -------
    np.ndarray
        Sorted array of radii (length >= *n* when *max_gap* is active).
    """
    r = profile.coords["r"].values.astype(np.float64)
    v = profile.values.astype(np.float64)

    density = np.abs(np.gradient(v, r))

    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(r))]
    )
    total = cdf[-1]
    if total <= 0.0:
        return np.linspace(r[0], r[-1], n)
    cdf /= total

    r_vals = np.interp(np.linspace(0.0, 1.0, n), cdf, r)

    if max_gap is not None:
        filled: list[float] = [r_vals[0]]
        for a, b in zip(r_vals[:-1], r_vals[1:]):
            if b - a > max_gap:
                n_fill = int(np.ceil((b - a) / max_gap)) - 1
                filled.extend(np.linspace(a, b, n_fill + 2)[1:-1].tolist())
            filled.append(b)
        r_vals = np.array(filled)

    return np.asarray(r_vals, dtype=np.float32)


def _profile_to_field(
    r: np.ndarray,
    values: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """Reconstruct 2-D field from a 1-D radial profile via Pchip interpolation.

    Parameters
    ----------
    r : np.ndarray
        Sorted 1-D radii of the profile, shape ``(N,)``.
    values : np.ndarray
        Profile values at each radius, shape ``(N,)``.
    xx : np.ndarray
        2-D x-coordinate grid, shape ``(ny, nx)``.
    yy : np.ndarray
        2-D y-coordinate grid, shape ``(ny, nx)``.

    Returns
    -------
    np.ndarray
        Reconstructed field, shape ``(ny, nx)``.
    """
    rr = np.sqrt(xx**2 + yy**2)

    # Deduplicate radii (keep first occurrence)
    _, unique_idx = np.unique(r, return_index=True)
    r_u = r[unique_idx]
    v_u = values[unique_idx]

    result: np.ndarray = PchipInterpolator(r_u, v_u, extrapolate=True)(rr).astype(
        np.float32
    )
    return result


def radial_distances(
    source: xr.DataArray | xr.Dataset,
    var_name: str | None = None,
    *,
    center: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat float32 radial-distance and value arrays.

    Parameters
    ----------
    source : xr.DataArray or xr.Dataset
        Input array or Dataset. When a Dataset is provided, *var_name* is
        required to select the variable.
    var_name : str or None, optional
        Variable name to extract from *source* when it is a Dataset.
    center : tuple[float, float] or None, optional
        ``(cx, cy)`` origin. Defaults to the coordinate mean.

    Returns
    -------
    rr : np.ndarray
        Flat float32 array of radial distances, shape ``(n_pixels,)``.
    vv : np.ndarray
        Flat float32 array of pixel values, shape ``(n_pixels,)``.
    """
    if isinstance(source, xr.Dataset):
        if var_name is None:
            raise ValueError("var_name is required when source is an xr.Dataset")
        da = source[var_name]
    else:
        da = source
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


def natural_npix(
    source: xr.DataArray | xr.Dataset,
    var_name: str | None = None,
) -> int:
    """Return the maximum bin count that keeps bin width >= 1 pixel.

    Parameters
    ----------
    source : xr.DataArray or xr.Dataset
        Input array or Dataset. When a Dataset is provided, *var_name* is
        required to select the variable.
    var_name : str or None, optional
        Variable name to extract from *source* when it is a Dataset.

    Returns
    -------
    int
        Maximum number of radial bins with no empty bins guaranteed.
    """
    if isinstance(source, xr.Dataset):
        if var_name is None:
            raise ValueError("var_name is required when source is an xr.Dataset")
        da = source[var_name]
    else:
        da = source
    return _natural_bins(min(da.shape[-2], da.shape[-1]))


def _natural_bins(side: int) -> int:
    """Return the bin count that keeps every bin at least one pixel wide."""
    return max(int((side - 1) / math.sqrt(2)) - 1, 2)


def edges_from_centres(centres: "torch.Tensor") -> "torch.Tensor":
    """Return the bin edges a set of bin *centres* implies.

    Edges sit halfway between consecutive centres; the outer two are
    extrapolated by half a step and the innermost is clamped at zero,
    since a radius cannot be negative.

    Parameters
    ----------
    centres : torch.Tensor
        Bin centre radii, shape ``(n_bins,)``, increasing.

    Returns
    -------
    torch.Tensor
        Edges, shape ``(n_bins + 1,)``.
    """
    import torch

    step = centres[1:] - centres[:-1]
    edges = torch.empty(centres.numel() + 1, dtype=centres.dtype)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = (centres[0] - 0.5 * step[0]).clamp(min=0.0)
    edges[-1] = centres[-1] + 0.5 * step[-1]
    return edges


def annulus_areas(edges: "torch.Tensor") -> "torch.Tensor":
    """Return the area of the annulus each bin covers.

    Parameters
    ----------
    edges : torch.Tensor
        Bin edges, shape ``(n_bins + 1,)``.

    Returns
    -------
    torch.Tensor
        Annulus areas, shape ``(n_bins,)``.
    """
    return math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


class RadialBinning:
    """Pixels grouped by their distance to a centre, computed once.

    The pixel to bin mapping and the bin counts depend on the geometry
    alone, not on the values, so a caller that reduces many fields on one
    grid (a landscape scan, a profile and its standard deviation) builds
    this once and calls the reductions repeatedly.

    Parameters
    ----------
    radius : torch.Tensor
        Flat distances, shape ``(n_pixels,)``.  Any unit; the radii that
        come back out are in that same unit.
    n_bins : int
        Number of bins.
    r_max : float or None, optional
        Distance the outermost edge sits at.  Defaults to the largest
        distance in *radius*.  Pixels beyond it fall into the last bin,
        so pass only distances inside *r_max* when that would skew it.

    Attributes
    ----------
    edges : torch.Tensor
        Bin edges, shape ``(n_bins + 1,)``.
    midpoints : torch.Tensor
        True bin midpoints, shape ``(n_bins,)``.
    centres : torch.Tensor
        Midpoints with the first one pulled to zero, shape ``(n_bins,)``.
        A radial profile starts at the centre pixel, so this is the
        abscissa to plot and interpolate against.
    counts : torch.Tensor
        Number of pixels per bin, shape ``(n_bins,)``.
    filled : torch.Tensor
        Boolean mask of the bins that got at least one pixel.
    """

    def __init__(
        self,
        radius: "torch.Tensor",
        n_bins: int,
        *,
        r_max: float | None = None,
    ) -> None:
        import torch

        self.n_bins = n_bins
        top = float(radius.max()) if r_max is None else float(r_max)
        self.edges = torch.linspace(0.0, top, n_bins + 1)
        self.index = (torch.bucketize(radius, self.edges, right=False) - 1).clamp(
            0, n_bins - 1
        )
        self.counts = torch.bincount(self.index, minlength=n_bins).float()
        self.filled = self.counts > 0

        self.midpoints = 0.5 * (self.edges[:-1] + self.edges[1:])
        centres = self.midpoints.clone()
        centres[0] = 0.0
        self.centres = centres
        self.area = annulus_areas(self.edges)

    @classmethod
    def on_square_grid(cls, n: int, res: float) -> "RadialBinning":
        """Return the binning of a square grid centred on itself.

        Parameters
        ----------
        n : int
            Grid side in pixels.
        res : float
            Pixel size, in the unit the radii come out in.

        Returns
        -------
        RadialBinning
            Binning with the natural bin count for that side.
        """
        import torch

        half = (n // 2) * res
        coords = np.linspace(-half, half, n, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords)
        radius = torch.from_numpy(np.hypot(xx, yy).ravel())
        return cls(radius, _natural_bins(n))

    def sum(self, values: "torch.Tensor") -> "torch.Tensor":
        """Return the sum of *values* per bin, shape ``(n_bins,)``."""
        import torch

        return torch.bincount(self.index, weights=values.ravel(), minlength=self.n_bins)

    def mean(
        self, values: "torch.Tensor", fill: float = float("nan")
    ) -> "torch.Tensor":
        """Return the azimuthal mean of *values* per bin.

        Parameters
        ----------
        values : torch.Tensor
            Values to average, one per pixel.
        fill : float, optional
            Value written where a bin caught no pixel.  Defaults to
            ``nan``; pass ``0.0`` when the result feeds an integral.

        Returns
        -------
        torch.Tensor
            Shape ``(n_bins,)``.
        """
        import torch

        total = self.sum(values)
        out = torch.full((self.n_bins,), fill, dtype=torch.float32)
        out[self.filled] = total[self.filled] / self.counts[self.filled]
        return out

    def std(self, values: "torch.Tensor", fill: float = float("nan")) -> "torch.Tensor":
        """Return the azimuthal standard deviation of *values* per bin."""
        import torch

        mean = self.sum(values)[self.filled] / self.counts[self.filled]
        mean_sq = self.sum(values**2)[self.filled] / self.counts[self.filled]
        out = torch.full((self.n_bins,), fill, dtype=torch.float32)
        out[self.filled] = torch.sqrt((mean_sq - mean**2).clamp(min=0.0))
        return out

    def cdf(self, values: "torch.Tensor", *, normalize: bool = True) -> "torch.Tensor":
        """Return the energy cumulated over radius, read at the edges.

        The abscissa is :attr:`edges`, not :attr:`centres`: what an
        annulus contributes is enclosed by its outer edge, so that is the
        radius the running total belongs to.  The curve therefore starts
        at zero, at radius zero, and carries one more point than there
        are bins.

        Parameters
        ----------
        values : torch.Tensor
            Values to integrate, one per pixel.
        normalize : bool, optional
            Divide by the total, so that the curve reaches one at the
            edge of the grid (default ``True``).

        Returns
        -------
        torch.Tensor
            Shape ``(n_bins + 1,)``, aligned on :attr:`edges`.
        """
        profile = self.mean(values, fill=0.0).clamp(min=0.0)
        return cumulate(profile, self.edges, normalize=normalize)

    def radius_at(self, cdf: "torch.Tensor", fraction: float) -> float:
        """Return the radius enclosing *fraction* of the energy.

        Interpolates inside the bin the fraction falls in, rather than
        snapping to an edge: against a King profile of known encircled
        energy that is the difference between seven percent of error and
        two tenths of one.

        Parameters
        ----------
        cdf : torch.Tensor
            Cumulated energy from :meth:`cdf`, shape ``(n_bins + 1,)``.
        fraction : float
            Energy fraction, between zero and one.

        Returns
        -------
        float
            Radius, in the unit the distances came in.
        """
        return float(np.interp(fraction, cdf.numpy(), self.edges.numpy()))


def cumulate(
    profile: "torch.Tensor",
    edges: "torch.Tensor",
    *,
    normalize: bool = True,
) -> "torch.Tensor":
    """Integrate a radial *profile* over the disc, annulus by annulus.

    Parameters
    ----------
    profile : torch.Tensor
        Azimuthal mean per bin, shape ``(n_bins,)``.
    edges : torch.Tensor
        Bin edges, shape ``(n_bins + 1,)``.
    normalize : bool, optional
        Divide by the total (default ``True``).

    Returns
    -------
    torch.Tensor
        Cumulated energy, shape ``(n_bins + 1,)``, starting at zero: the
        value at index ``i`` is the energy enclosed by ``edges[i]``.
    """
    import torch

    cdf = torch.zeros(profile.numel() + 1, dtype=profile.dtype)
    cdf[1:] = torch.cumsum(profile * annulus_areas(edges), dim=0)
    if normalize and cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return cdf
