"""Radial binning of 2-D fields, shared by the radial analysis.

Classes
-------
    RadialBinning
        Pixels grouped by radius once, reduced many times.

Functions
---------
    radial_distances
        Flat radii and values of a field.
    natural_npix
        Largest bin count leaving no bin empty.
    edges_from_centres
        Bin edges implied by bin centres.
    annulus_areas
        Area of each annulus.
    cumulate
        Radial profile integrated over the disc.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import xarray as xr
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from .._logging import get_logger

logger = get_logger(__name__)

#: Extrapolation beyond the last radius, as a fraction, above which a
#: warning is logged.  Square-grid corners alone overshoot by 0.4 %.
OVERSHOOT_WARNS = 0.05


def _sample_radial_from_cdf(
    profile: xr.DataArray,
    n: int,
    max_gap: float | None = None,
) -> np.ndarray:
    """Return *n* radii, denser where the profile varies most.

    Sampled by inverting the CDF of ``|df/dr|``; with *max_gap*, points
    are added so that no two consecutive radii are further apart.
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
    """Return the 2-D field on ``(xx, yy)`` of a radial profile, by Pchip."""
    rr = np.sqrt(xx**2 + yy**2)
    _, unique_idx = np.unique(r, return_index=True)
    r_u = r[unique_idx]
    v_u = values[unique_idx]

    # Pchip extrapolates past the last radius: warn only when it goes far.
    outside = rr > r_u[-1]
    if outside.any():
        overshoot = float(rr.max()) / float(r_u[-1]) - 1.0
        emit = logger.warning if overshoot > OVERSHOOT_WARNS else logger.debug
        emit(
            "profile.extrapolated",
            pixels=int(outside.sum()),
            fraction=round(float(outside.mean()), 4),
            overshoot_pct=round(100.0 * overshoot, 3),
            profile_max=round(float(r_u[-1]), 4),
            grid_max=round(float(rr.max()), 4),
        )

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
    """Return the flat float32 radii and values of a field.

    A kernel (dims ``y_psf``, ``x_psf``) is centred on 0; a field on its
    coordinate mean, unless *center* ``(cx, cy)`` is given.  *var_name*
    selects the variable of a Dataset.
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
    """Return the largest bin count keeping every bin one pixel wide."""
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
    """Return the ``n + 1`` edges halfway between *centres*, the first >= 0."""
    step = centres[1:] - centres[:-1]
    edges = torch.empty(centres.numel() + 1, dtype=centres.dtype)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = (centres[0] - 0.5 * step[0]).clamp(min=0.0)
    edges[-1] = centres[-1] + 0.5 * step[-1]
    return edges


def annulus_areas(edges: "torch.Tensor") -> "torch.Tensor":
    """Return the area of each annulus between consecutive *edges*."""
    return math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


class RadialBinning:
    """Pixels grouped by distance to a centre, for repeated reductions.

    Parameters
    ----------
    radius : torch.Tensor
        Flat distances, any unit.
    n_bins : int
        Number of bins.
    r_max : float or None, optional
        Outer edge, the largest distance by default; beyond it, pixels
        fall into the last bin.

    Attributes
    ----------
    edges : torch.Tensor
        ``n_bins + 1`` bin edges.
    midpoints : torch.Tensor
        Bin midpoints.
    centres : torch.Tensor
        Midpoints with the first at 0: the abscissa of a profile.
    counts, filled : torch.Tensor
        Pixels per bin, and whether the bin caught any.
    area : torch.Tensor
        Area of each annulus.
    """

    def __init__(
        self,
        radius: "torch.Tensor",
        n_bins: int,
        *,
        r_max: float | None = None,
    ) -> None:
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
        """Return the natural binning of an ``n × n`` grid of pixel *res*."""
        half = (n // 2) * res
        coords = np.linspace(-half, half, n, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords)
        radius = torch.from_numpy(np.hypot(xx, yy).ravel())
        return cls(radius, _natural_bins(n))

    def sum(self, values: "torch.Tensor") -> "torch.Tensor":
        """Return the sum of *values* per bin, shape ``(n_bins,)``."""
        return torch.bincount(self.index, weights=values.ravel(), minlength=self.n_bins)

    def mean(
        self, values: "torch.Tensor", fill: float = float("nan")
    ) -> "torch.Tensor":
        """Return the mean of *values* per bin, *fill* in empty bins."""
        total = self.sum(values)
        out = torch.full((self.n_bins,), fill, dtype=torch.float32)
        out[self.filled] = total[self.filled] / self.counts[self.filled]
        return out

    def std(self, values: "torch.Tensor", fill: float = float("nan")) -> "torch.Tensor":
        """Return the azimuthal standard deviation of *values* per bin."""
        mean = self.sum(values)[self.filled] / self.counts[self.filled]
        mean_sq = self.sum(values**2)[self.filled] / self.counts[self.filled]
        out = torch.full((self.n_bins,), fill, dtype=torch.float32)
        out[self.filled] = torch.sqrt((mean_sq - mean**2).clamp(min=0.0))
        return out

    def cdf(self, values: "torch.Tensor", *, normalize: bool = True) -> "torch.Tensor":
        """Return the energy enclosed by each of :attr:`edges`, from 0."""
        profile = self.mean(values, fill=0.0).clamp(min=0.0)
        return cumulate(profile, self.edges, normalize=normalize)

    def radius_at(self, cdf: "torch.Tensor", fraction: float) -> float:
        """Return the radius enclosing *fraction* of a :meth:`cdf`, interpolated."""
        return float(np.interp(fraction, cdf.numpy(), self.edges.numpy()))


def cumulate(
    profile: "torch.Tensor",
    edges: "torch.Tensor",
    *,
    normalize: bool = True,
) -> "torch.Tensor":
    """Return the energy enclosed by each of *edges*, from 0, of a radial profile."""
    cdf = torch.zeros(profile.numel() + 1, dtype=profile.dtype)
    cdf[1:] = torch.cumsum(profile * annulus_areas(edges), dim=0)
    if normalize and cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return cdf
