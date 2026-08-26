"""How much of a kernel sits where, and what it does to spatial detail.

The encircled energy answers the first question and the modulation
transfer function the second.  Both were missing from the package: the
encircled energy existed twice, once in the radial accessor and once
rewritten inside ``optim/landscape.py`` for speed, and the MTF, which is
the canonical way to compare a PSF against the instrument literature,
was nowhere.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import xarray as xr

__all__ = [
    "encircled_energy",
    "encircled_radii",
    "encircled_radius",
    "fwhm",
    "mtf",
]


class _RadialGrid:
    """The radial binning of a square grid, computed once.

    A landscape scan evaluates hundreds of kernels on one grid, so the
    pixel to bin mapping, the bin counts and the annulus areas are built
    here and reused, rather than rebuilt per kernel.

    Parameters
    ----------
    n : int
        Grid side in pixels.
    res : float
        Pixel size, in the unit the radii come out in.
    """

    def __init__(self, n: int, res: float) -> None:
        self.n = n
        self.res = res
        npix = max(int((n - 1) / math.sqrt(2)) - 1, 2)
        self.npix = npix

        half = (n // 2) * res
        coords = np.linspace(-half, half, n, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords)
        radius = torch.from_numpy(np.hypot(xx, yy).ravel())

        bins = torch.linspace(0.0, float(radius.max()), npix + 1)
        self.index = (torch.bucketize(radius, bins, right=False) - 1).clamp(
            0, npix - 1
        )
        self.counts = torch.bincount(self.index, minlength=npix).float()
        self.filled = self.counts > 0

        centres = (0.5 * (bins[:-1] + bins[1:])).clone()
        centres[0] = 0.0
        step = centres[1:] - centres[:-1]
        edges = torch.empty(npix + 1, dtype=centres.dtype)
        edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
        edges[0] = centres[0] - 0.5 * step[0]
        edges[-1] = centres[-1] + 0.5 * step[-1]

        self.centres = centres
        self.area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

    def profile(self, values: torch.Tensor) -> torch.Tensor:
        """Return the azimuthal mean of *values* per annulus."""
        total = torch.bincount(
            self.index, weights=values.ravel(), minlength=self.npix
        )
        mean = torch.zeros(self.npix, dtype=torch.float32)
        mean[self.filled] = total[self.filled] / self.counts[self.filled]
        return mean

    def cdf(self, values: torch.Tensor) -> torch.Tensor:
        """Return the cumulated energy, normalised to one at the edge."""
        cumulated = torch.cumsum(self.profile(values) * self.area, dim=0)
        if cumulated[-1] > 0:
            cumulated = cumulated / cumulated[-1]
        return cumulated

    def radius_at(self, cdf: torch.Tensor, fraction: float) -> float:
        """Return the radius where *cdf* first reaches *fraction*."""
        index = min(
            int(np.searchsorted(cdf.numpy(), fraction)), self.npix - 1
        )
        return float(self.centres[index])


def _as_grid(kernel: xr.DataArray) -> tuple[_RadialGrid, torch.Tensor]:
    """Return the binning of *kernel*'s grid and its clamped values."""
    values = torch.from_numpy(
        np.asarray(kernel.values, dtype=np.float32)
    ).clamp(min=0.0)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(
            f"a kernel must be a square 2-D array, got shape {values.shape}"
        )
    name = "x_psf" if "x_psf" in kernel.coords else "x"
    coord = np.asarray(kernel.coords[name].values, dtype=float)
    res = float(abs(coord[1] - coord[0]))
    return _RadialGrid(values.shape[0], res), values


def encircled_energy(kernel: xr.DataArray) -> xr.DataArray:
    """Return the fraction of a kernel's energy within each radius.

    Parameters
    ----------
    kernel : xr.DataArray
        Square 2-D kernel, with either ``x_psf`` or ``x`` coordinates.

    Returns
    -------
    xr.DataArray
        Cumulated energy against radius, with dim ``"r"``, rising to one
        at the edge of the grid.

    Notes
    -----
    The normalisation is the grid's, not the plane's.  A profile with a
    heavy tail keeps part of its energy outside any finite grid, so the
    radii below describe the kernel *as sampled*, which is also how it is
    convolved.
    """
    grid, values = _as_grid(kernel)
    return xr.DataArray(
        grid.cdf(values).numpy(),
        dims=["r"],
        coords={"r": grid.centres.numpy()},
    )


def encircled_radius(kernel: xr.DataArray, fraction: float = 0.5) -> float:
    """Return the radius encircling *fraction* of a kernel's energy.

    Parameters
    ----------
    kernel : xr.DataArray
        Square 2-D kernel.
    fraction : float, optional
        Energy fraction, between zero and one.

    Returns
    -------
    float
        Radius, in the unit of the kernel's coordinates.
    """
    grid, values = _as_grid(kernel)
    return grid.radius_at(grid.cdf(values), fraction)


def encircled_radii(
    kernels: list[torch.Tensor],
    n: int,
    res: float,
    fractions: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Return the encircled-energy radii of many kernels on one grid.

    Parameters
    ----------
    kernels : list[torch.Tensor]
        Square kernels, all on the same grid.
    n : int
        Grid side in pixels.
    res : float
        Pixel size.
    fractions : list[float] or None, optional
        Energy fractions.  Defaults to ``[0.10, 0.50, 0.99]``.

    Returns
    -------
    dict[str, np.ndarray]
        One array per fraction, keyed ``"EE10%"``, ``"EE50%"`` and so on,
        in the order the kernels were given.
    """
    if fractions is None:
        fractions = [0.10, 0.50, 0.99]
    keys = [f"EE{int(f * 100)}%" for f in fractions]
    if not kernels:
        return {key: np.array([], dtype=np.float32) for key in keys}

    grid = _RadialGrid(n, res)
    out: dict[str, list[float]] = {key: [] for key in keys}
    with torch.no_grad():
        for kernel in kernels:
            cdf = grid.cdf(kernel.detach().cpu().float().clamp(min=0.0))
            for fraction, key in zip(fractions, keys, strict=True):
                out[key].append(grid.radius_at(cdf, fraction))
    return {key: np.array(v, dtype=np.float32) for key, v in out.items()}


def fwhm(kernel: xr.DataArray) -> float:
    """Return the full width at half maximum of a kernel.

    Parameters
    ----------
    kernel : xr.DataArray
        Square 2-D kernel, assumed to peak at its centre.

    Returns
    -------
    float
        Width, in the unit of the kernel's coordinates.  ``nan`` when the
        profile never falls to half its peak inside the grid.
    """
    grid, values = _as_grid(kernel)
    profile = grid.profile(values).numpy()
    radii = grid.centres.numpy()
    half = 0.5 * float(profile[0])
    below = np.nonzero(profile <= half)[0]
    if below.size == 0:
        return float("nan")
    first = int(below[0])
    if first == 0:
        return 0.0
    # Linear crossing between the last point above and the first below.
    r0, r1 = radii[first - 1], radii[first]
    v0, v1 = profile[first - 1], profile[first]
    crossing = r0 + (v0 - half) * (r1 - r0) / max(v0 - v1, 1e-30)
    return float(2.0 * crossing)


def mtf(kernel: xr.DataArray) -> xr.DataArray:
    """Return the modulation transfer function of a kernel.

    The MTF is the modulus of the Fourier transform of the PSF,
    normalised to one at zero frequency, averaged over azimuth.  It says
    how much contrast survives at each spatial frequency, which is the
    quantity instrument papers report.

    Parameters
    ----------
    kernel : xr.DataArray
        Square 2-D kernel.

    Returns
    -------
    xr.DataArray
        MTF against spatial frequency, with dim ``"f"`` in cycles per
        unit of the kernel's coordinates, from zero to the Nyquist
        frequency.  The first sample is the zero frequency, where the
        MTF is one by construction.
    """
    grid, values = _as_grid(kernel)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(values.numpy())))
    spectrum = spectrum / spectrum.max()

    n, res = grid.n, grid.res
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=res))
    fx, fy = np.meshgrid(freq, freq)
    radius = np.hypot(fx, fy).ravel()

    nyquist = 0.5 / res
    n_bins = max(n // 4, 8)
    edges = np.linspace(0.0, nyquist, n_bins + 1)
    index = np.clip(np.digitize(radius, edges) - 1, 0, n_bins - 1)
    inside = radius <= nyquist

    total = np.bincount(
        index[inside], weights=spectrum.ravel()[inside], minlength=n_bins
    )
    counts = np.bincount(index[inside], minlength=n_bins)
    profile = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)

    # The zero frequency is prepended rather than binned: for a
    # non-negative kernel the spectrum peaks there, so the MTF is one by
    # construction, whereas the lowest band averages it with its
    # neighbours and lands just below.
    centres = np.concatenate([[0.0], 0.5 * (edges[:-1] + edges[1:])])
    profile = np.concatenate([[1.0], profile])

    return xr.DataArray(
        profile.astype(np.float32), dims=["f"], coords={"f": centres}
    )
