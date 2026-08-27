"""How much of a kernel sits where, and what it does to spatial detail.

The encircled energy answers the first question and the modulation
transfer function the second.  Both were missing from the package: the
encircled energy existed twice, once in the radial accessor and once
rewritten inside ``optim/landscape.py`` for speed, and the MTF, which is
the canonical way to compare a PSF against the instrument literature,
was nowhere.  The binning itself lives in :mod:`adjeff.utils.radial`,
shared with the radial profile.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import xarray as xr

from adjeff.exceptions import ConfigurationError
from adjeff.utils.radial import RadialBinning

from ._plane import grid_share

__all__ = [
    "encircled_energy",
    "encircled_radii",
    "encircled_radius",
    "fwhm",
    "mtf",
]


def _as_grid(
    kernel: xr.DataArray,
) -> tuple[RadialBinning, torch.Tensor, float]:
    """Return the binning of *kernel*'s grid, with values and pixel size.

    The values come back clamped to non-negative, which is what every
    energy measure below assumes.
    """
    values = torch.from_numpy(np.asarray(kernel.values, dtype=np.float32)).clamp(
        min=0.0
    )
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(
            f"a kernel must be a square 2-D array, got shape {values.shape}"
        )
    name = "x_psf" if "x_psf" in kernel.coords else "x"
    coord = np.asarray(kernel.coords[name].values, dtype=float)
    res = float(abs(coord[1] - coord[0]))
    return RadialBinning.on_square_grid(values.shape[0], res), values, res


def encircled_energy(
    kernel: xr.DataArray, *, normalize: Literal["grid", "plane"] = "grid"
) -> xr.DataArray:
    """Return the fraction of a kernel's energy within each radius.

    Parameters
    ----------
    kernel : xr.DataArray
        Square 2-D kernel, with either ``x_psf`` or ``x`` coordinates.
    normalize : {"grid", "plane"}, optional
        What the curve is a fraction *of* (default ``"grid"``).

        - ``"grid"`` divides by the energy the grid holds, so the curve
          reaches one at its edge.  This is the kernel as sampled, and
          as convolved.
        - ``"plane"`` divides by the integral of the fitted profile over
          the whole plane, so the curve stops at the fraction the grid
          actually captured.  Needs an analytical kernel carrying its
          model and parameters.

    Returns
    -------
    xr.DataArray
        Cumulated energy against radius, with dim ``"r"``, starting at
        zero.  The radii are the bin edges: each value is the energy
        enclosed by that radius, which is what makes the curve
        invertible.

    Raises
    ------
    ConfigurationError
        With ``normalize="plane"`` on a kernel that carries no
        provenance, or whose profile has no finite energy over the
        plane.

    Notes
    -----
    The two normalisations answer different questions and neither is
    always right.  Grid normalisation describes the operator: that is the
    kernel the convolution applies, truncation included.  Plane
    normalisation describes the profile the fit believes in, and makes
    the truncation visible: on the manuscript's aerosol sweep a King
    fitted at an optical thickness of 0.1 stops at 0.956 where one at 0.7
    stops at 0.994, a difference grid normalisation hides by sending both
    to one.

    The plane ceiling is an extrapolation. Nothing constrains the fitted
    profile beyond the simulated domain, and it is the widest kernel
    whose ceiling is least certain.
    """
    grid, values, _ = _as_grid(kernel)
    curve = grid.cdf(values).numpy()
    if normalize == "plane":
        curve = curve * grid_share(kernel, float(grid.edges[-1]))
    elif normalize != "grid":
        raise ConfigurationError(
            f"normalize must be 'grid' or 'plane', not {normalize!r}"
        )
    return xr.DataArray(curve, dims=["r"], coords={"r": grid.edges.numpy()})


def encircled_radius(
    kernel: xr.DataArray,
    fraction: float = 0.5,
    *,
    normalize: Literal["grid", "plane"] = "grid",
) -> float:
    """Return the radius encircling *fraction* of a kernel's energy.

    Parameters
    ----------
    kernel : xr.DataArray
        Square 2-D kernel.
    fraction : float, optional
        Energy fraction, between zero and one.
    normalize : {"grid", "plane"}, optional
        Fraction of what; see :func:`encircled_energy` (default
        ``"grid"``).

    Returns
    -------
    float
        Radius, in the unit of the kernel's coordinates.  ``nan`` when
        *fraction* lies above what the grid captured, which
        ``normalize="plane"`` makes possible: a fitted King at an optical
        thickness of 0.1 never reaches 0.99 of its plane energy inside a
        240 km domain.

    Notes
    -----
    The two normalisations can differ by a factor of two.  On the
    manuscript's sweep the 90 % radius of the low-load kernel is 15.8 km
    of what the grid holds, and 31.7 km of what the profile implies.
    """
    grid, values, _ = _as_grid(kernel)
    curve = grid.cdf(values)
    if normalize == "plane":
        curve = curve * grid_share(kernel, float(grid.edges[-1]))
    elif normalize != "grid":
        raise ConfigurationError(
            f"normalize must be 'grid' or 'plane', not {normalize!r}"
        )
    if fraction > float(curve[-1]):
        return float("nan")
    return grid.radius_at(curve, fraction)


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

    grid = RadialBinning.on_square_grid(n, res)
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
    grid, values, _ = _as_grid(kernel)
    profile = grid.mean(values, fill=0.0).numpy()
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
    grid, values, res = _as_grid(kernel)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(values.numpy())))
    spectrum = spectrum / spectrum.max()

    n = values.shape[0]
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=res))
    fx, fy = np.meshgrid(freq, freq)
    radius = np.hypot(fx, fy).ravel()

    nyquist = 0.5 / res
    inside = radius <= nyquist
    bands = RadialBinning(
        torch.from_numpy(radius[inside]).float(),
        max(n // 4, 8),
        r_max=nyquist,
    )
    profile = bands.mean(torch.from_numpy(spectrum.ravel()[inside]).float()).numpy()

    # The zero frequency is prepended rather than binned: for a
    # non-negative kernel the spectrum peaks there, so the MTF is one by
    # construction, whereas the lowest band averages it with its
    # neighbours and lands just below.  The bands keep their true
    # midpoints for the same reason.
    centres = np.concatenate([[0.0], bands.midpoints.numpy()])
    profile = np.concatenate([[1.0], profile])

    return xr.DataArray(profile.astype(np.float32), dims=["f"], coords={"f": centres})
