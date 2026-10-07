"""Tools for analyzing point spread functions (PSFs).

Functions
---------
    encircled_energy
        Fraction of PSF energy enclosed within each radius.
    encircled_radius
        Radius enclosing a given fraction of PSF energy.
    encircled_radii
        Same, for many kernels on one grid.
    fwhm
        Full width at half maximum of the PSF.
    mtf
        Modulation transfer function of the PSF.
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
    """Return the binning of *kernel*'s grid, its values (>= 0) and pixel size."""
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
    """Return the energy fraction enclosed by each radius, on dim ``"r"``.

    Parameters
    ----------
    kernel : xr.DataArray
        Square 2-D kernel, on ``x_psf`` or ``x``.
    normalize : {"grid", "plane"}, optional
        Divide by the energy on the grid (reaching 1 at its edge), or
        by the plane integral of the fitted profile (analytical kernels
        only).  ``"grid"`` by default.

    Raises
    ------
    ConfigurationError
        For ``"plane"`` on a kernel with no finite plane energy.
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
    """Return the radius enclosing *fraction* of a kernel's energy.

    *normalize* is that of :func:`encircled_energy`.  ``nan`` when the
    grid holds less than *fraction*, which ``"plane"`` allows.
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
    """Return the encircled radii of *kernels*, all on one ``n × n`` grid.

    *res* is the pixel size.  One array per fraction, ``[0.10, 0.50, 0.99]``
    by default, keyed ``"EE10%"``…, in the order of *kernels*.
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
    """Return the full width at half maximum of a kernel peaking at its centre.

    ``nan`` when the profile stays above half its peak on the grid.
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
    """Return the azimuthally averaged MTF of a kernel, 1 at zero frequency.

    On dim ``"f"``, in cycles per coordinate unit, up to Nyquist.
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
