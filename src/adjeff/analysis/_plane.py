"""Share of an analytical kernel's plane energy that a finite grid holds.

Normalising an encircled-energy curve by the grid total makes every
curve reach one at the grid edge, which hides how much energy each
kernel loses off the grid (4.4 % for a King at AOT 0.1 on a 240 km
domain, 0.6 % at AOT 0.7).  Normalising by the plane total keeps that
difference, but the plane total is the integral of the fitted profile:
it exists only for an analytical kernel whose integral converges, so
not for a Voigt.  Unsupported cases raise.

Functions
---------
    grid_share
        Fraction of the kernel's plane energy within a given radius.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

from adjeff.exceptions import ConfigurationError

__all__ = ["grid_share"]

_QUADRATURE_STEPS = 20001


def _midpoints(radius: float) -> "np.ndarray":
    """Return the radii the quadrature evaluates a profile at."""
    return np.linspace(0.0, radius, _QUADRATURE_STEPS)


def _plane_total_gaussian(params: dict[str, float]) -> float:
    """Return the plane integral of ``exp(-r**2 / (2 sigma**2))``."""
    sigma = float(params["sigma"])
    return 2.0 * math.pi * sigma**2


def _cum_gaussian(r: float, params: dict[str, float]) -> float:
    """Return the integral of the Gaussian profile over a disc of radius *r*."""
    sigma = float(params["sigma"])
    return 2.0 * math.pi * sigma**2 * (1.0 - math.exp(-(r**2) / (2.0 * sigma**2)))


def _plane_total_king(params: dict[str, float]) -> float:
    """Return the plane integral of ``(1 + r**2 / a)**-gamma``."""
    sigma, gamma = float(params["sigma"]), float(params["gamma"])
    if gamma <= 1.0:
        raise ConfigurationError(
            f"a King profile of gamma={gamma:g} has no finite energy over the "
            "plane: its tail falls as r**-2 or slower. Normalise on the grid, "
            "or refit with gamma bounded away from one."
        )
    return math.pi * (2.0 * gamma * sigma**2) / (gamma - 1.0)


def _cum_king(r: float, params: dict[str, float]) -> float:
    """Return the integral of the King profile over a disc of radius *r*."""
    sigma, gamma = float(params["sigma"]), float(params["gamma"])
    a = 2.0 * gamma * sigma**2
    return float(
        math.pi * a * (1.0 - (1.0 + r**2 / a) ** (1.0 - gamma)) / (gamma - 1.0)
    )


def _plane_total_gg(params: dict[str, float]) -> float:
    """Return the plane integral of ``exp(-(r / sigma)**n)``."""
    sigma, n = float(params["sigma"]), float(params["n"])
    return 2.0 * math.pi * sigma**2 * math.gamma(2.0 / n) / n


def _cum_gg(r: float, params: dict[str, float]) -> float:
    """Return the integral of the generalised Gaussian over a disc."""
    sigma, n = float(params["sigma"]), float(params["n"])
    radius = _midpoints(r)
    return float(
        2.0 * math.pi * np.trapezoid(radius * np.exp(-((radius / sigma) ** n)), radius)
    )


def _plane_total_moffat(params: dict[str, float]) -> float:
    """Return the plane integral of ``(1 + (r / alpha)**(2 beta))**-gamma``."""
    alpha = float(params["alpha"])
    beta, gamma = float(params["beta"]), float(params["gamma"])
    if gamma * beta <= 1.0:
        raise ConfigurationError(
            f"a generalised Moffat of beta={beta:g}, gamma={gamma:g} has no "
            "finite energy over the plane: it needs gamma * beta > 1, and "
            f"this is {gamma * beta:g}. Normalise on the grid instead."
        )
    log_beta_fn = (
        math.lgamma(1.0 / beta) + math.lgamma(gamma - 1.0 / beta) - math.lgamma(gamma)
    )
    return math.pi * alpha**2 * math.exp(log_beta_fn) / beta


def _cum_moffat(r: float, params: dict[str, float]) -> float:
    """Return the integral of the generalised Moffat over a disc."""
    alpha = float(params["alpha"])
    beta, gamma = float(params["beta"]), float(params["gamma"])
    radius = _midpoints(r)
    profile = (1.0 + (radius / alpha) ** (2.0 * beta)) ** (-gamma)
    return float(2.0 * math.pi * np.trapezoid(radius * profile, radius))


_MODELS = {
    "Gaussian": (_plane_total_gaussian, _cum_gaussian),
    "King": (_plane_total_king, _cum_king),
    "GeneralizedGaussian": (_plane_total_gg, _cum_gg),
    "MoffatGeneralized": (_plane_total_moffat, _cum_moffat),
}


def grid_share(kernel: xr.DataArray, radius_km: float) -> float:
    """Return the fraction of a kernel's plane energy inside *radius_km*.

    Parameters
    ----------
    kernel : xr.DataArray
        Kernel carrying the provenance :meth:`PSFModule.to_dataarray`
        stamps: ``adjeff:model`` and ``adjeff:params``.
    radius_km : float
        Radius the grid reaches, normally its half-diagonal.

    Returns
    -------
    float
        A number in ``(0, 1]``: one when the grid holds everything, and
        the ceiling an encircled-energy curve should stop at otherwise.

    Raises
    ------
    ConfigurationError
        When the kernel carries no provenance, when its model has no
        entry here, or when its parameters put its energy beyond any
        finite total.  Guessing would silently rescale a published curve.

    Examples
    --------
    >>> from adjeff.core import KingPSF, PSFGrid, S2Band
    >>> psf = KingPSF(PSFGrid(0.12, 1999), S2Band.B03, sigma=0.173, gamma=1.242)
    >>> round(grid_share(psf.to_dataarray(), 169.6), 4)
    0.9557
    """
    model = kernel.attrs.get("adjeff:model")
    params = kernel.attrs.get("adjeff:params")
    if model is None or params is None:
        raise ConfigurationError(
            "plane normalisation needs the kernel's model and parameters, "
            "which only an analytical PSF carries. This kernel has "
            f"{sorted(kernel.attrs)}. Normalise on the grid instead."
        )
    if model not in _MODELS:
        known = ", ".join(sorted(_MODELS))
        raise ConfigurationError(
            f"no plane integral is known for the {model!r} kernel; the "
            f"models that have one are {known}. A Voigt is excluded on "
            "purpose: its Lorentzian part has no finite energy over the "
            "plane."
        )
    plane_total, disc_total = _MODELS[model]
    return disc_total(radius_km, dict(params)) / plane_total(dict(params))
