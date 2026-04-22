"""Implement analytical subclasses of PSFModule.

Each subclass is a torch.nn.Module with trainable nn.Parameters.
Each subclass implements the PSFModule protocol (forward + to_dataarray).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from adjeff.utils import ConstrainedParameter, ExpTransform, SigmoidTransform

from ._psf import PSFGrid, PSFModule
from .bands import SensorBand


class GaussPSF(nn.Module, PSFModule):
    """Trainable Gaussian PSF model with constrained sigma.

    Parameters
    ----------
    grid : PSFGrid
        Spatial sampling configuration (size and resolution).
    band : SensorBand
        Spectral band this PSF applies to.
    sigma : float
        Initial Gaussian standard deviation [km].
    """

    def __init__(self, grid: PSFGrid, band: SensorBand, sigma: float) -> None:
        nn.Module.__init__(self)
        self.grid = grid
        self.band = band

        self.sigma = ConstrainedParameter(
            init_value=torch.tensor(sigma, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-3,
            max_val=50.0,
            name="sigma",
        )

    def forward(self) -> torch.Tensor:
        """Return normalised Gaussian kernel on the grid."""
        X, Y = self.grid.meshgrid()
        r2 = X**2 + Y**2
        kernel = torch.exp(-r2 / (2.0 * self.sigma.value**2))
        return kernel / kernel.sum()

    def param_dict(self) -> dict[str, float]:
        """Return ``{"sigma": <value>}``."""
        return {"sigma": float(self.sigma.value)}

    @torch.no_grad()
    def to_dataarray(self) -> xr.DataArray:
        """Return the Gaussian kernel as a DataArray."""
        kernel = self.forward().detach().cpu().numpy()
        coords = self.grid.as_coords()
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=coords,
            attrs={
                "adjeff:kind": "analytical",
                "adjeff:model": "Gaussian",
                "adjeff:params": {
                    "sigma": float(self.sigma.value),
                },
                "band": self.band,
            },
        )


class GaussGeneralPSF(nn.Module, PSFModule):
    """Trainable Generalised Gaussian PSF model.

    The kernel follows ``exp(-(r/σ)ⁿ)``, where *n* controls the shape:
    ``n=2`` yields a standard Gaussian, ``n=1`` a Laplacian.

    Parameters
    ----------
    grid : PSFGrid
        Spatial sampling configuration.
    band : SensorBand
        Spectral band this PSF applies to.
    sigma : float
        Initial scale parameter [km]. Constrained to ``[1e-6, 1.0]``.
    n : float
        Initial shape exponent. Constrained to ``[0.1, 0.4]``.
    """

    def __init__(
        self, grid: PSFGrid, band: SensorBand, sigma: float, n: float
    ) -> None:
        nn.Module.__init__(self)
        self.grid = grid
        self.band = band

        self.sigma = ConstrainedParameter(
            init_value=torch.tensor(sigma, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-6,
            max_val=1e0,
            name="sigma",
        )

        self.n = ConstrainedParameter(
            init_value=torch.tensor(n, dtype=torch.float32),
            transform=SigmoidTransform(0.1, 0.4),
            min_val=0.1,
            max_val=0.4,
            name="n",
        )

    def forward(self) -> torch.Tensor:
        """Return normalised Generalised Gaussian kernel on the grid."""
        X, Y = self.grid.meshgrid()
        r = torch.sqrt(X**2 + Y**2)
        sigma = self.sigma.value
        n = self.n.value
        kernel = torch.exp(-((r / sigma) ** n))
        return kernel / kernel.sum()

    def param_dict(self) -> dict[str, float]:
        """Return ``{"sigma": <value>, "n": <value>}``."""
        return {"sigma": float(self.sigma.value), "n": float(self.n.value)}

    @torch.no_grad()
    def to_dataarray(self) -> xr.DataArray:
        """Return the Gaussian Generalized kernel as a DataArray."""
        kernel = self.forward().detach().cpu().numpy()
        coords = self.grid.as_coords()
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=coords,
            attrs={
                "adjeff:kind": "analytical",
                "adjeff:model": "GaussianGeneral",
                "adjeff:params": {
                    "sigma": float(self.sigma.value),
                    "n": float(self.n.value),
                },
                "band": self.band,
            },
        )


class VoigtPSF(nn.Module, PSFModule):
    """Trainable pseudo-Voigt PSF kernel.

    Linearly mixes a Gaussian (width *sigma*) and a Lorentzian (width
    *gamma*) using the Thompson et al. mixing parameter η.

    Parameters
    ----------
    grid : PSFGrid
        Spatial sampling configuration.
    band : SensorBand
        Spectral band this PSF applies to.
    sigma : float
        Initial Gaussian width [km].
    gamma : float
        Initial Lorentzian half-width [km].
    """

    def __init__(
        self,
        grid: PSFGrid,
        band: SensorBand,
        sigma: float,
        gamma: float,
    ) -> None:
        nn.Module.__init__(self)
        self.grid = grid
        self.band = band

        self.sigma = ConstrainedParameter(
            init_value=torch.tensor(sigma, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-3,
            max_val=50.0,
            name="sigma",
        )

        self.gamma = ConstrainedParameter(
            init_value=torch.tensor(gamma, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-3,
            max_val=50.0,
            name="gamma",
        )

    def _eta(self) -> torch.Tensor:
        """Return the pseudo-Voigt mixing parameter eta."""
        FG = 2.0 * torch.sqrt(torch.tensor(2.0 * np.log(2.0)))
        FG = FG * self.sigma.value
        FL = 2.0 * self.gamma.value
        F = (
            FG**5
            + 2.69269 * FG**4 * FL
            + 2.42843 * FG**3 * FL**2
            + 4.47163 * FG**2 * FL**3
            + 0.07842 * FG * FL**4
            + FL**5
        ).pow(1 / 5)
        y = (FL / F).clamp(0.0, 1.0)
        return (1.36603 * y - 0.47719 * y**2 + 0.11116 * y**3).clamp(0.0, 1.0)

    def forward(self) -> torch.Tensor:
        """Return normalised Voigt kernel on the grid."""
        X, Y = self.grid.meshgrid()
        r = torch.sqrt(X**2 + Y**2)
        G = torch.exp(-(r**2) / (2.0 * self.sigma.value**2))
        L = 1.0 / (1.0 + (r / self.gamma.value) ** 2)
        eta = self._eta()
        V = eta * L + (1.0 - eta) * G
        return V / V.sum()  # type: ignore[no-any-return, unused-ignore]

    def param_dict(self) -> dict[str, float]:
        """Return ``{"sigma": <value>, "gamma": <value>}``."""
        return {
            "sigma": float(self.sigma.value),
            "gamma": float(self.gamma.value),
        }

    @torch.no_grad()
    def to_dataarray(self) -> xr.DataArray:
        """Return the Voigt kernel as a DataArray."""
        kernel = self.forward().detach().cpu().numpy()
        coords = self.grid.as_coords()
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=coords,
            attrs={
                "adjeff:kind": "analytical",
                "adjeff:model": "Voigt",
                "adjeff:params": {
                    "sigma": float(self.sigma.value),
                    "gamma": float(self.gamma.value),
                },
                "band": self.band,
            },
        )


class KingPSF(nn.Module, PSFModule):
    """Trainable King profile PSF kernel.

    The kernel follows ``(1 + r² / (2σ²γ))^{-γ}``, producing a
    power-law tail controlled by *gamma*.

    Parameters
    ----------
    grid : PSFGrid
        Spatial sampling configuration.
    band : SensorBand
        Spectral band this PSF applies to.
    sigma : float
        Initial core width [km].
    gamma : float
        Initial power-law index.
    """

    def __init__(
        self,
        grid: PSFGrid,
        band: SensorBand,
        sigma: float,
        gamma: float,
    ) -> None:
        nn.Module.__init__(self)
        self.grid = grid
        self.band = band

        self.sigma = ConstrainedParameter(
            init_value=torch.tensor(sigma, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-3,
            max_val=1e2,
            name="sigma",
        )

        self.gamma = ConstrainedParameter(
            init_value=torch.tensor(gamma, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-1,
            max_val=1e1,
            name="gamma",
        )

    def forward(self) -> torch.Tensor:
        """Return normalised King kernel on the grid."""
        X, Y = self.grid.meshgrid()
        r2 = X**2 + Y**2
        core = 1.0 + r2 / (2.0 * self.sigma.value**2 * self.gamma.value)
        k = core.pow(-self.gamma.value)
        return k / k.sum()

    def param_dict(self) -> dict[str, float]:
        """Return ``{"sigma": <value>, "gamma": <value>}``."""
        return {
            "sigma": float(self.sigma.value),
            "gamma": float(self.gamma.value),
        }

    @torch.no_grad()
    def to_dataarray(self) -> xr.DataArray:
        """Return the King kernel as a DataArray."""
        kernel = self.forward().detach().cpu().numpy()
        coords = self.grid.as_coords()
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=coords,
            attrs={
                "adjeff:kind": "analytical",
                "adjeff:model": "King",
                "adjeff:params": {
                    "sigma": float(self.sigma.value),
                    "gamma": float(self.gamma.value),
                },
                "band": self.band,
            },
        )


class MoffatGeneralizedPSF(nn.Module, PSFModule):
    """Trainable Generalised Moffat PSF kernel.

    The kernel follows ``(1 + (r/α)^{2β})^{-γ}``.  Setting *beta* = 1
    and *gamma* = β recovers the standard Moffat profile.

    Parameters
    ----------
    grid : PSFGrid
        Spatial sampling configuration.
    band : SensorBand
        Spectral band this PSF applies to.
    alpha : float
        Initial scale radius [km].
    beta : float
        Initial shape exponent controlling the power-law slope.
    gamma : float, optional
        Initial outer power-law index, by default 1.0.
    """

    def __init__(
        self,
        grid: PSFGrid,
        band: SensorBand,
        alpha: float,
        beta: float,
        gamma: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        self.grid = grid
        self.band = band

        self.alpha = ConstrainedParameter(
            init_value=torch.tensor(alpha, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-3,
            max_val=50.0,
            name="alpha",
        )

        self.beta = ConstrainedParameter(
            init_value=torch.tensor(beta, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-3,
            max_val=50.0,
            name="beta",
        )

        self.gamma = ConstrainedParameter(
            init_value=torch.tensor(gamma, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-3,
            max_val=50.0,
            name="gamma",
        )

    def forward(self) -> torch.Tensor:
        """Return normalised Generalised Moffat kernel on the grid."""
        X, Y = self.grid.meshgrid()
        r = torch.sqrt(X**2 + Y**2)
        k = (1.0 + (r / self.alpha.value).pow(2 * self.beta.value)).pow(
            -self.gamma.value
        )
        return k / k.sum()

    def param_dict(self) -> dict[str, float]:
        """Return ``{"alpha": <value>, "beta": <value>, "gamma": <value>}``."""
        return {
            "alpha": float(self.alpha.value),
            "beta": float(self.beta.value),
            "gamma": float(self.gamma.value),
        }

    @torch.no_grad()
    def to_dataarray(self) -> xr.DataArray:
        """Return the Moffat Generalized kernel as a DataArray."""
        kernel = self.forward().detach().cpu().numpy()
        coords = self.grid.as_coords()
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=coords,
            attrs={
                "adjeff:kind": "analytical",
                "adjeff:model": "MoffatGeneralized",
                "adjeff:params": {
                    "alpha": float(self.alpha.value),
                    "beta": float(self.beta.value),
                    "gamma": float(self.gamma.value),
                },
                "band": self.band,
            },
        )
