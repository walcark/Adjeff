"""Implement analytical subclasses of PSFModule.

Each subclass is a trainable PSFModule with ConstrainedParameter fields.
The common to_dataarray(), grid/band init, and nn.Module wiring are all
inherited from PSFModule — subclasses only declare their parameters,
forward(), and param_dict().
"""

from __future__ import annotations

from typing import ClassVar

import torch

from adjeff.utils import ConstrainedParameter, ExpTransform, SigmoidTransform

from ._psf import PSFGrid, PSFModule
from .bands import SensorBand


class GaussPSF(PSFModule):
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

    _model_name: ClassVar[str] = "Gaussian"

    def __init__(self, grid: PSFGrid, band: SensorBand, sigma: float) -> None:
        super().__init__(grid, band)
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
        return {"sigma": self.sigma.scalar}


class GeneralizedGaussianPSF(PSFModule):
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

    _model_name: ClassVar[str] = "GeneralizedGaussian"

    def __init__(self, grid: PSFGrid, band: SensorBand, sigma: float, n: float) -> None:
        super().__init__(grid, band)
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
        kernel = torch.exp(-((r / self.sigma.value) ** self.n.value))
        return kernel / kernel.sum()

    def param_dict(self) -> dict[str, float]:
        """Return ``{"sigma": <value>, "n": <value>}``."""
        return {"sigma": self.sigma.scalar, "n": self.n.scalar}


class VoigtPSF(PSFModule):
    """Trainable pseudo-Voigt PSF kernel.

    Linearly mixes a Gaussian (width *sigma*) and a Lorentzian (width
    *gamma*) using the Thompson et al. mixing parameter η.

    The Lorentzian falls as ``r^{-2}``, so ``∫ P(r) r dr`` diverges
    logarithmically on a plane whenever ``η > 0``: this kernel is always
    normalised against the grid it is evaluated on, and its encircled
    energy radii are properties of that grid rather than of the profile.

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

    _model_name: ClassVar[str] = "Voigt"

    _FWHM_GAUSS_FACTOR: float = 2.0 * (2.0 * 0.6931471805599453) ** 0.5

    def __init__(
        self,
        grid: PSFGrid,
        band: SensorBand,
        sigma: float,
        gamma: float,
    ) -> None:
        super().__init__(grid, band)
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
        FG = self._FWHM_GAUSS_FACTOR * self.sigma.value
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
            "sigma": self.sigma.scalar,
            "gamma": self.gamma.scalar,
        }


class KingPSF(PSFModule):
    """Trainable King profile PSF kernel.

    The kernel follows ``(1 + r² / (2σ²γ))^{-γ}``, producing a
    power-law tail in ``r^{-2γ}``.

    On a plane, ``∫ P(r) r dr`` converges only for ``γ > 1``: below that
    the kernel carries no scale of its own and the grid, rather than the
    physics, sets its normalisation.  At ``γ = 0.4`` on a 141 km grid,
    half of the energy sits beyond 79 km.  *gamma* is therefore bounded
    to :data:`GAMMA_BOUNDS` through a sigmoid rather than merely kept
    positive.  The region below the bound is also where the loss surface
    turns concave, which stalls a quasi-Newton step.

    Parameters
    ----------
    grid : PSFGrid
        Spatial sampling configuration.
    band : SensorBand
        Spectral band this PSF applies to.
    sigma : float
        Initial core width [km].
    gamma : float
        Initial power-law index.  Clamped into :data:`GAMMA_BOUNDS`.
    """

    #: Range the power-law index is confined to.  The lower end is the
    #: integrability threshold; the upper end is far above any value an
    #: atmospheric fit has produced, and only keeps the sigmoid on a
    #: usable slope.
    GAMMA_BOUNDS: ClassVar[tuple[float, float]] = (1.0, 5.0)

    _model_name: ClassVar[str] = "King"

    def __init__(
        self,
        grid: PSFGrid,
        band: SensorBand,
        sigma: float,
        gamma: float,
    ) -> None:
        super().__init__(grid, band)
        self.sigma = ConstrainedParameter(
            init_value=torch.tensor(sigma, dtype=torch.float32),
            transform=ExpTransform(),
            min_val=1e-3,
            max_val=1e2,
            name="sigma",
        )
        low, high = self.GAMMA_BOUNDS
        self.gamma = ConstrainedParameter(
            init_value=torch.tensor(
                min(max(gamma, low + 1e-3), high - 1e-3),
                dtype=torch.float32,
            ),
            transform=SigmoidTransform(low, high),
            min_val=low,
            max_val=high,
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
            "sigma": self.sigma.scalar,
            "gamma": self.gamma.scalar,
        }


class MoffatGeneralizedPSF(PSFModule):
    """Trainable Generalised Moffat PSF kernel.

    The kernel follows ``(1 + (r/α)^{2β})^{-γ}``.  Setting *beta* = 1
    and *gamma* = β recovers the standard Moffat profile.

    The tail falls as ``r^{-2βγ}``, so the profile has finite energy on a
    plane only for ``βγ > 1``.  That is a constraint on the product,
    which a per-parameter bound cannot express, so it is left to the
    caller: check it on the fitted values rather than assume it.

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

    _model_name: ClassVar[str] = "MoffatGeneralized"

    def __init__(
        self,
        grid: PSFGrid,
        band: SensorBand,
        alpha: float,
        beta: float,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(grid, band)
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
            "alpha": self.alpha.scalar,
            "beta": self.beta.scalar,
            "gamma": self.gamma.scalar,
        }
