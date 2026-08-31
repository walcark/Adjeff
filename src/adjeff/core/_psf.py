"""Base class and grid for Point Spread Functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from .bands import SensorBand


@dataclass(frozen=True)
class PSFGrid:
    """Spatial sampling configuration for a PSF.

    Parameters
    ----------
    res : float
        Pixel size in km. Must be > 0.
    n : int
        Number of pixels per side of the square 2-D grid.
        Must be odd and ≥ 3.
    """

    res: float
    n: int

    def __post_init__(self) -> None:
        """Ensure that the PSF grid is valid."""
        from adjeff.exceptions import ConfigurationError

        if self.res <= 0:
            raise ConfigurationError(f"PSFGrid.res must be > 0, got {self.res}.")
        if (self.n < 3) or (self.n % 2 == 0):
            raise ConfigurationError(f"PSFGrid.n must be odd and ≥ 3, got {self.n}.")

    def as_coords(self) -> xr.Coordinates:
        """Return centered xarray coordinates for the PSF grid."""
        half = (self.n // 2) * self.res
        coords = np.linspace(-half, half, self.n)
        return xr.Coordinates({"x_psf": coords, "y_psf": coords})

    def meshgrid(
        self, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (X, Y) float32 meshgrid tensors centered on the grid.

        Parameters
        ----------
        device : torch.device or str or None, optional
            Where to allocate the tensors.  ``None`` keeps the torch
            default, which is the CPU.
        """
        half = (self.n // 2) * self.res
        t = torch.linspace(-half, half, self.n, dtype=torch.float32, device=device)
        X, Y = torch.meshgrid(t, t, indexing="xy")
        return X, Y


def radial_power(
    r: torch.Tensor, scale: torch.Tensor, power: torch.Tensor
) -> torch.Tensor:
    """Return ``(r / scale) ** power``, differentiable at ``r = 0``.

    Written plainly, this expression is finite at the origin but its
    derivative is not computable there.  Differentiating with respect to
    *scale* brings out a factor ``(r / scale) ** (power - 1)``, and a
    *power* below one makes that ``0`` raised to a negative exponent,
    which is ``inf``.  The chain rule then multiplies it by ``r / scale**2``,
    which is ``0``, and IEEE 754 answers ``NaN``.  The limit exists and is
    zero; autograd never computes limits, only products.

    A grid whose centre pixel lands exactly on ``r = 0`` therefore poisons
    the whole gradient, one pixel out of millions being enough, and every
    parameter of the model becomes ``NaN`` on the next optimiser step.
    Whether it happens is decided by floating-point rounding: a
    401-pixel grid at 0.5 km lands on zero, a 1999-pixel grid at 0.1 km
    misses it by 5e-08.

    The origin is evaluated on a stand-in radius and then discarded, so
    that no infinity is ever created.  Discarding the *result* would not
    be enough: the gradient of a value that is overwritten is zero, and
    ``0 * NaN`` is still ``NaN``.

    Parameters
    ----------
    r : torch.Tensor
        Radial distances, non-negative.
    scale : torch.Tensor
        Length the radii are measured in.
    power : torch.Tensor
        Exponent.  Values below one are what make this necessary.

    Returns
    -------
    torch.Tensor
        ``(r / scale) ** power``, with ``0`` at the origin and a
        gradient of zero there, which is the limit.
    """
    positive = r > 0
    stand_in = torch.where(positive, r, torch.ones_like(r))
    powered = (stand_in / scale) ** power
    return torch.where(positive, powered, torch.zeros_like(powered))


class PSFModule(nn.Module, ABC):
    r"""Abstract base for all PSF models.

    Subclasses must define:

    - ``_model_name``: display name stored in DataArray ``adjeff:model`` attr.
    - :meth:`forward`: return a normalised 2-D kernel tensor.

    :meth:`param_dict` returns ``{}`` by default; override in parametric
    subclasses to expose current parameter values.

    :meth:`to_dataarray` is fully implemented here using :meth:`forward` and
    :meth:`param_dict` — subclasses only override it when the attrs layout
    differs (e.g. :class:`~adjeff.core.NonAnalyticalPSF`).

    Notes
    -----
    Every kernel here is sampled at the centre of each pixel, which is
    the midpoint approximation of what a discrete convolution actually
    needs.  Writing the continuous convolution over an image that is
    constant per pixel gives

    .. math::

        \rho_{env}(x_i) = \sum_j \rho_{unif}(x_j)
                          \int_{\mathrm{cell}\ j} P(x_i - x')\,dx',

    so a tap *is* the integral of the profile over one cell, and the
    value at the cell's centre only stands in for it.  That stand-in is
    excellent wherever the profile is close to linear across a pixel: at
    one pixel from the centre it is already within 0.4 % for a
    generalised Gaussian at ``n = 0.2``, and within 0.2 % beyond.

    It is not excellent at the centre pixel, where a peaked profile
    varies by orders of magnitude across one cell.  Measured against the
    true cell average, the centre tap is 8 % too high for a Gaussian of
    ``sigma = 0.33 km`` on a 0.1 km grid, and 550 % too high for a King
    profile of ``sigma = 0.01 km`` on the same grid.  The criterion is
    not the family of the kernel but how much of its energy falls inside
    one pixel.

    Integrating the profile over the central pixels rather than sampling
    it there would remove this, at a measured cost of under 2 % of a
    training step: sub-sampling a 33x33 patch at 16x16 and the centre
    pixel alone at 256x256 lands every tap within 0.5 % of its cell
    average, even in the sharpest regime.  It is not done, for a reason
    worth stating: the fitted parameters absorb the bias, so a kernel
    that is more faithful to the continuous physics does not
    automatically fit this discrete problem better.  Deciding it needs a
    measurement, not an argument: fit at two resolutions with and
    without, and compare the loss reached and how far the parameters
    move between grids.

    What was fixed rather than argued about is the gradient, which used
    to be ``NaN`` outright on some grids.  See :func:`radial_power`.

    Parameters
    ----------
    grid : PSFGrid
        Spatial sampling configuration.
    band : SensorBand
        Spectral band this PSF applies to.
    """

    _model_name: ClassVar[str] = ""

    _r2: torch.Tensor

    def __init__(self, grid: PSFGrid, band: SensorBand) -> None:
        super().__init__()
        self.grid = grid
        self.band = band
        # The radial grid never changes and carries no gradient, yet
        # forward() used to rebuild it on every call, on whatever device
        # torch defaults to.  Measured at n = 3999: 268 ms of CPU per
        # evaluation on one core, against 1 ms once it sits on the GPU,
        # and the optimiser pays it once per training landscape per step.
        # Held as a buffer so that `.to(device)` moves it along with the
        # parameters, and non-persistent because it is derived from
        # `grid` rather than learned.
        X, Y = grid.meshgrid()
        self.register_buffer("_r2", X * X + Y * Y, persistent=False)

    @property
    def r2(self) -> torch.Tensor:
        """Squared radial distance of each pixel, on the module's device."""
        return self._r2

    @property
    def r(self) -> torch.Tensor:
        """Radial distance of each pixel, on the module's device."""
        return torch.sqrt(self._r2)

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Return normalised 2D PSF kernel."""

    def param_dict(self) -> dict[str, float]:
        """Return current parameter values as a plain ``{name: value}`` dict.

        Returns an empty dict for non-parametric PSFs (e.g.
        :class:`~adjeff.core.NonAnalyticalPSF`).
        Analytical subclasses override this method.
        """
        return {}

    @torch.no_grad()
    def to_dataarray(self) -> xr.DataArray:
        """Return the kernel as a DataArray with metadata attrs."""
        kernel = self.forward().detach().cpu().numpy()
        coords = self.grid.as_coords()
        params = self.param_dict()
        attrs: dict[str, object] = {
            "adjeff:kind": "analytical",
            "adjeff:model": self._model_name,
            # The id, not the enum: attrs have to survive a zarr write,
            # and a SensorBand is not JSON serialisable.
            "band": self.band.id,
        }
        if params:
            attrs["adjeff:params"] = params
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=coords,
            attrs=attrs,
        )
