"""Base class and sampling grid of every PSF.

Classes
-------
    PSFGrid
        Square, odd-sized grid a PSF is sampled on.
    PSFModule
        Abstract trainable PSF: ``forward`` returns the normalised kernel.

Functions
---------
    radial_power
        ``(r / scale) ** power`` with a finite gradient at ``r = 0``.
"""

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
    """Return ``(r / scale) ** power``, with a zero gradient at ``r = 0``.

    Written plainly, the gradient at the origin is ``inf * 0 = NaN`` when
    ``power < 1``, and one such pixel turns every parameter into ``NaN``.
    The origin is evaluated on a non-zero radius instead, then set to 0.
    """
    positive = r > 0
    stand_in = torch.where(positive, r, torch.ones_like(r))
    powered = (stand_in / scale) ** power
    return torch.where(positive, powered, torch.zeros_like(powered))


class PSFModule(nn.Module, ABC):
    """Abstract trainable PSF.

    Subclasses set ``_model_name`` and implement :meth:`forward`;
    parametric ones also override :meth:`param_dict`.

    Notes
    -----
    Kernels are sampled at pixel centres rather than integrated over each
    pixel. This overestimates the centre tap of a sharply peaked kernel
    (8 % for a Gaussian of 0.33 km on a 0.1 km grid), a bias the fitted
    parameters absorb.

    Parameters
    ----------
    grid : PSFGrid
        Sampling grid.
    band : SensorBand
        Band the PSF applies to.
    """

    model_name: ClassVar[str] = ""

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
