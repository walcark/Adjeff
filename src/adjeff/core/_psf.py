"""Base class and grid for Point Spread Functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
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
            raise ConfigurationError(
                f"PSFGrid.res must be > 0, got {self.res}."
            )
        if (self.n < 3) or (self.n % 2 == 0):
            raise ConfigurationError(
                f"PSFGrid.n must be odd and ≥ 3, got {self.n}."
            )

    def as_coords(self) -> xr.Coordinates:
        """Return centered xarray coordinates for the PSF grid."""
        half = (self.n // 2) * self.res
        coords = np.linspace(-half, half, self.n)
        return xr.Coordinates({"x_psf": coords, "y_psf": coords})

    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (X, Y) float32 meshgrid tensors centered on the grid."""
        half = (self.n // 2) * self.res
        t = torch.linspace(-half, half, self.n, dtype=torch.float32)
        X, Y = torch.meshgrid(t, t, indexing="xy")
        return X, Y


class PSFModule:
    """Abstract base for all PSF nn.Module subclasses.

    Subclasses must be torch.nn.Module subclasses. They must implement
    the forward() method based on their parameters, and return a 2D
    kernel.

    Parameters
    ----------
    grid : PSFGrid
        The grid to use for the coordinates of the PSF.
    band : SensorBand
        The spectral band of interest for the PSF.
    """

    grid: PSFGrid
    band: SensorBand

    def forward(self) -> "torch.Tensor":
        """Return normalised 2D PSF kernel."""
        raise NotImplementedError

    @torch.no_grad()
    def to_dataarray(self) -> xr.DataArray:
        """Return the kernel as DataArray."""
        kernel = self.forward().detach().cpu().numpy()
        coords = self.grid.as_coords()
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=coords,
            attrs={
                "atcor:kind": "analytical",
                "band": self.band,
            },
        )
