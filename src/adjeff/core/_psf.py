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


class PSFModule(nn.Module, ABC):
    """Abstract base for all PSF models.

    Subclasses must define:

    - ``_model_name``: display name stored in DataArray ``adjeff:model`` attr.
    - :meth:`forward`: return a normalised 2-D kernel tensor.

    :meth:`param_dict` returns ``{}`` by default; override in parametric
    subclasses to expose current parameter values.

    :meth:`to_dataarray` is fully implemented here using :meth:`forward` and
    :meth:`param_dict` — subclasses only override it when the attrs layout
    differs (e.g. :class:`~adjeff.core.NonAnalyticalPSF`).

    Parameters
    ----------
    grid : PSFGrid
        Spatial sampling configuration.
    band : SensorBand
        Spectral band this PSF applies to.
    """

    _model_name: ClassVar[str] = ""

    def __init__(self, grid: PSFGrid, band: SensorBand) -> None:
        super().__init__()
        self.grid = grid
        self.band = band

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
            "band": self.band,
        }
        if params:
            attrs["adjeff:params"] = params
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=coords,
            attrs=attrs,
        )
