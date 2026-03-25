"""Implement the non-analytical subclass of PSFModule.

The subclass implements the PSFModule protocol (forward + to_dataarray).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from ._psf import PSFGrid, PSFModule
from .bands import SensorBand


class NonAnalyticalPSF(nn.Module, PSFModule):
    """PSF defined by a fixed kernel tensor derived from a Smart-G output.

    Parameters do not require gradients — this PSF is not trained.
    Useful for physics-based kernels produced by radiative transfer
    simulation rather than fitted to data.

    Parameters
    ----------
    grid : PSFGrid
        Spatial sampling configuration.
    band : SensorBand
        Band identifier this PSF applies to.
    kernel :
        2-D array of shape ``(n, n)``.  Will be normalised
        to sum to 1.  Accepts ``np.ndarray`` or ``torch.Tensor``.
    source:
        Optional provenance tag stored in ``adjeff:source`` attribute.
    """

    def __init__(
        self,
        grid: PSFGrid,
        band: SensorBand,
        kernel: np.ndarray | torch.Tensor,
        source: str = "SmartG",
    ) -> None:
        nn.Module.__init__(self)
        self.grid = grid
        self.band = band
        self._source = source
        self._kernel: torch.Tensor

        if isinstance(kernel, np.ndarray):
            k = torch.tensor(kernel, dtype=torch.float32)
        else:
            k = kernel.float()

        if k.shape != (grid.n, grid.n):
            from adjeff.exceptions import ConfigurationError

            raise ConfigurationError(
                f"NonAnalyticalPSF kernel shape {tuple(k.shape)} "
                f"does not match PSFGrid ({grid.n}, {grid.n})."
            )

        k = k / k.sum()
        self.register_buffer("_kernel", k)

    def forward(self) -> torch.Tensor:
        """Return the fixed normalised kernel (no gradient)."""
        return self._kernel

    @torch.no_grad()
    def to_dataarray(self) -> xr.DataArray:
        """Return the PSF DataArray with adapted attrs."""
        kernel = self._kernel.cpu().numpy()
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=self.grid.as_coords(),
            attrs={
                "adjeff:kind": "non_analytical",
                "adjeff:source": self._source,
                "band": self.band,
            },
        )
