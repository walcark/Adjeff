"""Fixed PSF kernel, typically from a radiative transfer simulation.

Classes
-------
    NonAnalyticalPSF
        Non-trainable PSF holding a given kernel.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
import xarray as xr

from ._psf import PSFGrid, PSFModule
from .bands import SensorBand


class NonAnalyticalPSF(PSFModule):
    """Non-trainable PSF holding a fixed kernel.

    Parameters
    ----------
    grid : PSFGrid
        Sampling grid.
    band : SensorBand
        Band the PSF applies to.
    kernel : np.ndarray or torch.Tensor
        Kernel of shape ``(grid.n, grid.n)``, normalised to sum to 1.
    source : str, optional
        Provenance stored in ``adjeff:source``, ``"SmartG"`` by default.
    """

    _model_name: ClassVar[str] = "NonAnalytical"

    def __init__(
        self,
        grid: PSFGrid,
        band: SensorBand,
        kernel: np.ndarray | torch.Tensor,
        source: str = "SmartG",
    ) -> None:
        super().__init__(grid, band)
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
        """Return the PSF DataArray with non-analytical attributes."""
        kernel = self._kernel.cpu().numpy()
        return xr.DataArray(
            kernel,
            dims=["y_psf", "x_psf"],
            coords=self.grid.as_coords(),
            attrs={
                "adjeff:kind": "non_analytical",
                "adjeff:source": self._source,
                # The id, not the enum: see PSFModule.to_dataarray.
                "band": self.band.id,
            },
        )
