"""Generic base class for PSF-convolution scene modules."""

from typing import Any, ClassVar, cast

import torch
import torch.nn as nn

from adjeff.core import ImageDict, SensorBand
from adjeff.core._psf import PSFModule
from adjeff.utils import CacheStore, fft_convolve_2D, fft_convolve_2D_torch

from ..scene_module import TrainableSceneModule


class PSFConvModule(TrainableSceneModule):
    """Abstract base for modules applying one PSF convolution + a formula.

    Subclasses declare two class attributes:

    - ``_conv_input``: name of the variable in the scene dataset to convolve
      with the PSF kernel (e.g. ``"rho_unif"``).
    - ``_formula``: callable that receives all ``required_vars`` as keyword
      arguments plus ``rho_env`` (the convolution output) and returns the
      module output.  Assign a :func:`staticmethod` so that ``self._formula``
      does not receive ``self``.

    ``_compute`` (xarray inference, handles extra dims via broadcasting) and
    ``forward_band`` (2-D tensor training, autograd preserved) are both fully
    derived from these two declarations — subclasses need not override either.

    Parameters
    ----------
    psfs : list[PSFModule]
        One PSFModule (``nn.Module`` subclass) per band.
    cache : CacheStore or None, optional
        Cache backend for the xarray inference path.
    device : torch.device or str, optional
        Device used for tensor convolutions (default ``"cuda"``).
    """

    _conv_input: ClassVar[str]
    _formula: ClassVar[Any]

    def __init__(
        self,
        psfs: list[PSFModule],
        cache: CacheStore | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__(cache=cache)
        self._device = torch.device(device)
        self._psfs: nn.ModuleDict = nn.ModuleDict(
            {psf.band.id: psf for psf in psfs}  # type: ignore[misc]
        )

    # ------------------------------------------------------------------
    # TrainableSceneModule interface
    # ------------------------------------------------------------------

    @property
    def psf_modules(self) -> dict[str, PSFModule]:
        """Mapping of band IDs to PSF modules."""
        return {k: cast(PSFModule, v) for k, v in self._psfs.items()}

    def forward_band(
        self, band: SensorBand, **inputs: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable per-band forward pass (2-D tensors, autograd)."""
        d = self._device
        kernel = self.psf_modules[band.id].forward().to(d)
        rho_env = fft_convolve_2D_torch(
            inputs[self._conv_input].to(d),
            kernel,
            padding="reflect",
            conv_type="same",
        )
        return self._formula(  # type: ignore[no-any-return]
            **{k: v.to(d) for k, v in inputs.items()},
            rho_env=rho_env,
        )

    # ------------------------------------------------------------------
    # SceneModule interface
    # ------------------------------------------------------------------

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Xarray inference — extra dims handled by broadcasting."""
        for band in scene.bands:
            ds = scene[band]
            kernel_da = self.psf_modules[band.id].to_dataarray()
            rho_env = fft_convolve_2D(
                ds[self._conv_input],
                kernel_da,
                padding="reflect",
                conv_type="same",
                device=self._device,
            )
            ds[self.output_vars[0]] = self._formula(
                **{k: ds[k] for k in self.required_vars},
                rho_env=rho_env,
            )
        return scene

    def _config_dict(self) -> dict[str, object]:
        return {
            band_id: psf.to_dataarray().values  # type: ignore[operator]
            for band_id, psf in self._psfs.items()
        }
