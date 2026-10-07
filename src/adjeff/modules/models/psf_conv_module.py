"""Base class of the models applying a per-band PSF convolution.

Classes
-------
    PSFConvModule
        Holds live PSFs (training) or a frozen PSF tree (inference).

Functions
---------
    psf_kernel_values
        Kernel of one band in a PSF tree, as an array.
"""

from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
import xarray as xr

from adjeff.core import ImageDict, SensorBand
from adjeff.core._psf import PSFModule
from adjeff.core.psf_tree import freeze, psf_kernel
from adjeff.exceptions import ConfigurationError
from adjeff.utils import CacheStore, fft_convolve_2D, fft_convolve_2D_torch

from ..scene_module import TrainableSceneModule


class PSFConvModule(TrainableSceneModule):
    """Base of the models convolving one variable with a per-band PSF.

    Subclasses declare ``_conv_input``, the variable convolved, and
    ``_formula``, a staticmethod taking the required variables and
    ``rho_env`` (the convolution) and returning the output.  Both the
    xarray path (:meth:`_compute`) and the tensor path
    (:meth:`forward_band`) follow from them.

    Parameters
    ----------
    psfs : dict[SensorBand, PSFModule] or None
        Live PSFs, for training.
    kernels : xr.DataTree or None
        Frozen PSF tree, for inference.  Exactly one of the two.
    cache, rename : optional
        See :class:`SceneModule`.
    device : torch.device or str, optional
        Device of the convolutions, ``"cuda"`` by default.

    Raises
    ------
    ConfigurationError
        Unless exactly one of *psfs* and *kernels* is given.
    """

    _conv_input: ClassVar[str]
    _formula: ClassVar[Any]

    def __init__(
        self,
        psfs: dict[SensorBand, PSFModule] | None = None,
        kernels: xr.DataTree | None = None,
        cache: CacheStore | None = None,
        device: torch.device | str = "cuda",
        rename: dict[str, str] | None = None,
    ) -> None:
        if (psfs is None) == (kernels is None):
            raise ConfigurationError(
                "Pass exactly one of `psfs` (live modules, for training) "
                "or `kernels` (a frozen PSF tree, for inference)."
            )
        super().__init__(cache=cache, rename=rename)
        self._device = torch.device(device)
        self._kernels = kernels
        self._psfs: nn.ModuleDict = nn.ModuleDict(
            {b.id: cast(nn.Module, m) for b, m in (psfs or {}).items()}
        )

    @property
    def psf_modules(self) -> dict[str, PSFModule]:
        """Mapping of band IDs to PSF modules (training mode only)."""
        return {k: cast(PSFModule, v) for k, v in self._psfs.items()}

    def psf_params(self, band: SensorBand) -> dict[str, float]:
        """Return the current PSF parameters of *band*, ``{}`` if none.

        Raises
        ------
        KeyError
            If no PSF is held for *band*.
        """
        if self._kernels is not None:
            return {}
        if band.id not in self._psfs:
            held = ", ".join(sorted(self._psfs)) or "none"
            raise KeyError(f"No PSF for band {band.id!r}; holds: {held}.")
        return cast(PSFModule, self._psfs[band.id]).param_dict()

    def forward_band(
        self,
        band: SensorBand,
        *,
        kernel: torch.Tensor | None = None,
        **inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable forward pass of one band, on 2-D tensors.

        *kernel* (when given) replaces the band's live PSF for this call.
        """
        d = self._device
        if kernel is None:
            kernel = self.psf_modules[band.id].forward()
        kernel = kernel.to(d)
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

    def _kernel_for(self, band: SensorBand) -> xr.DataArray:
        """Return the kernel to convolve with for *band*."""
        if self._kernels is not None:
            return psf_kernel(self._kernels, band)
        return self.psf_modules[band.id].to_dataarray()

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Apply the model on DataArrays, extra dims broadcast."""
        for band in scene.bands:
            ds = scene[band]
            source = ds[self._slot(self._conv_input)]
            self._log.debug(
                "psf.convolve",
                band=str(band),
                shape=tuple(source.sizes.values()),
                device=self._device,
            )
            rho_env = fft_convolve_2D(
                source.compute(),
                self._kernel_for(band),
                padding="reflect",
                conv_type="same",
                device=self._device,
            )
            # The formula is written in roles, so the slots are resolved
            # on the way in and on the way out, never inside it.
            ds[self.output_vars[0]] = self._formula(
                **{
                    role: ds[self._slot(role)].compute() for role in self._required_vars
                },
                rho_env=rho_env,
            )
        return scene

    def _config_dict(self) -> dict[str, object]:
        """Override to hash PSF kernel arrays instead of module attributes."""
        if self._kernels is not None:
            return {
                band_id: psf_kernel_values(self._kernels, band_id)
                for band_id in sorted(self._kernels.children)
            }
        return {
            band_id: cast(PSFModule, psf).to_dataarray().values
            for band_id, psf in self._psfs.items()
        }


def psf_kernel_values(tree: xr.DataTree, band_id: str) -> Any:
    """Return the raw kernel array stored under *band_id*."""
    return tree[band_id].ds["kernel"].values
