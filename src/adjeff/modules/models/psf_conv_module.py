"""Generic base class for PSF-convolution scene modules."""

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
    derived from these two declarations, subclasses need not override either.

    Training and inference are two different inputs, not two modes of one
    object.  Pass *psfs* to optimise live :class:`PSFModule` objects, or
    *kernels* to apply a frozen PSF tree; exactly one of the two.

    Parameters
    ----------
    psfs : dict[SensorBand, PSFModule] or None
        Live PSF modules, registered for autograd.  Required for
        training, and the only form :meth:`forward_band` accepts.
    kernels : xr.DataTree or None
        Frozen PSF tree, as returned by
        :func:`~adjeff.core.psf_tree.freeze` or the optimiser.  Inference
        only.
    cache : CacheStore or None, optional
        Cache backend for the xarray inference path.
    device : torch.device or str, optional
        Device used for tensor convolutions (default ``"cuda"``).

    Raises
    ------
    ConfigurationError
        If neither or both of *psfs* and *kernels* are given.
    """

    _conv_input: ClassVar[str]
    _formula: ClassVar[Any]

    def __init__(
        self,
        psfs: dict[SensorBand, PSFModule] | None = None,
        kernels: xr.DataTree | None = None,
        cache: CacheStore | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        if (psfs is None) == (kernels is None):
            raise ConfigurationError(
                "Pass exactly one of `psfs` (live modules, for training) "
                "or `kernels` (a frozen PSF tree, for inference)."
            )
        super().__init__(cache=cache)
        self._device = torch.device(device)
        self._kernels = kernels
        self._psfs: nn.ModuleDict = nn.ModuleDict(
            {b.id: cast(nn.Module, m) for b, m in (psfs or {}).items()}
        )

    # ------------------------------------------------------------------
    # TrainableSceneModule interface
    # ------------------------------------------------------------------

    @property
    def is_trainable(self) -> bool:
        """Return True when this model holds live PSF modules."""
        return self._kernels is None

    @property
    def psf_modules(self) -> dict[str, PSFModule]:
        """Mapping of band IDs to PSF modules (training mode only)."""
        return {k: cast(PSFModule, v) for k, v in self._psfs.items()}

    def psf_params(self, band: SensorBand) -> dict[str, float]:
        """Return the current PSF parameters of *band*.

        Empty when the PSF has no parameters, e.g. a kernel loaded from a
        frozen tree or a purely numerical PSF.

        Parameters
        ----------
        band : SensorBand
            Band whose PSF module is read.

        Returns
        -------
        dict[str, float]
            Parameter name to value, as held by the module right now.
        """
        if self._kernels is not None:
            return {}
        if band.id not in self._psfs:
            held = ", ".join(sorted(self._psfs)) or "none"
            raise KeyError(f"No PSF for band {band.id!r}; holds: {held}.")
        return cast(PSFModule, self._psfs[band.id]).param_dict()

    def to_psf_tree(self) -> xr.DataTree:
        """Export the current kernels to a frozen PSF tree.

        Returns
        -------
        xr.DataTree
            One group per band.  In inference mode the tree the model was
            built with is returned unchanged.
        """
        if self._kernels is not None:
            return self._kernels
        modules = [cast(PSFModule, m) for m in self._psfs.values()]
        return freeze({m.band: m for m in modules})

    def forward_band(
        self,
        band: SensorBand,
        *,
        kernel: torch.Tensor | None = None,
        **inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable per-band forward pass (2-D tensors, autograd).

        Only available in training mode.

        Parameters
        ----------
        band : SensorBand
            Band to run.
        kernel : torch.Tensor or None, optional
            Kernel to convolve with, overriding the band's own PSF.
            Used to evaluate a candidate without installing it in the
            model, which is what mapping a loss surface amounts to.
        **inputs : torch.Tensor
            The variables named in ``required_vars``.
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

    # ------------------------------------------------------------------
    # SceneModule interface
    # ------------------------------------------------------------------

    def _kernel_for(self, band: SensorBand) -> xr.DataArray:
        """Return the kernel to convolve with for *band*."""
        if self._kernels is not None:
            return psf_kernel(self._kernels, band)
        return self.psf_modules[band.id].to_dataarray()

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Xarray inference, extra dims handled by broadcasting."""
        for band in scene.bands:
            ds = scene[band]
            rho_env = fft_convolve_2D(
                ds[self._conv_input].compute(),
                self._kernel_for(band),
                padding="reflect",
                conv_type="same",
                device=self._device,
            )
            ds[self.output_vars[0]] = self._formula(
                **{k: ds[k].compute() for k in self.required_vars},
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
