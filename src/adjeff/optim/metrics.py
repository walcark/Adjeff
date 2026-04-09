"""Metrics and Metric enum for PSF optimisation."""

from enum import Enum
from typing import Callable

import torch

from adjeff.utils.torchutils import radial_mask, radial_weights

_MASK_THRESHOLD = 0.99

MetricFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    torch.Tensor,
]

# ---------------------------------------------------------------------------
# Public metric functions
# All share the same signature so Metric.__call__ can dispatch uniformly.
# Non-RAD metrics ignore *mask_tensor*; RAD metrics use it when provided.
# ---------------------------------------------------------------------------


def mae(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dists: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """MAE on scale-normalised residuals, masked to 99% radial CDF."""
    resid = _residual(tensor1, tensor2)
    w = radial_mask(resid, dists, _MASK_THRESHOLD).float()
    return (w * resid.abs()).sum() / w.sum()


def mse(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dists: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on scale-normalised residuals, masked to 99% radial CDF."""
    resid = _residual(tensor1, tensor2)
    w = radial_mask(resid, dists, _MASK_THRESHOLD).float()
    return (w * resid.pow(2)).sum() / w.sum()


def rmse(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dists: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSE on scale-normalised residuals, masked to 99% radial CDF."""
    return torch.sqrt(mse(tensor1, tensor2, dists))


def mae_rad(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dists: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Radially weighted MAE. Optional CDF mask driven by *mask_tensor*."""
    resid = _residual(tensor1, tensor2)
    w = _rad_weights(dists, mask_tensor)
    return (w * resid.abs()).sum() / w.sum()


def mse_rad(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dists: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Radially weighted MSE. Optional CDF mask driven by *mask_tensor*."""
    resid = _residual(tensor1, tensor2)
    w = _rad_weights(dists, mask_tensor)
    return (w * resid.pow(2)).sum() / w.sum()


def rmse_rad(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dists: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Radially weighted RMSE. Optional CDF mask driven by *mask_tensor*."""
    return torch.sqrt(mse_rad(tensor1, tensor2, dists, mask_tensor))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_scale(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    scale1 = tensor1.abs().max().clamp(min=torch.finfo(tensor1.dtype).eps)
    scale2 = tensor2.abs().max().clamp(min=torch.finfo(tensor2.dtype).eps)
    return torch.maximum(scale1, scale2)


def _residual(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    return (tensor1 - tensor2) / _get_scale(tensor1, tensor2)


def _rad_weights(
    dists: torch.Tensor, mask_tensor: torch.Tensor | None
) -> torch.Tensor:
    w = radial_weights(dists)
    if mask_tensor is not None:
        w = w * radial_mask(mask_tensor, dists, _MASK_THRESHOLD).float()
    return w


# ---------------------------------------------------------------------------
# Metric enum — defined after functions so members can reference them
# ---------------------------------------------------------------------------


class Metric(Enum):
    """Available loss metrics for PSF optimisation.

    Each member is directly callable with the same signature as the
    underlying metric function::

        Metric.RMSE_RAD(tensor1, tensor2, dists, mask_tensor)
    """

    MAE = (mae,)
    MSE = (mse,)
    RMSE = (rmse,)
    MAE_RAD = (mae_rad,)
    MSE_RAD = (mse_rad,)
    RMSE_RAD = (rmse_rad,)

    def __call__(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        dists: torch.Tensor,
        mask_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to the underlying metric function."""
        fn: MetricFn = self.value[0]
        return fn(tensor1, tensor2, dists, mask_tensor)
