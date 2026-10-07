"""Loss metrics on tensors, for the training loop.

Every metric takes ``(prediction, reference, dists, mask)`` and works on
residuals divided by ``max(|reference|)``.

Functions
---------
    mae, mse, rmse
        Uniformly weighted over the domain.
    mae_rad, mse_rad, rmse_rad
        Weighted so that every radius carries the same total weight.

Classes
-------
    Metric
        Enum of the six metrics, each member callable.
"""

from enum import Enum
from typing import Callable

import torch

from adjeff.utils.torchutils import radial_mask, radial_weights

_MASK_THRESHOLD = 0.99

MetricFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    torch.Tensor,
]


def mae(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dists: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """MAE on scale-normalised residuals, over the given domain."""
    resid = _residual(tensor1, tensor2)
    w = _flat_weights(dists, mask_tensor)
    return (w * resid.abs()).sum() / w.sum()


def mse(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dists: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on scale-normalised residuals, over the given domain."""
    resid = _residual(tensor1, tensor2)
    w = _flat_weights(dists, mask_tensor)
    return (w * resid.pow(2)).sum() / w.sum()


def rmse(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dists: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSE on scale-normalised residuals, over the given domain."""
    return torch.sqrt(mse(tensor1, tensor2, dists, mask_tensor))


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


def _get_scale(target: torch.Tensor) -> torch.Tensor:
    """Return ``max(|target|)``, the amplitude residuals are divided by.

    The reference alone sets it, so that the loss neither moves with the
    prediction nor depends on pixels outside the mask.
    """
    return target.abs().max().clamp(min=torch.finfo(target.dtype).eps)


def _residual(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Return the prediction error, in units of the reference amplitude."""
    return (tensor1 - tensor2) / _get_scale(tensor2)


def _domain(dists: torch.Tensor, mask_tensor: torch.Tensor | None) -> torch.Tensor:
    """Return the 0/1 domain *mask_tensor* stands for.

    ``None`` is everything, a boolean tensor is the domain itself, and a float
    field keeps the pixels within its 99 % radial energy. The domain always
    comes from the caller: deriving it from the residual let the optimiser
    shrink its own mask instead of fitting.
    """
    if mask_tensor is None:
        return torch.ones_like(dists)
    if mask_tensor.dtype == torch.bool:
        return mask_tensor.float()
    return radial_mask(mask_tensor, dists, _MASK_THRESHOLD).float()


def _flat_weights(
    dists: torch.Tensor, mask_tensor: torch.Tensor | None
) -> torch.Tensor:
    """Return uniform weights over the domain."""
    return _domain(dists, mask_tensor)


def _rad_weights(dists: torch.Tensor, mask_tensor: torch.Tensor | None) -> torch.Tensor:
    """Return the radial weights, restricted to *mask_tensor*."""
    return radial_weights(dists) * _domain(dists, mask_tensor)


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
