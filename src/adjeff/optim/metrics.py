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
    """Return the amplitude the residual is expressed against.

    The reference alone sets it.  Taking the larger of the two, as this
    did, made the metric depend on the prediction in two unwanted ways:
    the scale moved while the optimiser searched, and a masked metric
    stopped being a function of its own mask.  A prediction that goes
    wrong far outside the mask raised the scale for everyone, and the
    error measured on unchanged pixels inside the mask fell by two
    orders of magnitude.
    """
    return target.abs().max().clamp(min=torch.finfo(target.dtype).eps)


def _residual(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Return the prediction error, in units of the reference amplitude."""
    return (tensor1 - tensor2) / _get_scale(tensor2)


def _domain(
    dists: torch.Tensor, mask_tensor: torch.Tensor | None
) -> torch.Tensor:
    """Return the 0/1 domain *mask_tensor* stands for.

    A float field is a *source*: the domain is the pixels within its 99%
    radial energy, which is how ``rho_unif`` has always been used here.
    A boolean tensor is the domain itself, already decided by the caller,
    which is what a fixed radius amounts to.
    """
    if mask_tensor is None:
        return torch.ones_like(dists)
    if mask_tensor.dtype == torch.bool:
        return mask_tensor.float()
    return radial_mask(mask_tensor, dists, _MASK_THRESHOLD).float()


def _flat_weights(
    dists: torch.Tensor, mask_tensor: torch.Tensor | None
) -> torch.Tensor:
    """Return uniform weights over the domain.

    These metrics used to derive their domain from the residual they
    were measuring, which let the optimiser lower the loss by shrinking
    its own mask rather than by fitting better.  Measured on the
    manuscript's landscapes, that collapses the fit: the King core width
    falls to a thirtieth of its value and the generalisation error grows
    by a factor 2.6.  The domain now comes from the caller, like it does
    for the radially weighted metrics.
    """
    return _domain(dists, mask_tensor)


def _rad_weights(
    dists: torch.Tensor, mask_tensor: torch.Tensor | None
) -> torch.Tensor:
    """Return the radial weights, restricted to *mask_tensor*.

    Two kinds of mask are accepted, told apart by their dtype.  A float
    field is a *source*: the mask keeps the pixels within its 99% radial
    energy, which is how ``rho_unif`` has always been used here.  A
    boolean tensor is the mask itself, already decided by the caller,
    which is what a fixed radius amounts to.
    """
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
