"""Loss computation for PSF optimisation."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .metrics import Metric
from .training_set import TrainingSet


@dataclass
class Loss:
    """Weighted loss over a :class:`TrainingSet`.

    Parameters
    ----------
    metric : Metric
        Which metric to use, e.g. ``Metric.RMSE_RAD``.
    mask_on : {"rho_unif", None}, optional
        When ``"rho_unif"`` (default), the RAD metrics receive
        ``rho_unif`` as their optional CDF mask tensor.  Pass ``None``
        to disable masking.  Ignored for non-RAD metrics.
    """

    metric: Metric
    mask_on: str | None = "rho_unif"

    def __post_init__(self) -> None:  # noqa: D105
        if self.mask_on not in (None, "rho_unif"):
            raise ValueError(
                f"mask_on must be None or 'rho_unif'got {self.mask_on!r}"
            )

    def __call__(
        self,
        forward_fn: Callable[[dict[str, Tensor]], Tensor],
        train_set: TrainingSet,
    ) -> Tensor:
        """Compute the weighted loss over all samples in *train_set*."""
        losses = []
        for sample in train_set:
            pred = forward_fn(sample.inputs)
            mask = (
                sample.inputs.get("rho_unif")
                if self.mask_on == "rho_unif"
                else None
            )
            losses.append(
                self.metric(pred, sample.target, sample.dist, mask)
                * sample.weight
            )
        return torch.stack(losses).sum()
