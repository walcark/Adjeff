"""Loss computation for PSF optimisation."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from adjeff.exceptions import ConfigurationError

from .metrics import Metric
from .training_set import TrainingSample, TrainingSet


@dataclass
class Loss:
    """Weighted loss over a :class:`TrainingSet`.

    Parameters
    ----------
    metric : Metric
        Which metric to use, e.g. ``Metric.RMSE_RAD``.
    mask_on : str, float or None, optional
        Which pixels the RAD metrics take part in.  Ignored by the plain
        metrics, which mask on their own residual.

        - a **variable name** (default ``"rho_unif"``): keep the pixels
          within the 99% radial energy of that input.  Any name the
          training set carries is accepted.
        - a **radius in km**: keep a disc of that radius.  Unlike the
          energy mask, it does not move when the prediction changes,
          which matters to a quasi-Newton optimiser that assumes a fixed
          objective.
        - ``None``: no mask.
    """

    metric: Metric
    mask_on: str | float | None = "rho_unif"

    def __post_init__(self) -> None:  # noqa: D105
        if isinstance(self.mask_on, bool) or (
            isinstance(self.mask_on, (int, float))
            and not isinstance(self.mask_on, str)
            and self.mask_on <= 0.0
        ):
            raise ConfigurationError(
                "mask_on as a radius must be a positive number of "
                f"kilometres, got {self.mask_on!r}"
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
            mask = self._mask_for(sample)
            losses.append(
                self.metric(pred, sample.target, sample.dist, mask) * sample.weight
            )
        return torch.stack(losses).sum()

    def _mask_for(self, sample: TrainingSample) -> Tensor | None:
        """Return what the metric should restrict itself to, if anything."""
        if self.mask_on is None:
            return None
        if isinstance(self.mask_on, str):
            field = sample.inputs.get(self.mask_on)
            if field is None:
                available = ", ".join(sorted(sample.inputs)) or "none"
                raise ConfigurationError(
                    f"mask_on={self.mask_on!r} is not among the training "
                    f"inputs; the sample carries: {available}."
                )
            return field
        return sample.dist <= float(self.mask_on)
