"""Shared optimizer configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from .loss import Loss


@dataclass(frozen=True)
class OptimizerConfig:
    """Shared configuration for all PSF optimizer stages.

    Parameters
    ----------
    min_steps : int
        Minimum steps before early stopping is allowed.
    max_steps : int
        Hard upper bound on the number of steps.
    loss_relative_tolerance : float
        Stop when ``|Δloss / previous_loss| ≤ loss_relative_tolerance``.
    loss : Loss
        Loss function instance.
    """

    min_steps: int
    max_steps: int
    loss_relative_tolerance: float
    loss: Loss
