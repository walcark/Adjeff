"""Settings shared by every optimisation stage.

Classes
-------
    OptimizerConfig
        Step bounds, stopping tolerance and loss of a stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from .loss import Loss


@dataclass(frozen=True)
class OptimizerConfig:
    """Settings shared by every optimisation stage.

    Parameters
    ----------
    min_steps : int
        Steps run before early stopping may trigger.
    max_steps : int
        Maximum number of steps.
    loss_relative_tolerance : float
        Stop once ``|Δloss / loss|`` falls below it.
    loss : Loss
        Loss the stage minimises.
    """

    min_steps: int
    max_steps: int
    loss_relative_tolerance: float
    loss: Loss
