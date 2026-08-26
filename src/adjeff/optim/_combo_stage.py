"""Abstract base for single-combo optimization stages."""

from __future__ import annotations

import abc
from typing import cast

import torch
import torch.nn as nn

from adjeff.core.bands import SensorBand
from adjeff.modules.scene_module import TrainableSceneModule

from ._config import OptimizerConfig
from .training_set import TrainingSet

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def project_all_params(model: TrainableSceneModule) -> None:
    """Bring every constrained parameter back onto its domain.

    Called after each optimiser step: see
    :meth:`~adjeff.utils.ConstrainedParameter.project` for why a bound
    enforced only inside the forward pass is not enough.
    """
    for module in cast(nn.Module, model).modules():
        project = getattr(module, "project", None)
        if callable(project):
            project()


def _loss_delta(previous: float, current: float, step: int) -> float | None:
    """Return the relative loss change in percent, or ``None`` on the first step.

    A number rather than the formatted string it used to be: a log line
    that carries it as data can be filtered on, plotted, or written out
    as JSON, which is the whole point of logging key-values.
    """
    if step == 0 or previous >= float("inf"):
        return None
    return 100.0 * (previous - current) / max(abs(previous), 1e-9)


# ---------------------------------------------------------------------------
# Parameter snapshot helpers
# ---------------------------------------------------------------------------


def save_all_params(
    model: TrainableSceneModule,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return a snapshot of all PSF unconstrained parameters."""
    return {
        band_id: {
            name: p.data.clone() for name, p in cast(nn.Module, psf).named_parameters()
        }
        for band_id, psf in model.psf_modules.items()
    }


@torch.no_grad()
def restore_all_params(
    model: TrainableSceneModule,
    saved: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Restore all PSF parameters from a snapshot."""
    for band_id, psf in model.psf_modules.items():
        for name, p in cast(nn.Module, psf).named_parameters():
            p.copy_(saved[band_id][name])


# ---------------------------------------------------------------------------
# Abstract combo stage
# ---------------------------------------------------------------------------


class _ComboStage(abc.ABC):
    """Abstract base for a single-combo optimization stage.

    A :class:`_ComboStage` encapsulates the optimization logic for one
    atmospheric combo ``(aot_i, rh_i, ...)``.  It owns the per-combo state
    (loss history, best params, step counter) and stopping logic.

    Subclasses implement :meth:`_run_combo`.

    Parameters
    ----------
    config : OptimizerConfig
        Stage-specific configuration (steps, loss, tolerance).
    """

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self.loss_history: list[float] = []
        self.params_history: list[dict[str, dict[str, torch.Tensor]]] = []
        self.nloop: int = 0
        self.previous_loss: float = float("inf")
        self.best_loss: float = float("inf")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset per-combo counters before each run."""
        self.loss_history = []
        self.params_history = []
        self.nloop = 0
        self.previous_loss = float("inf")
        self.best_loss = float("inf")

    def record(
        self,
        loss: float,
        params: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Append current loss and parameter snapshot to history."""
        self.loss_history.append(loss)
        self.params_history.append(params)

    def improved_loss_or_under_min_steps(self, loss: float) -> bool:
        """Return ``True`` if training should continue.

        Always returns ``True`` when fewer than ``min_steps`` have run.
        After that, continues only if the relative loss improvement
        exceeds ``loss_relative_tolerance``.
        """
        if self.nloop < self.config.min_steps:
            return True
        rel = (self.previous_loss - loss) / max(abs(self.previous_loss), 1e-9)
        return abs(rel) > self.config.loss_relative_tolerance

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _total_loss(
        self,
        model: TrainableSceneModule,
        band: SensorBand,
        data: TrainingSet,
    ) -> torch.Tensor:
        """Loss of one band at one atmospheric combo."""

        def forward(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            return model.forward_band(band, **inputs)

        return self.config.loss(forward, data)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _run_combo(
        self,
        model: TrainableSceneModule,
        band: SensorBand,
        data: TrainingSet,
        combo_str: str,
    ) -> None:
        """Run optimization for a single atmospheric combo.

        Must update ``self.best_loss`` and ``self.nloop`` via
        :meth:`record` and :meth:`improved_loss_or_under_min_steps`.
        """
