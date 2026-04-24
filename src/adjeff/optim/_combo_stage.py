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


def _loss_delta(previous: float, current: float, step: int) -> str:
    """Return a formatted relative loss change string, or '' on first step."""
    if step == 0 or previous >= float("inf"):
        return ""
    pct = 100.0 * (previous - current) / max(abs(previous), 1e-9)
    return f"  Δ={pct:+.1f}%"


# ---------------------------------------------------------------------------
# Parameter snapshot helpers
# ---------------------------------------------------------------------------


def save_all_params(
    model: TrainableSceneModule,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return a snapshot of all PSF unconstrained parameters."""
    return {
        band_id: {
            name: p.data.clone()
            for name, p in cast(nn.Module, psf).named_parameters()
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
        """Return True if training should continue."""
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
        band_sets: list[tuple[SensorBand, TrainingSet]],
    ) -> torch.Tensor:
        """Sum of losses over all bands for a single atmospheric combo."""
        losses = []
        for band, ts in band_sets:
            _band = band

            def _fwd(
                inputs: dict[str, torch.Tensor],
                _b: SensorBand = _band,
            ) -> torch.Tensor:
                return model.forward_band(_b, **inputs)

            losses.append(self.config.loss(_fwd, ts))
        return torch.stack(losses).sum()

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _run_combo(
        self,
        model: TrainableSceneModule,
        band_sets: list[tuple[SensorBand, TrainingSet]],
        combo_str: str,
    ) -> None:
        """Run optimization for a single atmospheric combo.

        Must update ``self.best_loss`` and ``self.nloop`` via
        :meth:`record` and :meth:`improved_loss_or_under_min_steps`.
        """
