"""L-BFGS PSF optimizer stage and convenience optimizer."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from adjeff.core.bands import SensorBand
from adjeff.exceptions import OptimizationWarning
from adjeff.modules.scene_module import TrainableSceneModule

from .._logging import get_logger
from ._combo_stage import (
    _ComboStage,
    _loss_delta,
    project_all_params,
    restore_all_params,
    save_all_params,
)
from ._config import OptimizerConfig
from .training_set import TrainingSet

logger = get_logger(__name__)


@dataclass(frozen=True)
class LBFGSConfig(OptimizerConfig):
    """Configuration for the L-BFGS optimizer stage.

    Parameters
    ----------
    learning_rate : float
        Step size passed to ``torch.optim.LBFGS`` (default 1.0).
    max_iter : int
        Maximum inner L-BFGS iterations per step (default 20).
    history_size : int
        Number of past gradients kept in memory (default 100).
    tolerance_grad : float
        Gradient norm tolerance for convergence (default 1e-10).
    tolerance_change : float
        Parameter change tolerance (default 1e-12).
    line_search_fn : str
        Line-search strategy (default ``"strong_wolfe"``).
    """

    learning_rate: float = 1.0
    max_iter: int = 20
    history_size: int = 100
    tolerance_grad: float = 1e-10
    tolerance_change: float = 1e-12
    line_search_fn: str = "strong_wolfe"


class LBFGSStage(_ComboStage):
    """Single-combo optimization stage using L-BFGS.

    Implements :meth:`_run_combo` with a ``torch.optim.LBFGS`` optimizer
    and strong-Wolfe line search.  Selected by :func:`~adjeff.optim.fit`
    for every :class:`LBFGSConfig` it is given.
    """

    def __init__(self, config: LBFGSConfig) -> None:
        super().__init__(config)
        self.config: LBFGSConfig = config

    def _run_combo(
        self,
        model: TrainableSceneModule,
        band: SensorBand,
        data: TrainingSet,
        combo_str: str,
    ) -> None:
        """L-BFGS optimisation loop for one combo."""
        best_params = save_all_params(model)
        params_to_opt = list(cast(nn.Module, model.psf_modules[band.id]).parameters())
        opt = torch.optim.LBFGS(
            params=params_to_opt,
            lr=self.config.learning_rate,
            max_iter=self.config.max_iter,
            history_size=self.config.history_size,
            line_search_fn=self.config.line_search_fn,
            tolerance_grad=self.config.tolerance_grad,
            tolerance_change=self.config.tolerance_change,
        )

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            loss = self._total_loss(model, band, data)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        while self.nloop < self.config.max_steps:
            try:
                loss_tensor = opt.step(closure)  # type: ignore[no-untyped-call]
                project_all_params(model)
            except IndexError:
                # PyTorch strong-Wolfe line search can raise IndexError when
                # the bracket collapses on a numerically flat loss surface.
                # Treat as convergence and exit cleanly.
                # Two channels on purpose: `warnings` is the public,
                # catchable signal, but Python shows it once per call
                # site, which hides how often it happens over a sweep of
                # thousands of fits.  The log line is the one that counts
                # them, hence `warning` and not `info`.
                msg = "L-BFGS line search degenerated, stopping early."
                logger.warning(msg, step=self.nloop)
                warnings.warn(msg, OptimizationWarning, stacklevel=2)
                break
            loss = float(loss_tensor.item())
            params = save_all_params(model)
            self.record(loss, params)

            delta = _loss_delta(self.previous_loss, loss, self.nloop)
            logger.info(
                f"L-BFGS  {self.nloop + 1}/{self.config.max_steps}"
                f"  loss={loss:.4g}{delta}"
            )

            if loss < self.best_loss:
                self.best_loss = loss
                best_params = params

            self.nloop += 1

            if not self.improved_loss_or_under_min_steps(loss):
                break

            self.previous_loss = loss

        restore_all_params(model, best_params)
