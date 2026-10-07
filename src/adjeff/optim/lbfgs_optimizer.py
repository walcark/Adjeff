"""L-BFGS optimisation stage.

Classes
-------
    LBFGSConfig
        Settings of an L-BFGS stage.
    LBFGSStage
        Stage running ``torch.optim.LBFGS``, typically as a refinement.
"""

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
    """Settings of an L-BFGS stage, passed to ``torch.optim.LBFGS``.

    Parameters
    ----------
    learning_rate : float, optional
        Step size, 1 by default.
    max_iter : int, optional
        Inner iterations per step, 20 by default.
    history_size : int, optional
        Past gradients kept, 100 by default.
    tolerance_grad, tolerance_change : float, optional
        Inner convergence tolerances on the gradient and the parameters.
    line_search_fn : str, optional
        Line search, ``"strong_wolfe"`` by default.
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
                # The strong-Wolfe bracket collapses on a flat loss: treat it
                # as convergence.  Logged as well as warned, since a warning
                # shows only once per call site over a sweep of fits.
                msg = "L-BFGS line search degenerated, stopping early."
                logger.warning("fit.linesearch_stalled", detail=msg, step=self.nloop)
                warnings.warn(msg, OptimizationWarning, stacklevel=2)
                break
            loss = float(loss_tensor.item())
            params = save_all_params(model)
            self.record(loss, params)

            delta = _loss_delta(self.previous_loss, loss, self.nloop)
            logger.info(
                "fit.step",
                optimizer="lbfgs",
                step=self.nloop + 1,
                of=self.config.max_steps,
                loss=loss,
                delta_pct=delta,
            )

            if loss < self.best_loss:
                self.best_loss = loss
                best_params = params

            self.nloop += 1

            if not self.improved_loss_or_under_min_steps(loss):
                break

            self.previous_loss = loss

        restore_all_params(model, best_params)
