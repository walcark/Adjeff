"""Adam-based PSF optimizer stage and convenience optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import structlog
import torch
import torch.nn as nn

from adjeff.core.bands import SensorBand
from adjeff.modules.scene_module import TrainableSceneModule

from ._combo_stage import (
    _ComboStage,
    _loss_delta,
    project_all_params,
    restore_all_params,
    save_all_params,
)
from ._config import OptimizerConfig
from .training_set import TrainingSet

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AdamConfig(OptimizerConfig):
    """Configuration for the Adam optimizer stage.

    Parameters
    ----------
    lr : float
        Adam learning rate (default ``1e-2``).
    """

    lr: float = 1e-2


class AdamStage(_ComboStage):
    """Single-combo optimization stage using Adam gradient descent.

    Implements :meth:`_run_combo` with ``torch.optim.Adam``.  Typically
    used as a warm-up stage before :class:`LBFGSStage`; selected by
    :func:`~adjeff.optim.fit` for every :class:`AdamConfig` it is given.
    """

    def __init__(self, config: AdamConfig) -> None:
        super().__init__(config)
        self.config: AdamConfig = config

    def _run_combo(
        self,
        model: TrainableSceneModule,
        band: SensorBand,
        data: TrainingSet,
        combo_str: str,
    ) -> None:
        """Adam optimisation loop for one combo."""
        best_params = save_all_params(model)
        params_to_opt = list(cast(nn.Module, model.psf_modules[band.id]).parameters())
        adam = torch.optim.Adam(params_to_opt, lr=self.config.lr)

        while self.nloop < self.config.max_steps:
            adam.zero_grad(set_to_none=True)
            loss_t = self._total_loss(model, band, data)
            loss_t.backward()  # type: ignore[no-untyped-call]
            adam.step()
            project_all_params(model)
            loss = float(loss_t)
            params = save_all_params(model)
            self.record(loss, params)

            delta = _loss_delta(self.previous_loss, loss, self.nloop)
            logger.info(
                f"Adam  {self.nloop + 1}/{self.config.max_steps}"
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
