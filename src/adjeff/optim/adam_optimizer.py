"""Adam-based PSF optimizer stage and convenience optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch

from adjeff.core.bands import SensorBand
from adjeff.modules.scene_module import TrainableSceneModule

from ._combo_stage import (
    _ComboStage,
    _loss_delta,
    restore_all_params,
    save_all_params,
)
from ._config import OptimizerConfig
from .optimizer import SingleStageOptimizer
from .training_set import TrainingImages, TrainingSet

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
    """Runs Adam gradient descent for a single atmospheric combo."""

    def __init__(self, config: AdamConfig) -> None:
        super().__init__(config)
        self.config: AdamConfig = config

    def _run_combo(
        self,
        model: TrainableSceneModule,
        band_sets: list[tuple[SensorBand, TrainingSet]],
        combo_str: str,
    ) -> None:
        """Adam optimisation loop for one combo."""
        best_params = save_all_params(model)
        adam = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
        )

        while self.nloop < self.config.max_steps:
            adam.zero_grad(set_to_none=True)
            loss_t = self._total_loss(model, band_sets)
            loss_t.backward()  # type: ignore[no-untyped-call]
            adam.step()
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


class AdamOptimizer(SingleStageOptimizer):
    """PSF optimizer using Adam gradient descent.

    Parameters
    ----------
    train_images : TrainingImages
        Collection of training scenes.
    config : AdamConfig
        Adam-specific configuration.
    device : str
        PyTorch device (default ``"cuda"``).

    Example
    -------
    >>> optimizer = AdamOptimizer(
    ...     train_images=train_images,
    ...     config=AdamConfig(
    ...         min_steps=5,
    ...         max_steps=50,
    ...         loss_relative_tolerance=1e-4,
    ...         loss=Loss(Metric.RMSE_RAD),
    ...     ),
    ... )
    >>> psf_dict = optimizer.run(model)
    """

    def __init__(
        self,
        train_images: TrainingImages,
        config: AdamConfig,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            stage=AdamStage(config),
            train_images=train_images,
            device=device,
        )
        self.config: AdamConfig = config
