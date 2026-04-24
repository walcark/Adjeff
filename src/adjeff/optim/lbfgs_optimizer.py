"""L-BFGS PSF optimizer stage and convenience optimizer."""

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
    """Runs L-BFGS optimisation for a single atmospheric combo."""

    def __init__(self, config: LBFGSConfig) -> None:
        super().__init__(config)
        self.config: LBFGSConfig = config

    def _run_combo(
        self,
        model: TrainableSceneModule,
        band_sets: list[tuple[SensorBand, TrainingSet]],
        combo_str: str,
    ) -> None:
        """L-BFGS optimisation loop for one combo."""
        best_params = save_all_params(model)

        opt = torch.optim.LBFGS(
            params=list(model.parameters()),
            lr=self.config.learning_rate,
            max_iter=self.config.max_iter,
            history_size=self.config.history_size,
            line_search_fn=self.config.line_search_fn,
            tolerance_grad=self.config.tolerance_grad,
            tolerance_change=self.config.tolerance_change,
        )

        def closure(
            _band_sets: list[tuple[SensorBand, TrainingSet]] = band_sets,
        ) -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            loss = self._total_loss(model, _band_sets)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        while self.nloop < self.config.max_steps:
            loss_tensor = opt.step(closure)  # type: ignore[no-untyped-call]
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


class LBFGSOptimizer(SingleStageOptimizer):
    """PSF optimizer using L-BFGS.

    Parameters
    ----------
    train_images : TrainingImages
        Collection of training scenes.
    config : LBFGSConfig
        L-BFGS-specific configuration.
    device : str
        PyTorch device (default ``"cuda"``).

    Example
    -------
    >>> optimizer = LBFGSOptimizer(
    ...     train_images=train_images,
    ...     config=LBFGSConfig(
    ...         min_steps=5,
    ...         max_steps=50,
    ...         loss_relative_tolerance=1e-4,
    ...         loss=Loss(Metric.MSE_RAD),
    ...     ),
    ... )
    >>> psf_dict = optimizer.run(model)
    """

    def __init__(
        self,
        train_images: TrainingImages,
        config: LBFGSConfig,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            stage=LBFGSStage(config),
            train_images=train_images,
            device=device,
        )
        self.config: LBFGSConfig = config
