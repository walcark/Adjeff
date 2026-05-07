"""PSF optimisation: loss, metrics, training data, and optimizers.

The optimization workflow is built in three layers:

**Loss and metrics**

- :class:`Metric` — enum of available loss metrics (MAE, MSE, RMSE,
  and their radially-weighted ``_RAD`` variants).
- :class:`Loss` — wraps a :class:`Metric` and evaluates it over a
  :class:`TrainingSet`, with optional ``rho_unif``-based CDF masking.

**Training data**

- :class:`TrainingImages` — collection of reference
  :class:`~adjeff.core.ImageDict` instances with per-image weights.
- :class:`TrainingSet` — tensors sliced at a single atmospheric combo,
  ready for the gradient loop.
- :class:`TrainingSample` — one ``(inputs, target, dist, weight)`` item.

**Optimizers**

All optimizers inherit from the internal ``_Optimizer`` base class,
which handles the outer loop over atmospheric combos and kernel stacking.

- :class:`LBFGSOptimizer` / :class:`LBFGSConfig` / :class:`LBFGSStage`
- :class:`AdamOptimizer` / :class:`AdamConfig` / :class:`AdamStage`
- :class:`OptimizerPipeline` — chains multiple stages per combo.
- :class:`SingleStageOptimizer` — wraps a single stage.
- :class:`OptimizerConfig` — shared config (steps, tolerance, loss).

**Diagnostics**

- :func:`loss_landscape` — evaluate loss over a PSF parameter grid.
- :func:`energy_radius_landscape` — encircled-energy radii over a grid.
"""

from ._config import OptimizerConfig
from .adam_optimizer import AdamConfig, AdamOptimizer, AdamStage
from .landscape import energy_radius_landscape, loss_landscape
from .lbfgs_optimizer import LBFGSConfig, LBFGSOptimizer, LBFGSStage
from .loss import Loss
from .metrics import Metric
from .optimizer import OptimizerPipeline, SingleStageOptimizer
from .training_set import TrainingImages, TrainingSample, TrainingSet

__all__ = [
    # Loss and metrics
    "Metric",
    "Loss",
    # Training data
    "TrainingImages",
    "TrainingSet",
    "TrainingSample",
    # Config
    "OptimizerConfig",
    "LBFGSConfig",
    "AdamConfig",
    # Optimizers
    "LBFGSOptimizer",
    "LBFGSStage",
    "AdamOptimizer",
    "AdamStage",
    "OptimizerPipeline",
    "SingleStageOptimizer",
    # Diagnostics
    "loss_landscape",
    "energy_radius_landscape",
]
