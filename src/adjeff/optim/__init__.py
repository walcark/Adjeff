"""PSF optimisation: loss, metrics, training data, and the fit loop.

**Fitting**

- :func:`fit`: optimise a model's PSF over every atmospheric combo of a
  training set and return the frozen PSF tree.
- :func:`default_stages`: the Adam warm-up then L-BFGS refinement used
  when :func:`fit` is called without explicit stages.

**Loss and metrics**

- :class:`Metric`: enum of available loss metrics (MAE, MSE, RMSE,
  and their radially-weighted ``_RAD`` variants).
- :class:`Loss`: wraps a :class:`Metric` and evaluates it over a
  :class:`TrainingSet`, with optional ``rho_unif``-based CDF masking.

**Training data**

- :class:`TrainingImages`: collection of reference
  :class:`~adjeff.core.ImageDict` instances with per-image weights.  It
  is the only training type to build by hand: :func:`fit` slices it into
  per-combo tensors itself.

**Stage configuration**

- :class:`AdamConfig`, :class:`LBFGSConfig`: per-stage settings, both
  refining :class:`OptimizerConfig` (steps, tolerance, loss).

**Diagnostics**

- :func:`loss_landscape`: evaluate loss over a PSF parameter grid.
- :func:`energy_radius_landscape`: encircled-energy radii over a grid.
"""

from ._config import OptimizerConfig
from .adam_optimizer import AdamConfig
from .fit import default_stages, fit
from .landscape import energy_radius_landscape, loss_landscape
from .lbfgs_optimizer import LBFGSConfig
from .loss import Loss
from .metrics import Metric
from .training_set import TrainingImages

__all__ = [
    # Fitting
    "fit",
    "default_stages",
    # Loss and metrics
    "Metric",
    "Loss",
    # Training data
    "TrainingImages",
    # Config
    "OptimizerConfig",
    "AdamConfig",
    "LBFGSConfig",
    # Diagnostics
    "loss_landscape",
    "energy_radius_landscape",
]
