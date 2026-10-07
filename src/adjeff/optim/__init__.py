"""PSF optimisation: fit loop, loss, metrics, training data, diagnostics.

Functions
---------
    fit
        Fit every PSF of a model over each atmospheric combo, as a frozen
        tree.
    default_stages
        Adam warm-up then L-BFGS refinement.
    loss_landscape
        Loss of each PSF of a list, averaged over the combos.
    energy_radius_landscape
        Encircled-energy radii of each PSF of a list.

Classes
-------
    Loss
        Weighted metric over a training set, optionally masked.
    Metric
        MAE, MSE, RMSE and their radially weighted ``_RAD`` variants.
    TrainingImages
        Reference scenes and their loss weights.
    OptimizerConfig, AdamConfig, LBFGSConfig
        Settings of an optimisation stage.
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
