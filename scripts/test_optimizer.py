"""End-to-end test for the PSF optimiser.

Pipeline:
  disk_image_dict (rho_s)
    → run_forward_pipeline  (radiatives + rho_toa + rho_unif)
    → optimize_adam_lbfgs   trains a KingPSF on three scenes
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import structlog
import torch
import xarray as xr

from adjeff.api import make_full_config, make_model, optimize_adam_lbfgs, run_forward_pipeline
from adjeff.core import KingPSF, S2Band, disk_image_dict
from adjeff.modules.models.unif2surface import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages
from adjeff.utils import CacheStore
from adjeff.utils.logger import MultilineConsoleRenderer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        MultilineConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAND = S2Band.B03
RES_KM = 0.120
N_PIX = 1999
CACHE_DIR = Path("/tmp/adjeff_optim_cache")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

cfg = make_full_config(
    bands=[BAND],
    aot=[0.0, 0.15, 0.3, 0.45],
    rh=50.0,
    h=0.0,
    href=2.0,
    sza=30.0,
    vza=10.0,
    saa=120.0,
    vaa=120.0,
    sat_height=786.0,
)

cache = CacheStore(CACHE_DIR)

# ---------------------------------------------------------------------------
# Build training scenes
# ---------------------------------------------------------------------------

logger.info("Building training scenes.")

train_images = TrainingImages(
    images=run_forward_pipeline(
        [
            disk_image_dict(
                radius=r, res_km=RES_KM, rho_min=0.0, rho_max=1.0,
                bands=[BAND], var="rho_s", n=N_PIX,
            )
            for r in [1.0, 5.0, 50.0]
        ],
        **cfg,
        cache=cache,
        n_ph=10000
    ),
    weights=[1.0, 1.0, 1.0],
)

im = train_images.images[0]
print(im[S2Band.B03]["rho_unif"])
import sys
sys.exit()

# ---------------------------------------------------------------------------
# Model + optimisation
# ---------------------------------------------------------------------------

model = make_model(
    Unif2Surface, KingPSF, [BAND],
    res_km=RES_KM, n=N_PIX,
    init_parameters={"sigma": 1e-1, "gamma": 1e0},
)

psf_dict = optimize_adam_lbfgs(
    model, train_images, Loss(Metric.RMSE_RAD),
)

print(psf_dict.params(band=BAND))
logger.info("Optimisation complete.", psf_dict=repr(psf_dict))

# ---------------------------------------------------------------------------
# Plot rho_s radial profiles: ground truth vs prediction
# ---------------------------------------------------------------------------
logger.info("Plotting rho_s radial profiles true vs prediction.")

scene_labels = ["r=1 km", "r=5 km", "r=50 km"]

N: int = len(train_images.images)
fig, axes = plt.subplots(1, N, figsize=(5 * N, 4))

for ax, scene, label in zip(axes, train_images.images, scene_labels):
    scene_pred = model(scene)

    truth_da = scene[BAND]["rho_s"]
    pred_da = scene_pred[BAND]["rho_s"]

    extra_dims = [d for d in pred_da.dims if d not in ("y", "x")]

    # vérité (unique)
    r_true = truth_da.adjeff.radial()
    ax.plot(r_true.coords["r"], r_true.values, label="rho_s", lw=2, color="black")

    # 👉 boucle sur chaque dim séparément
    for dim in extra_dims:
        for i in range(pred_da.sizes[dim]):
            pred_sel = pred_da.isel({dim: i})

            # fixer les autres dims à 0
            for other_dim in extra_dims:
                if other_dim != dim:
                    pred_sel = pred_sel.isel({other_dim: 0})

            r_pred = pred_sel.adjeff.radial()

            ax.plot(
                r_pred.coords["r"],
                r_pred.values,
                ls="--",
                lw=1,
                label=f"{dim}={i}",
                alpha=0.7,
            )

    ax.set_title(label, fontsize=9)
    ax.set_xlabel("r [km]")
    ax.set_ylabel("rho_s")
    ax.grid(True, alpha=0.3)

    # éviter une légende illisible
    ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
plt.show()
