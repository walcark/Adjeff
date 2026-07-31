"""End-to-end test for the atmospheric PSF sampler."""

from pathlib import Path

import structlog
import matplotlib.pyplot as plt

from adjeff.api import make_full_config, sample_psf_atm, optimize_adam_lbfgs
from adjeff.core import S2Band
from adjeff.utils import CacheStore
import adjeff.accessor

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANDS = [S2Band.B03]
RES_KM = 0.010
N_PIX = 7999
CACHE_DIR = Path("/tmp/adjeff_optim_cache")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

cfg = make_full_config(
    bands=BANDS,
    aot=[0.1, 0.8],
    rh=50.0,
    h=0.0,
    href=2.0,
    sza=0.0,
    vza=10.0,
    saa=120.0,
    vaa=120.0,
    species={"seasalt": 1.0}
)

cache = CacheStore(CACHE_DIR)

# ---------------------------------------------------------------------------
# Sample PSF
# ---------------------------------------------------------------------------

psf_dict = sample_psf_atm(
    bands=BANDS,
    res_km=RES_KM,
    n=N_PIX,
    atmo_config=cfg["atmo_config"],
    geo_config=cfg["geo_config"],
    remove_rayleigh=True,
    n_ph=int(1e8),
    #cache=cache
)


for ao in [0.1, 0.8]:
    psf_dict.to_dataarray(S2Band.B03).sel(aot=ao).adjeff.radial().plot(label=f"aot={ao}")
plt.yscale("log")
plt.legend()
plt.show()

for ao in [0.1, 0.8]:
    psf_dict.to_dataarray(S2Band.B03).sel(aot=ao).adjeff.radial(stat="cdf").plot(label=f"aot={ao}")
plt.xscale("log")
plt.legend()
plt.show()

logger.info("PSF sampled.", bands=psf_dict.bands)
print(psf_dict)
