"""End-to-end test for the atmospheric PSF sampler."""

from pathlib import Path

import structlog

from adjeff.api import make_full_config, sample_psf_atm
from adjeff.core import S2Band
from adjeff.utils import CacheStore

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANDS = [S2Band.B02, S2Band.B03, S2Band.B8A, S2Band.B12]
RES_KM = 0.120
N_PIX = 1999
CACHE_DIR = Path("/tmp/adjeff_optim_cache")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

cfg = make_full_config(
    bands=BANDS,
    aot=[0.2],
    rh=50.0,
    h=0.0,
    href=2.0,
    sza=30.0,
    vza=10.0,
    saa=120.0,
    vaa=120.0,
    species={"blackcar": 1.0}
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
    cache=cache
)

logger.info("PSF sampled.", bands=psf_dict.bands)
print(psf_dict)
