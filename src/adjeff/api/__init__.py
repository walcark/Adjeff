"""High-level API, composing the modules of adjeff.

Functions
---------
    load_scene, load_maja
        Load a product into a scene.
    make_full_config, load_config
        Configs from scalars, or from the fields of a scene.
    run_radiatives_from_scene, run_forward_pipeline
        Radiative terms, or the whole forward chain down to ``rho_unif``.
    make_model, fit_psf, apply_psf
        Build, fit and apply a PSF model.
    sample_psf_atm, sample_psf_atm_from_scene
        Atmospheric PSF of Wu et al. (2024).

Common parameters
-----------------
bands : list[SensorBand]
    Bands to process.
aot, rh, h, href : float, list or DataArray
    Aerosol optical thickness, relative humidity [%], ground elevation
    [km] and aerosol scale height [km]; a list is swept.
sza, vza, saa, vaa : float, list or DataArray
    Sun and viewing zenith and azimuth angles [°].
species : dict[str, float] or None
    Aerosol species fractions; ``{"sulphate": 1.0}`` by default.
atmo_config, geo_config, spectral_config : configs
    As returned by :func:`make_full_config`.
res_km : float
    Pixel size [km].
n : int
    Grid side in pixels, odd.
remove_rayleigh : bool
    Suppress Rayleigh scattering.
afgl_type : str
    AFGL atmosphere profile, ``"afgl_exp_h8km"`` by default.
n_ph : int
    Photons per Smart-G run.
n_bins : int or None
    Digitise ``aot`` and ``h`` to this many values, to cut Smart-G runs.
dedup : bool
    Merge repeated atmospheric states; worth it for spatial maps.
cache : CacheStore or None
    Shared disk cache.
device : str
    Torch device, ``"cuda"`` by default.

Example
-------
>>> cfg = make_full_config(bands=[S2Band.B03], aot=0.1, rh=50.0, sza=30.0, vza=0.0)
>>> model = make_model(
...     Unif2Surface,
...     KingPSF,
...     [S2Band.B03],
...     res_km=0.12,
...     n=1999,
...     init_parameters={"sigma": 0.1, "gamma": 1.0},
... )
>>> scene = run_forward_pipeline(rho_s_scene, **cfg)
>>> psf_tree = fit(model, train_images, loss=Loss(Metric.RMSE_RAD))
"""

from ._config import FullConfig, load_config, make_full_config
from ._load import load_maja, load_scene
from ._pipeline import run_forward_pipeline, run_radiatives_from_scene
from ._psf import (
    ARTICLE_TRAIN_RADII_KM,
    ARTICLE_TRAIN_SIZE,
    apply_psf,
    fit_psf,
    make_model,
    sample_psf_atm,
    sample_psf_atm_from_scene,
)

__all__ = [
    "FullConfig",
    # Loading
    "load_scene",
    "load_maja",
    # Configuration
    "make_full_config",
    "load_config",
    # Pipelines
    "run_radiatives_from_scene",
    "run_forward_pipeline",
    # PSF
    "make_model",
    "fit_psf",
    "apply_psf",
    "sample_psf_atm",
    "sample_psf_atm_from_scene",
    # The manuscript's own choices, named so that a caller can tell them
    # apart from the library's defaults.
    "ARTICLE_TRAIN_RADII_KM",
    "ARTICLE_TRAIN_SIZE",
]
