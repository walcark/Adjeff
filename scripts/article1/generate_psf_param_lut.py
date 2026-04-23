"""Generate PSF parameter LUTs (GaussGeneral + King) for one CAMS species.

Writes one netCDF file per PSF model to OUT_DIR:
    gauss_general_psf_{species}.nc   (variables: sigma, n)
    king_psf_{species}.nc            (variables: sigma, gamma)

Each file contains dimensions (band, aot, rh, h, href).
The ``band`` coordinate is the S2 central wavelength in nm.

Axes swept
----------
Spectral    : all S2 bands (B01–B12, B8A)
Atmospheric : aot in [0.0, 0.1, …, 0.8]
              rh  in [50.0, 90.0, 95.0] %
              h   in [0.0, 3.0] km
              href in [2.0, 4.0] km
Geometric   : sza=40°, vza=0° (fixed)
Species     : one CAMS OPAC species (passed via --species)

Training scenes
---------------
Three Gaussian reflectance fields with sigma = 1, 5, 50 km at 120 m resolution.

Usage
-----
    python generate_psf_param_lut.py \\
        --species sulphate \\
        --cache-dir /path/to/cache \\
        [--output-dir /path/to/output] \\
        [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import xarray as xr

from adjeff.api import (
    make_full_config,
    make_model,
    optimize_adam_lbfgs,
    run_forward_pipeline,
)
from adjeff.core import (
    GeneralizedGaussianPSF,
    KingPSF,
    S2Band,
    SensorBand,
    gaussian_image_dict,
)
from adjeff.core._psf import PSFModule
from adjeff.modules.models.unif2surface import Unif2Surface
from adjeff.optim import Loss, TrainingImages
from adjeff.optim.adam_optimizer import AdamConfig
from adjeff.optim.lbfgs_optimizer import LBFGSConfig
from adjeff.optim.metrics import Metric
from adjeff.utils import CacheStore

# ---------------------------------------------------------------------------
# Configuration — edit these values to adjust the sweep
# ---------------------------------------------------------------------------

BANDS: list[SensorBand] = [
    S2Band.B01,
    S2Band.B02,
    S2Band.B03,
    S2Band.B04,
    S2Band.B05,
    S2Band.B06,
    S2Band.B07,
    S2Band.B08,
    S2Band.B8A,
    S2Band.B09,
    S2Band.B11,
    S2Band.B12,
]

# Fixed geometry
SZA: float = 40.0
VZA: float = 0.0
SAA: float = 0.0
VAA: float = 0.0

# Atmospheric sweep (batched inside the optimizer)
AOT_VALUES: list[float] = np.round(np.arange(0.0, 0.9, 0.1), 1).tolist()
RH_VALUES: list[float] = [50.0, 90.0, 95.0]
H_VALUES: list[float] = [0.0, 3.0]
HREF_VALUES: list[float] = [2.0, 4.0]

# Training scenes (Gaussian fields at three spatial scales)
GAUSSIAN_SIGMAS: list[float] = [1.0, 5.0, 50.0]
GAUSSIAN_WEIGHTS: list[float] = [1.0, 1.0, 1.0]

# PSF and scene grid (120 m resolution, ~240 km extent)
RES_KM: float = 0.12
N: int = 1999

# Smart-G photon count (reduce for quick tests, raise for production)
N_PH: int = int(1e5)

# Chunking for the RadiativePipeline (limits GPU memory per Smart-G call)
RADIATIVE_CHUNKS: dict[str, int] = {"wl": 4, "aot": 3}

# CAMS OPAC aerosol species
CAMS_SPECIES: list[str] = [
    "ammonium",
    "blackcar",
    "dust",
    "nitrate",
    "organicm",
    "seasalt",
    "sulphate",
    "secondar",
]

# PSF models to generate LUTs for
MODELS: dict[str, dict[str, Any]] = {
    "gauss_general": {
        "cls": GeneralizedGaussianPSF,
        "init_params": {"sigma": 0.1, "n": 0.25},
    },
    "king": {
        "cls": KingPSF,
        "init_params": {"sigma": 0.1, "gamma": 1.0},
    },
}

LOSS = Loss(Metric.RMSE_RAD)
ADAM_MIN_STEPS = 5
ADAM_MAX_STEPS = 20
LBFGS_MIN_STEPS = 5
LBFGS_MAX_STEPS = 30

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_training_images(
    species: dict[str, float],
    cache: CacheStore,
) -> TrainingImages:
    """Run the forward pipeline for all Gaussian scenes at this species."""
    cfg = make_full_config(
        bands=BANDS,
        aot=AOT_VALUES,
        rh=RH_VALUES,
        h=H_VALUES,
        href=HREF_VALUES,
        sza=SZA,
        vza=VZA,
        saa=SAA,
        vaa=VAA,
        species=species,
    )
    scenes = []
    for sigma in GAUSSIAN_SIGMAS:
        base_scene = gaussian_image_dict(
            sigma=sigma,
            res_km=RES_KM,
            bands=BANDS,
            n=N,
        )
        scene = run_forward_pipeline(
            base_scene,
            **cfg,
            n_ph=N_PH,
            cache=cache,
            radiative_chunks=RADIATIVE_CHUNKS,
        )
        scenes.append(scene)
    return TrainingImages(images=scenes, weights=GAUSSIAN_WEIGHTS)


def _optimize_psf(
    train_images: TrainingImages,
    psf_cls: type[PSFModule],
    init_params: dict[str, float],
    device: str,
) -> dict[SensorBand, dict[str, xr.DataArray]]:
    """Optimise *psf_cls* on *train_images* and return per-band param DataArrays."""
    model = make_model(
        Unif2Surface,
        psf_cls,
        BANDS,
        res_km=RES_KM,
        n=N,
        init_parameters=init_params,
        device=device,
    )
    psf_dict = optimize_adam_lbfgs(
        model,
        train_images,
        LOSS,
        adam_config=AdamConfig(
            min_steps=ADAM_MIN_STEPS,
            max_steps=ADAM_MAX_STEPS,
            loss_relative_tolerance=1e-4,
            loss=LOSS,
        ),
        lbfgs_config=LBFGSConfig(
            min_steps=LBFGS_MIN_STEPS,
            max_steps=LBFGS_MAX_STEPS,
            loss_relative_tolerance=1e-6,
            loss=LOSS,
        ),
        device=device,
    )

    result: dict[SensorBand, dict[str, xr.DataArray]] = {}
    for band in BANDS:
        params = psf_dict.params(band)
        if params is None:
            raise RuntimeError(
                f"Optimizer returned no params for {band}. "
                "Check that training images have swept atmospheric dimensions."
            )
        param_das: dict[str, xr.DataArray] = {}
        for pname, pval in params.items():
            if not isinstance(pval, xr.DataArray):
                raise RuntimeError(
                    f"Expected multi-combo DataArray for {band}/{pname}, "
                    f"got {type(pval)}. "
                    "Check that training images have swept atmospheric dimensions."
                )
            param_das[pname] = pval
        result[band] = param_das
    return result


def _make_lut_dataset(
    params_by_band: dict[SensorBand, dict[str, xr.DataArray]],
    model_name: str,
    species_name: str,
) -> xr.Dataset:
    """Stack per-band params into a Dataset with dimension ``band`` (wl in nm)."""
    param_names = list(next(iter(params_by_band.values())).keys())

    stacked: dict[str, list[xr.DataArray]] = {p: [] for p in param_names}
    for band in sorted(params_by_band.keys(), key=lambda b: b.wl_nm):
        for pname in param_names:
            da = params_by_band[band][pname]
            stacked[pname].append(da.expand_dims({"band": [band.wl_nm]}))

    ds_vars: dict[str, xr.DataArray] = {
        pname: xr.concat(stacked[pname], dim="band") for pname in param_names
    }

    return xr.Dataset(
        ds_vars,
        attrs={
            "psf_model": model_name,
            "species": species_name,
            "sza_deg": SZA,
            "vza_deg": VZA,
            "saa_deg": SAA,
            "vaa_deg": VAA,
            "band_coordinate_unit": "nm",
            "training_scenes": f"Gaussian sigma={GAUSSIAN_SIGMAS} km",
            "res_km": RES_KM,
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    species_name: str,
    cache_dir: Path,
    output_dir: Path,
    device: str = "cuda",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = CacheStore(str(cache_dir))

    n_total = len(MODELS)
    task_idx = 0

    logger.info("Building training images.", species=species_name)
    train_images = _build_training_images(
        species={species_name: 1.0},
        cache=cache,
    )

    for model_name, model_cfg in MODELS.items():
        task_idx += 1
        out_path = output_dir / f"{model_name}_psf_{species_name}.nc"

        if out_path.exists():
            logger.info(
                "LUT already exists, skipping.",
                model=model_name,
                species=species_name,
                path=str(out_path),
            )
            continue

        logger.info(
            "Starting PSF optimisation.",
            model=model_name,
            species=species_name,
            progress=f"{task_idx}/{n_total}",
        )

        params_by_band = _optimize_psf(
            train_images=train_images,
            psf_cls=model_cfg["cls"],
            init_params=model_cfg["init_params"],
            device=device,
        )

        ds = _make_lut_dataset(params_by_band, model_name, species_name)
        ds.to_netcdf(out_path)
        logger.info(
            "LUT saved.",
            model=model_name,
            species=species_name,
            path=str(out_path),
            dims=dict(ds.dims),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate PSF parameter LUTs (GaussGeneral + King) "
            "for one CAMS OPAC species."
        ),
    )
    parser.add_argument(
        "--species",
        required=True,
        choices=CAMS_SPECIES,
        help="CAMS OPAC aerosol species to process.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory for Smart-G simulation cache (shared across runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output" / "psf_luts",
        help=(
            "Directory for output .nc files "
            "(default: <script_dir>/output/psf_luts/)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for PSF convolutions (default: cuda).",
    )
    args = parser.parse_args()
    main(
        species_name=args.species,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
