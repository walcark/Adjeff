"""Generate a King PSF parameter LUT for all atmospheric, geometric and spectral axes.

One Zarr file is written per CAMS aerosol species under ``LUT_DIR``.
Each file is an xarray Dataset with variables ``sigma`` and ``gamma``
carrying dimensions ``(band, sza, vza, aot, rh, h, href)``.

Axes swept
----------
Spectral  : B02, B03, B04, B8A, B11, B12
Geometric : sza in SZA_VALUES, vza in VZA_VALUES  (saa and vaa fixed at 0°)
Atmospheric: aot, rh, h, href  (swept as vector_dims inside the optimizer)

Computation cost
----------------
Total optimizer runs  = len(SZA_VALUES) × len(VZA_VALUES) per species
per run, the optimizer iterates over
    len(AOT_VALUES) × len(RH_VALUES) × len(H_VALUES) × len(HREF_VALUES)
atmospheric combos automatically.

Cache
-----
Smart-G calls (forward pipeline) are cached in CACHE_DIR.
Re-running the script with the same parameters skips already-computed scenes.

Storage
-------
LUTs are written to LUT_DIR / king_psf_{species}.zarr.
If the file already exists the script skips that species.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import structlog
import xarray as xr

from adjeff.api import make_full_config, make_model, optimize_adam_lbfgs
from adjeff.api import run_forward_pipeline
from adjeff.core import (
    KingPSF,
    PSFGrid,
    S2Band,
    SensorBand,
    disk_image_dict,
)
from adjeff.modules.models.unif2surface import Unif2Surface
from adjeff.optim import Loss, TrainingImages
from adjeff.optim.adam_optimizer import AdamConfig
from adjeff.optim.lbfgs_optimizer import LBFGSConfig
from adjeff.optim.metrics import Metric
from adjeff.utils import CacheStore

# ---------------------------------------------------------------------------
# Configuration — edit these values to adjust the sweep
# ---------------------------------------------------------------------------

LUT_DIR = Path(__file__).parent.parent / "data" / "luts" / "king_psf"
CACHE_DIR = Path("/tmp/adjeff-king-lut-cache")

BANDS: list[SensorBand] = [
    S2Band.B02,
    S2Band.B03,
    S2Band.B04,
    S2Band.B8A,
    S2Band.B11,
    S2Band.B12,
]

# Geometric sweep
SZA_VALUES: list[float] = [0.0, 15.0, 30.0, 45.0, 60.0]
VZA_VALUES: list[float] = [0.0, 5.0, 10.0, 15.0]
SAA: float = 0.0
VAA: float = 0.0

# Atmospheric sweep (batched inside the optimizer)
AOT_VALUES: list[float] = [0.05, 0.1, 0.2, 0.4, 0.8]
RH_VALUES: list[float] = [30.0, 50.0, 70.0, 90.0]
H_VALUES: list[float] = [0.0, 0.5, 1.0, 2.0]
HREF_VALUES: list[float] = [1.0, 2.0, 4.0]

# Training scenes (disk fields of three sizes)
DISK_RADII: list[float] = [1.0, 5.0, 50.0]
DISK_WEIGHTS: list[float] = [1.0, 1.0, 1.0]

# PSF grid
RES_KM: float = 0.12
N: int = 1999

# Smart-G photon count (reduce for testing)
N_PH: int = int(1e5)

# CAMS aerosol species
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

# Optimizer settings
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
    sza: float,
    vza: float,
    species: dict[str, float],
    cache: CacheStore,
) -> TrainingImages:
    """Run the forward pipeline for all disk scenes at this geometry."""
    cfg = make_full_config(
        bands=BANDS,
        aot=AOT_VALUES,
        rh=RH_VALUES,
        h=H_VALUES,
        href=HREF_VALUES,
        sza=sza,
        vza=vza,
        saa=SAA,
        vaa=VAA,
        species=species,
    )
    grid = PSFGrid(res=RES_KM, n=N)
    scenes = []
    for radius in DISK_RADII:
        base_scene = disk_image_dict(
            radius=radius,
            res_km=RES_KM,
            bands=BANDS,
            n=N,
        )
        scene = run_forward_pipeline(
            base_scene,
            **cfg,
            n_ph=N_PH,
            cache=cache,
        )
        scenes.append(scene)
    return TrainingImages(images=scenes, weights=DISK_WEIGHTS)


def _optimize_king(
    train_images: TrainingImages,
    device: str = "cuda",
) -> dict[SensorBand, dict[str, xr.DataArray]]:
    """Optimise KingPSF on *train_images* and return per-band param DataArrays.

    Returns
    -------
    dict[SensorBand, dict[str, xr.DataArray]]
        ``{band: {"sigma": DataArray(aot, rh, h, href),
                  "gamma": DataArray(aot, rh, h, href)}}``
    """
    grid = PSFGrid(res=RES_KM, n=N)
    model = make_model(
        Unif2Surface,
        KingPSF,
        BANDS,
        res_km=RES_KM,
        n=N,
        init_parameters={"sigma": 0.1, "gamma": 1.0},
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
        if params is None or not isinstance(
            params.get("sigma"), xr.DataArray
        ):
            raise RuntimeError(
                f"Optimizer did not return multi-combo params for {band}. "
                "Check that training images have swept atmo dimensions."
            )
        result[band] = {
            "sigma": params["sigma"],
            "gamma": params["gamma"],
        }
    return result


def _make_lut_dataset(
    pieces: dict[
        SensorBand,
        list[tuple[float, float, xr.DataArray, xr.DataArray]],
    ],
) -> xr.Dataset:
    """Stack (sza, vza) pieces into a full multi-dim Dataset.

    Parameters
    ----------
    pieces : dict[band, list[(sza, vza, sigma_da, gamma_da)]]

    Returns
    -------
    xr.Dataset
        Variables ``sigma`` and ``gamma`` with dims
        ``(band, sza, vza, aot, rh, h, href)``.
        ``band`` coordinate is wavelength in nm.
    """
    sigma_bands: list[xr.DataArray] = []
    gamma_bands: list[xr.DataArray] = []

    for band, geo_list in sorted(pieces.items(), key=lambda kv: kv[0].wl_nm):
        # Stack across (sza, vza) for this band
        sigma_pieces = []
        gamma_pieces = []
        for sza, vza, sigma_da, gamma_da in geo_list:
            sigma_pieces.append(
                sigma_da.expand_dims({"sza": [sza], "vza": [vza]})
            )
            gamma_pieces.append(
                gamma_da.expand_dims({"sza": [sza], "vza": [vza]})
            )

        sigma_band = xr.combine_by_coords(sigma_pieces)
        gamma_band = xr.combine_by_coords(gamma_pieces)

        # Add band dimension (wavelength in nm)
        sigma_bands.append(
            sigma_band.expand_dims({"band": [band.wl_nm]})
        )
        gamma_bands.append(
            gamma_band.expand_dims({"band": [band.wl_nm]})
        )

    sigma_full = xr.concat(sigma_bands, dim="band")
    gamma_full = xr.concat(gamma_bands, dim="band")

    ds = xr.Dataset(
        {"sigma": sigma_full, "gamma": gamma_full},
        attrs={
            "psf_model": "King",
            "kernel_formula": "King(r) = (1 + (r/sigma)^2)^(-gamma)",
            "bands_id": [b.id for b in sorted(pieces, key=lambda b: b.wl_nm)],
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(device: str = "cuda") -> None:
    LUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = CacheStore(str(CACHE_DIR))

    n_sza = len(SZA_VALUES)
    n_vza = len(VZA_VALUES)
    n_geo = n_sza * n_vza

    for species_name in CAMS_SPECIES:
        out_path = LUT_DIR / f"king_psf_{species_name}.zarr"
        if out_path.exists():
            logger.info(
                "LUT already exists, skipping.",
                species=species_name,
                path=str(out_path),
            )
            continue

        logger.info("Starting LUT generation.", species=species_name)
        species_dict = {species_name: 1.0}

        # pieces[band] = list of (sza, vza, sigma_da, gamma_da)
        pieces: dict[
            SensorBand,
            list[tuple[float, float, xr.DataArray, xr.DataArray]],
        ] = {band: [] for band in BANDS}

        geo_idx = 0
        for sza in SZA_VALUES:
            for vza in VZA_VALUES:
                geo_idx += 1
                logger.info(
                    "Geometry combo.",
                    species=species_name,
                    sza=sza,
                    vza=vza,
                    progress=f"{geo_idx}/{n_geo}",
                )

                train_images = _build_training_images(
                    sza=sza,
                    vza=vza,
                    species=species_dict,
                    cache=cache,
                )
                band_params = _optimize_king(
                    train_images, device=device
                )

                for band, params in band_params.items():
                    pieces[band].append(
                        (sza, vza, params["sigma"], params["gamma"])
                    )

        logger.info(
            "Building and saving LUT dataset.", species=species_name
        )
        ds = _make_lut_dataset(pieces)
        ds.to_zarr(out_path, mode="w")
        logger.info(
            "LUT saved.",
            species=species_name,
            path=str(out_path),
            dims=dict(ds.dims),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate King PSF parameter LUT."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for convolutions (default: cuda).",
    )
    args = parser.parse_args()
    main(device=args.device)
