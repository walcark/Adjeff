from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(["science", "nature"])

from adjeff import (
    FullConfig,
    make_full_config,
    make_model,
    optimize_adam_lbfgs,
    run_forward_pipeline,
    sample_psf_atm,
)
from adjeff.api import load_maja
from adjeff.core import GaussPSF, KingPSF, S2Band, disk_image_dict
from adjeff.modules.models.unif2surface import Unif2Surface
from adjeff.optim import Loss, Metric, TrainingImages
from adjeff.utils import CacheStore

BAND = S2Band.B8A
RES_KM = 0.12
N = 1999
DISK_RADII_KM = [1.0, 5.0, 50.0]
GAUSS_SIGMA_KM = 0.33  # 330 m


def main() -> None:
    cache = CacheStore("/work/scratch/data/walcark/tmp/adjeff")

    # Load scene with spatial maps and radiative quantities
    scene = load_maja(
        product_path=Path(
            "/work/CESBIO/projects/Maja/L2A_MAJA/LeCaire_noenv/36RVV/"
            "SENTINEL2A_20230602-084138-448_L2A_T36RVV_C_V1-0/"
        ),
        bands=[BAND],
        res=RES_KM,
        mnt_path=Path("/work/CESBIO/projects/Maja/DTM_120/"),
        as_map=True,
        cache=cache,
        n_bins=10,
        compute_radiatives=True,
        deduplicate_dims=["x", "y"],
    )
    
    # Le dossier contient rho_unif, pas rho_s : on le place dans le bon slot
    scene[BAND]["rho_unif"] = scene[BAND]["rho_s"]
    print(scene[BAND])

    # Build a scalar FullConfig from spatial mean of each atmospheric/geometric field
    ds = scene[BAND]
    cfg: FullConfig = make_full_config(
        bands=[BAND],
        aot=float(ds["aot"].mean()),
        h=float(ds["h"].mean()),
        rh=float(ds["rh"].mean()),
        href=float(ds["href"].mean()),
        sza=float(ds["sza"].mean()),
        vza=float(ds["vza"].mean()),
        saa=float(ds["saa"].mean()),
        vaa=float(ds["vaa"].mean()),
    )
    print(cfg)

    # Create 3 binary disk scenes (uniform surface) at 1, 5, 50 km radius
    disk_scenes = [
        disk_image_dict(
            radius=r,
            res_km=RES_KM,
            rho_min=0.0,
            rho_max=0.5,
            bands=[BAND],
            n=N,
        )
        for r in DISK_RADII_KM
    ]

    # Run full forward pipeline (radiatives → rho_toa → rho_unif) on each disk
    disk_scenes = run_forward_pipeline(disk_scenes, **cfg, cache=cache)

    train_images = TrainingImages(images=disk_scenes, weights=[1.0, 1.0, 1.0])

    # Fit King PSF via Adam warm-up + L-BFGS
    model_king = make_model(
        Unif2Surface,
        KingPSF,
        [BAND],
        res_km=RES_KM,
        n=N,
        init_parameters={"sigma": 0.5, "gamma": 2.0},
    )
    psf_king = optimize_adam_lbfgs(
        model=model_king,
        train_images=train_images,
        loss=Loss(Metric.RMSE_RAD),
    )
    king_params: dict[str, float] = (                                                                                                                                              
        model_king.to_psf_dict().params(BAND) or {}                                                                                                                          
    )  
    print("Paramètres King PSF :", king_params)

    # Rebuild eval models on N_EVAL grid (915×915, taille des images rho_unif)
    model_gauss = make_model(
        Unif2Surface,
        GaussPSF,
        [BAND],
        res_km=RES_KM,
        n=915,
        init_parameters={"sigma": GAUSS_SIGMA_KM},
    )
    model_king = make_model(
        Unif2Surface,
        KingPSF,
        [BAND],
        res_km=RES_KM,
        n=915,
        init_parameters=king_params,
    )
    model_gauss.eval()
    model_king.eval()

    scene_gauss = model_gauss(scene)
    scene_king = model_king(scene)

    # Transects at isel(x=150)
    tick_factor = 1.3
    fig, ax = plt.subplots(figsize=(13, 4))

    profile = scene_gauss[BAND]["rho_unif"].sel(x=500.0, method="nearest")
    profile.plot(ax=ax, color="green", marker="x", markersize=7, markevery=20, label=r"$\rho_{unif}$ — no adjacency effects correction",  linewidth=2.0)
    rho_s_gauss = scene_gauss[BAND]["rho_s"].sel(x=500.0, method="nearest").plot(ax=ax, label=r"$\rho_{s}$ — adjacency effects correction (Gauss $330m$)", color="blue", linewidth=2.0)
    rho_s_king = scene_king[BAND]["rho_s"].sel(x=500.0, method="nearest").plot(ax=ax, label=r"$\rho_{s}$ — adjacency effects correction (King optimized)", color="orange", linewidth=2.0)
    ax.legend(loc="upper right", fontsize=12*tick_factor)
    ax.grid(alpha=0.7, linestyle="-")
    y = profile.coords["y"]                                                                                                                                                                
    ax.set_xlabel(r"Distance $[km]$", fontsize=12*tick_factor)
    ax.set_ylabel(r"Surface reflectance", fontsize=12*tick_factor)
    ax.set_title("")
    xticks = [0, 22, 44, 66, 88, 110]
    ax.set_xticks(np.linspace(float(y.min()), float(y.max()), 6))
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlim(float(y.min()), float(y.max()))
    ax.set_ylim(0, 0.6)
    ax.tick_params(
        axis="both",
        which="major",
        width=1.5,
        length=6,
        labelsize=10 * tick_factor,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=1.5,
        length=3,
        labelsize=8 * tick_factor,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig("transect_psf.png", dpi=500)
    plt.show()

    # Sample atmospheric PSF via Smart-G Monte Carlo (désactivé par défaut)
    if False:
        psf_atm = sample_psf_atm(
            bands=[BAND],
            res_km=RES_KM,
            n=N,
            atmo_config=cfg["atmo_config"],
            geo_config=cfg["geo_config"],
            cache=cache,
            n_ph=int(1e7),
        )
        r_atm = psf_atm.to_dataarray(BAND).adjeff.radial(stat="mean")
        ax.plot(
            r_atm.coords["r"], r_atm.values, label="PSF atmosphérique", lw=2
        )
        ax.legend()


if __name__ == "__main__":
    main()
