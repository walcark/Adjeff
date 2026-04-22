from pathlib import Path

import matplotlib.pyplot as plt

from adjeff.api import load_maja
from adjeff.core import S2Band
from adjeff.utils import CacheStore


def main() -> None:
    cache = CacheStore("/tmp/adjeff-temp3")

    scene = load_maja(
        product_path=Path(
            "/home/kwalcarius/downloads/"
            "SENTINEL2B_20180616-102816-151_L2A_T32TMR_C_V1-0/"
        ),
        mnt_path=Path("/home/kwalcarius/downloads/DTM/"),
        bands=[S2Band.B02],
        res=0.120,
        as_map=False,
        cache=cache,
        compute_radiatives=True,
        n_bins=10,
    )
    
    for band in scene.bands:
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        ax = ax.flatten()
        scene[band]["rho_s"].plot(ax=ax[0])
        scene[band]["sph_alb"].plot(ax=ax[1])
        scene[band]["tdif_up"].plot(ax=ax[2])
        scene[band]["tdif_down"].plot(ax=ax[3])
        scene[band]["rho_atm"].plot(ax=ax[4])
        scene[band]["tdir_up"].plot(ax=ax[5])
        plt.show()


if __name__ == "__main__":
    main()
