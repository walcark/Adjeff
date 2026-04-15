from pathlib import Path

import matplotlib.pyplot as plt

from adjeff.modules.loaders import MajaLoader
from adjeff.modules.samplers import RadiativePipeline
from adjeff.core import S2Band, gaussian_image_dict
from adjeff.utils import CacheStore
from adjeff.api import make_full_config

def main():
    
    cache = CacheStore("/tmp/adjeff-temp")

    scene = gaussian_image_dict(sigma=15.0, res_km=0.120, bands=[S2Band.B02], n=915)
    scene = MajaLoader(
        product_path=Path("/home/kwalcarius/dev/current/adjeff/scripts/SENTINEL2B_20180616-102816-151_L2A_T32TMR_C_V1-0/"),
        mnt_path=Path("/home/kwalcarius/dev/current/adjeff/scripts/DTM/"),
        cache=cache
    ).forward(scene=scene)
    
    n_bins = 10
    config = make_full_config(                                                                                                                                                             
      bands=scene.bands,
      aot=scene[S2Band.B02]["aot"].adjeff.digitize(n_bins=n_bins),
      h=scene[S2Band.B02]["h"].adjeff.digitize(n_bins=n_bins),                                                                                                                           
      rh=scene[S2Band.B02]["rh"],
      vza=scene[S2Band.B02]["vza"],                                                                                                                                                      
      sza=scene[S2Band.B02]["sza"],
      saa=scene[S2Band.B02]["saa"],                                                                                                                                                      
      vaa=scene[S2Band.B02]["vaa"],                                                                                                                                                      
    )

    pipeline: RadiativePipeline = RadiativePipeline(
        atmo_config=config["atmo_config"],
        geo_config=config["geo_config"],
        spectral_config=config["spectral_config"],
        remove_rayleigh=False,
        n_ph_rho_atm=int(1E6),
        n_ph_sph_alb=int(1E6),
        n_ph_tdif_down=int(1E6),
        n_ph_tdif_up=int(1E6),
        deduplicate_dims=["x", "y"],
        cache=cache
    )

    scene = pipeline(scene)
    print(scene[S2Band.B02]["sph_alb"].compute())
    print(scene[S2Band.B02]["rho_atm"].compute())
    if True:
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        ax = ax.flatten()
        scene[S2Band.B02]["aot"].adjeff.digitize(n_bins=n_bins).plot(ax=ax[0])
        scene[S2Band.B02]["h"].adjeff.digitize(n_bins=n_bins).plot(ax=ax[1])
        scene[S2Band.B02]["sph_alb"].plot(ax=ax[2])
        scene[S2Band.B02]["tdif_up"].plot(ax=ax[3])
        scene[S2Band.B02]["tdif_down"].plot(ax=ax[4])
        scene[S2Band.B02]["rho_atm"].plot(ax=ax[5])

        plt.show()


if __name__ == "__main__":
    main()
