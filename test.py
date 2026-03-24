"""Test module."""

from pathlib import Path

import matplotlib.pyplot as plt

from adjeff.core import ImageDict, S2Band, disk_image_dict
from adjeff.modules import TestModule
from adjeff.utils import CacheStore


def main():

    cache: CacheStore = CacheStore(cache_dir="/home/kwalcarius/tmp")
    output_dir: Path = Path("/home/kwalcarius/tmp/output")

    if True:
        image: ImageDict = disk_image_dict(
            radius=5.0,
            bands=[S2Band.B02, S2Band.B01],
            var="rho_s",
            extent_km={S2Band.B02: 30.0, S2Band.B01: 50.0},
        )
    if False:
        image: ImageDict = disk_image_dict(
            n=1000,
            radius=5.0,
            bands=[S2Band.B02, S2Band.B01],
            var="rho_s",
        )

    for band in image.band_ids:
        image[band]["rho_s"].plot()
        plt.show()

    image2: ImageDict = TestModule(cache=cache)(image)
    image3: ImageDict = TestModule(cache=cache)(image)

    image2.write_to_directory(output_dir, var="rho_toa")

    print(image)


if __name__ == "__main__":
    main()
