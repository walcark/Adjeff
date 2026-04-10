"""Base class to load configuration from earth observation products."""

from pathlib import Path
from typing import ClassVar

import xarray as xr

from adjeff.core import ImageDict, SensorBand
from adjeff.utils import CacheStore

from ..scene_module import SceneModule


class ProductLoader(SceneModule):
    """Load configurations from earth observation products."""

    required_vars: ClassVar[list[str]] = []
    output_vars: ClassVar[list[str]] = [
        "aot",
        "rh",
        "wl",
        "href",
        "h",
        "vza",
        "vaa",
        "sza",
        "saa",
        "species",
    ]

    def __init__(
        self,
        product_path: Path,
        mnt_path: Path,
        href: float = 2.0,
        as_map: bool = False,
        cache: CacheStore | None = None,
    ) -> None:
        if not product_path.is_dir():
            raise FileNotFoundError(
                f"Path {str(product_path)} does not exist."
            )
        self.ensure_correct_folder(product_path)
        self.product_path = product_path
        self.extract_metadata()
        self.h_ref = href
        self.as_map = as_map
        self.mnt_path = mnt_path
        super().__init__(cache=cache)

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Return an ImageDict from the folder."""
        scene = (
            scene
            if scene is not None
            else ImageDict({b: xr.Dataset() for b in self.bands})
        )
        for band in scene.bands:
            res = scene[band]["rho_s"].adjeff.res
            scene[band]["rh"] = self.rh()
            scene[band]["aot"] = self.aot(res=res, ref=scene[band]["rho_s"])
            scene[band]["h"] = self.h(res=res, ref=scene[band]["rho_s"])
            scene[band]["href"] = self.href()
            vza_vaa = self.vza_vaa(band)
            sza_saa = self.sza_saa()
            scene[band]["sza"] = sza_saa[0]
            scene[band]["vza"] = vza_vaa[0]
            scene[band]["saa"] = sza_saa[1]
            scene[band]["vaa"] = vza_vaa[1]
            print(scene[band])
        return scene

    def ensure_correct_folder(self, path: Path) -> None:
        """Check that a product folder is well formatted."""
        ...

    def extract_metadata(self) -> None:
        """Extract metadata from the folder."""
        ...

    def href(self) -> xr.DataArray:
        """Return a default values of 2.0 for href."""
        return xr.DataArray(
            [self.h_ref],
            dims="href",
            coords=dict(href=[self.h_ref]),
        )

    def aot(self, res: float, ref: xr.DataArray) -> xr.DataArray:
        """Return the AOT, either as map or average."""
        ...

    def h(self, res: float, ref: xr.DataArray) -> xr.DataArray:
        """Return the MNT, either as map or average."""
        ...

    def rh(self) -> xr.DataArray:
        """Return the relative humidity in percent."""
        ...

    def vza_vaa(self, band: SensorBand) -> tuple[xr.DataArray, xr.DataArray]:
        """Return the Viewing Zenith and Azimuth angles."""
        ...

    def sza_saa(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Return the Sun Zenith and Azimuth angles."""
        ...

    def species(self) -> dict[str, float]:
        """Return the species proportion in the Atmosphere."""
        ...
