"""Class to load Maja metadata and attributes."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
import structlog
import xarray as xr

from adjeff.core import SensorBand
from adjeff.utils import CacheStore

from .product_loader import ProductLoader

logger = structlog.get_logger(__name__)


class MajaLoader(ProductLoader):
    """Load an output product from the MAJA processor."""

    def __init__(
        self,
        product_path: Path,
        href: float = 2.0,
        as_map: bool = False,
        res: float = 0.120,
        mnt_path: Path = Path("/work/CESBIO/projects/Maja/DTM_120"),
        cache: CacheStore | None = None,
    ) -> None:
        super().__init__(
            product_path=product_path,
            href=href,
            as_map=as_map,
            mnt_path=mnt_path,
        )

    def ensure_correct_folder(self, path: Path) -> None:
        """Check that the product name corresponds to MAJA output format."""
        split = str(path.name).split("_")
        if len(split) != 6:
            raise ValueError(
                f"Wrong folder format for MAJA. Got {str(path)} but should "
                "be <SENTINEL_TYPE>_<DATE>_L2A_<TILE>_C_<VERSION>"
            )

    def extract_metadata(self) -> None:
        """Extract metadata from the folder."""
        split = str(self.product_path.name).split("_")
        xml_path = list(self.product_path.glob("*.xml"))[0]
        assert xml_path.is_file()
        self.mtd = dict(
            sensor=split[0],
            date=split[1][:7],
            tile=split[3][1:],
            xml_path=xml_path,
        )

    def aot(self, res: float, ref: xr.DataArray) -> xr.DataArray:
        """Return the AOT, either as map or average."""
        glob_file = list(self.product_path.glob("*ATB_R2.tif"))
        if len(glob_file) == 0:
            raise FileNotFoundError("No file found for AOT.")
        # Read second index for AOT
        with rasterio.open(glob_file[0]) as src:
            arr = src.read(2).astype(float)

        if self.as_map:
            return xr.DataArray(arr, dims="aot", coords=dict(aot=arr))

        return xr.DataArray(
            change_res(arr, res=res) / 200,
            dims=ref.dims,
            coords=ref.coords,
        )

    def h(self, res: float, ref: xr.DataArray) -> xr.DataArray:
        """Return the MNT, either as map or average."""
        tile = self.mtd["tile"]
        pattern: str = f"S2*{tile}*.DBL.DIR/*{tile}*ALT_R2.TIF"
        glob_mnt = list(self.mnt_path.glob(pattern))
        if len(glob_mnt) == 0:
            raise FileNotFoundError(f"No MNT found for tile {tile}.")

        with rasterio.open(glob_mnt[0]) as src:
            arr = src.read(1).astype(float)

        if self.as_map:
            arr = np.atleast_1d(arr)
            return xr.DataArray(arr, dims="h", coords=dict(h=arr))

        return xr.DataArray(
            change_res(arr, res=res) / 1e3,
            dims=ref.dims,
            coords=ref.coords,
        )

    def rh(self) -> xr.DataArray:
        """Return the relative humidity in percent."""
        root = ET.parse(self.mtd["xml_path"]).getroot()
        rh = root.findtext(".//Product_Quality/Relative_Humidity")
        if rh is not None:
            rh = rh
        else:
            logger.info(
                "Relative humidity not found, default to 50%.",
                path=self.product_path.name,
            )
            rh = 50.0
        rh = np.atleast_1d(rh)
        return xr.DataArray(
            data=np.atleast_1d(rh),
            dims="rh",
            coords=dict(rh=rh),
        )

    def vza_vaa(self, band: SensorBand) -> tuple[xr.DataArray, xr.DataArray]:
        """Return the Viewing Zenith and Azimuth angles."""
        root = ET.parse(self.mtd["xml_path"]).getroot()
        band_id = band.id.replace("0", "")
        xpath = f".//Mean_Viewing_Incidence_Angle[@band_id='{band_id}']"
        view_elem = root.find(xpath)
        if view_elem is not None:
            vza = np.atleast_1d(
                round(float(view_elem.findtext("ZENITH_ANGLE")), 2)
            )
            vaa = np.atleast_1d(
                round(float(view_elem.findtext("AZIMUTH_ANGLE")), 2)
            )
            return (
                xr.DataArray(vza, dims="vza", coords=dict(vza=vza)),
                xr.DataArray(vaa, dims="vaa", coords=dict(vaa=vaa)),
            )
        raise ValueError(f"Viewing angles for band {band} not found.")

    def sza_saa(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Return the Sun Zenith and Azimuth angles."""
        root = ET.parse(self.mtd["xml_path"]).getroot()
        sza = root.findtext(".//Sun_Angles/ZENITH_ANGLE")
        saa = root.findtext(".//Sun_Angles/AZIMUTH_ANGLE")
        if sza is not None and saa is not None:
            sza = np.atleast_1d(round(float(sza), 2))
            saa = np.atleast_1d(round(float(saa), 2))
            return (
                xr.DataArray(sza, dims="sza", coords=dict(sza=sza)),
                xr.DataArray(saa, dims="saa", coords=dict(saa=saa)),
            )
        raise ValueError("Sun angles not found.")

    def species(self) -> dict[str, float]:
        """Return the species proportion in the Atmosphere."""
        cams_aer = [
            "ammonium",
            "blackcar",
            "dust",
            "nitrate",
            "organicm",
            "seasalt",
            "sulphate",
            "secondar",
        ]
        tree = ET.parse(self.mtd["xml_path"])
        root = tree.getroot()
        aots = {}
        aots_sum = 0.0
        for aer in cams_aer:
            aer_aot = root.find(f".//Model[@name='{aer}']")
            if aer_aot is not None:
                aer_str = aer_aot.text if aer_aot.text is not None else ""
                aots[aer] = float(aer_str)
                aots_sum += aots[aer]
        if not np.isclose(aots_sum, 1.0, rtol=1e-2):
            aots = {k: v / aots_sum for k, v in aots.items()}
        return aots


def change_res(data: np.ndarray, res: float) -> np.ndarray:
    """Change the resolution of a 2D array and return as DataArray."""
    # Reduce to self.res
    int_res = int(round(1000 * res))
    print(int_res)
    if (int_res % 20 != 0) or (int_res < 20):
        raise ValueError(f"Target res {res} should be divisible by 0.020km.")
    factor: int = int_res // 20
    return block_reduce(
        data,
        block_size=(factor, factor),
        func=np.nanmean,
    )


def block_reduce(arr, block_size, func=np.mean):
    """Reduce the shape of a 2D array with a specific operator."""
    sx, sy = block_size
    nx, ny = arr.shape
    arr = arr[: nx - nx % sx, : ny - ny % sy]
    arr = arr.reshape(nx // sx, sx, ny // sy, sy)
    return func(arr, axis=(1, 3))
