"""Class to load Maja metadata and attributes."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import structlog
import xarray as xr

from adjeff.core import SensorBand
from adjeff.utils import CacheStore

from .product_loader import ProductLoader

logger = structlog.get_logger(__name__)


class MajaLoader(ProductLoader):
    """Load an output product from the MAJA processor.

    Parameters
    ----------
    product_path : Path
        The folder containing the product data.
    mnt_path : Path
        The folder storing the DEM (Digital Elevation Model) for the MAJA
        atmospheric processor at 20m.
    href : float [default=2.0]
        The height scale of the aerosol in the atmosphere. This quantity
        is defined for an exponentially decreasing aerosol optical thickness
        with elevation.
    as_map : bool [default=False]
        Whether to load 2D parameters as 2D map (True) or as spatially
        averaged (False). For instance, Aerosol Optical Thickness and DEM are
        generally stored as 2D varying maps, but may need to be average for
        some processing purpose.
    cache : CacheStore | None [default=None]
        Whether to cache the loaded data for a next session.
    """

    def __init__(
        self,
        product_path: Path,
        href: float = 2.0,
        as_map: bool = False,
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
        self.mtd: dict[str, object] = dict(
            sensor=split[0],
            date=split[1][:7],
            tile=split[3][1:],
            xml_path=xml_path,
        )

    def aot(self, ref: xr.DataArray) -> xr.DataArray:
        """Return the AOT, either as map or average."""
        res = ref.adjeff.res
        glob_file = list(self.product_path.glob("*ATB_R2.tif"))
        if len(glob_file) == 0:
            raise FileNotFoundError("No file found for AOT.")
        # Read second index for AOT
        with rasterio.open(glob_file[0]) as src:
            arr = src.read(2).astype(float)
            print(100 * "=")
            print(arr.shape)
            print(100 * "=")

        if self.as_map:
            return xr.DataArray(arr, dims="aot", coords=dict(aot=arr))

        return xr.DataArray(
            downsample_res(arr, target_res=res, data_res=0.020) / 200,
            dims=ref.dims,
            coords=ref.coords,
        )

    def h(self, ref: xr.DataArray) -> xr.DataArray:
        """Return the MNT, either as map or average."""
        res = ref.adjeff.res
        tile = str(self.mtd["tile"])
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
            downsample_res(arr, target_res=res, data_res=0.020) / 1e3,
            dims=ref.dims,
            coords=ref.coords,
        )

    def rh(self) -> xr.DataArray:
        """Return the relative humidity in percent."""
        xml_path = str(self.mtd["xml_path"])
        root = ET.parse(xml_path).getroot()
        rh_str = root.findtext(".//Product_Quality/Relative_Humidity")
        if rh_str is not None:
            rh_val: float = float(rh_str)
        else:
            logger.info(
                "Relative humidity not found, default to 50%.",
                path=self.product_path.name,
            )
            rh_val = 50.0
        rh_arr = np.atleast_1d(rh_val)
        return xr.DataArray(
            data=rh_arr,
            dims="rh",
            coords=dict(rh=rh_arr),
        )

    def vza_vaa(self, band: SensorBand) -> tuple[xr.DataArray, xr.DataArray]:
        """Return the Viewing Zenith and Azimuth angles."""
        xml_path = str(self.mtd["xml_path"])
        root = ET.parse(xml_path).getroot()
        band_id = band.id.replace("0", "")
        xpath = f".//Mean_Viewing_Incidence_Angle[@band_id='{band_id}']"
        view_elem = root.find(xpath)
        if view_elem is not None:
            vza_text = view_elem.findtext("ZENITH_ANGLE") or "0"
            vaa_text = view_elem.findtext("AZIMUTH_ANGLE") or "0"
            vza = np.atleast_1d(round(float(vza_text), 2))
            vaa = np.atleast_1d(round(float(vaa_text), 2))
            return (
                xr.DataArray(vza, dims="vza", coords=dict(vza=vza)),
                xr.DataArray(vaa, dims="vaa", coords=dict(vaa=vaa)),
            )
        raise ValueError(f"Viewing angles for band {band} not found.")

    def sza_saa(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Return the Sun Zenith and Azimuth angles."""
        xml_path = str(self.mtd["xml_path"])
        root = ET.parse(xml_path).getroot()
        sza_str = root.findtext(".//Sun_Angles/ZENITH_ANGLE")
        saa_str = root.findtext(".//Sun_Angles/AZIMUTH_ANGLE")
        if sza_str is not None and saa_str is not None:
            sza_arr = np.atleast_1d(round(float(sza_str), 2))
            saa_arr = np.atleast_1d(round(float(saa_str), 2))
            return (
                xr.DataArray(sza_arr, dims="sza", coords=dict(sza=sza_arr)),
                xr.DataArray(saa_arr, dims="saa", coords=dict(saa=saa_arr)),
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
        tree = ET.parse(str(self.mtd["xml_path"]))
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


def downsample_res(
    data: np.ndarray,
    target_res: float,
    data_res: float,
) -> np.ndarray:
    """Downsample the resolution of a 2D array and return as DataArray.

    The target resolution must be bigger than the original resolution.
    """
    # Reduce to self.res
    int_target_res = int(round(1000 * target_res))
    int_data_res = int(round(1000 * data_res))
    if (int_target_res % int_data_res != 0) or (int_target_res < int_data_res):
        raise ValueError(
            f"Target res {int_target_res} should "
            f" be divisible by {int_data_res}."
        )
    factor: int = int_target_res // int_data_res
    print(factor)
    return block_reduce(
        data,
        block_size=(factor, factor),
        func=np.nanmean,
    )


def block_reduce(
    arr: np.ndarray,
    block_size: tuple[int, int],
    func: Any = np.mean,
) -> np.ndarray:
    """Reduce the shape of a 2D array with a specific operator."""
    sx, sy = block_size
    nx, ny = arr.shape
    arr = arr[: nx - nx % sx, : ny - ny % sy]
    arr = arr.reshape(nx // sx, sx, ny // sy, sy)
    return func(arr, axis=(1, 3))  # type: ignore[no-any-return]
