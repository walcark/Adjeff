"""Class to load Maja metadata and attributes."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import structlog
import xarray as xr

from adjeff.core import SensorBand
from adjeff.utils import CacheStore

from .product_loader import (
    AtmosphereMixin,
    ElevationMixin,
    GeometryMixin,
    ProductLoader,
)

logger = structlog.get_logger(__name__)


class MajaLoader(
    ProductLoader,
    GeometryMixin,
    AtmosphereMixin,
    ElevationMixin,
):
    """Load an output product from the MAJA processor.

    Parameters
    ----------
    product_path : Path
        The folder containing the product data.
    bands : list[SensorBand]
        Bands to load from the product.
    res : float [default=0.12]
        Target spatial resolution in km (e.g. 0.12 for 120 m).
    mnt_path : Path
        The folder storing the DEM at 20 m resolution.
    href : float [default=2.0]
        Height scale of the aerosol (exponentially decreasing AOT model).
    as_map : bool [default=False]
        When ``True``, load 2-D parameters as full spatial maps instead of
        spatially-averaged scalars.
    cache : CacheStore | None [default=None]
        Optional on-disk cache for a next session.
    """

    def __init__(
        self,
        product_path: Path,
        bands: list[SensorBand],
        res: float | list[float],
        href: float = 2.0,
        as_map: bool = False,
        mnt_path: Path = Path("/work/CESBIO/projects/Maja/DTM_120"),
        cache: CacheStore | None = None,
    ) -> None:
        self.mnt_path = mnt_path
        super().__init__(
            product_path=product_path,
            bands=bands,
            res=res,
            href=href,
            as_map=as_map,
            cache=cache,
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

    def reflectance(
        self,
        band: SensorBand,
        btype: Literal["SRE", "FRE"] = "SRE",
    ) -> xr.DataArray:
        """Load the surface reflectance at :attr:`resolution_m`.

        Coordinates are normalised to pixel-spaced km values so that
        ``da.adjeff.res`` returns the correct resolution in km.  The
        original CRS is preserved in the ``crs`` attribute.

        Parameters
        ----------
        band : SensorBand | list[SensorBand]
            Band(s) to load.  When a list is passed only the first band
            is returned (multi-band loading is not yet supported).
        btype : {"SRE", "FRE"}, optional
            Reflectance type: surface (SRE) or flat-surface (FRE).

        Returns
        -------
        xr.DataArray
            2-D ``(y, x)`` DataArray with values in [0, 1] and coordinates
            in km.
        """
        from sensorsio.sentinel2 import Sentinel2

        # Strip leading zeros only: "B02" → "B2", "B10" → "B10", "B8A" → "B8A"
        sio_name = re.sub(r"^B0+", "B", band.id)
        band_sio = Sentinel2.Band[sio_name]
        btype_sio = Sentinel2.BandType[btype]

        s2 = Sentinel2(str(self.product_path))
        data = s2.read_as_xarray(
            [band_sio],
            band_type=btype_sio,
            resolution=self.bands_to_res[band],
        )

        # Extract the single reflectance variable, drop the time dimension.
        var_name = band_sio.name  # e.g. "B2"
        rho_s = data[var_name].squeeze("t", drop=True)

        # Convert UTM coordinates from metres to km so that da.adjeff.res
        # returns the correct resolution in km while preserving the
        # relative spatial layout of the original CRS.
        rho_s = rho_s.assign_coords(
            x=rho_s.coords["x"] / 1000.0,
            y=rho_s.coords["y"] / 1000.0,
        )
        return rho_s

    def aot(self, ref: xr.DataArray) -> xr.DataArray:
        """Return the AOT, either as map or average."""
        res = ref.adjeff.res
        glob_file = list(self.product_path.glob("*ATB_R2.tif"))
        if len(glob_file) == 0:
            raise FileNotFoundError("No file found for AOT.")
        # Band index 2 (1-based) is the AOT layer.
        with rasterio.open(glob_file[0]) as src:
            arr = src.read(2).astype(float)

        if self.as_map:
            return xr.DataArray(
                downsample_res(arr, target_res=res, data_res=0.020) / 200,
                dims=ref.dims,
                coords=ref.coords,
            )

        mean_val = np.atleast_1d(np.nanmean(arr / 200))
        return xr.DataArray(mean_val, dims="aot", coords=dict(aot=mean_val))

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
            return xr.DataArray(
                downsample_res(arr, target_res=res, data_res=0.020) / 1e3,
                dims=ref.dims,
                coords=ref.coords,
            )

        mean_val = np.atleast_1d(np.nanmean(arr / 1e3))
        return xr.DataArray(mean_val, dims="h", coords=dict(h=mean_val))

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
        band_id = re.sub(r"^B0+", "B", band.id)
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
        """Return the aerosol species proportions from CAMS data in the XML."""
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
        aots: dict[str, float] = {}
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
    """Downsample *data* from *data_res* to *target_res* (both in km).

    Both resolutions must be expressed in the same unit.  *target_res*
    must be an integer multiple of *data_res*.
    """
    int_target_res = int(round(1000 * target_res))
    int_data_res = int(round(1000 * data_res))
    if (int_target_res % int_data_res != 0) or (int_target_res < int_data_res):
        raise ValueError(
            f"Target res {int_target_res} should "
            f" be divisible by {int_data_res}."
        )
    factor: int = int_target_res // int_data_res
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
    """Reduce a 2-D array by applying *func* over non-overlapping blocks."""
    sx, sy = block_size
    nx, ny = arr.shape
    arr = arr[: nx - nx % sx, : ny - ny % sy]
    arr = arr.reshape(nx // sx, sx, ny // sy, sy)
    return func(arr, axis=(1, 3))  # type: ignore[no-any-return]
