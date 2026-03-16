"""Define bands for different sensors to facilitate indexing in ImageDict."""

from enum import Enum


class SensorBand(Enum):
    """Base Enum for a SensorBand."""


class S2Band(SensorBand):
    """Define the bands for Sentinel-2.

    Wavelengths are store in nm, and resolutions in km.
    """

    B01 = ("B01", 443.0, 0.06)
    B02 = ("B02", 490.0, 0.01)
    B03 = ("B03", 560.0, 0.01)
    B04 = ("B04", 665.0, 0.01)
    B05 = ("B05", 705.0, 0.02)
    B06 = ("B06", 740.0, 0.02)
    B07 = ("B07", 783.0, 0.02)
    B08 = ("B08", 842.0, 0.01)
    B8A = ("B8A", 865.0, 0.02)
    B09 = ("B09", 945.0, 0.06)
    B10 = ("B10", 1375.0, 0.06)
    B11 = ("B11", 1610.0, 0.02)
    B12 = ("B12", 2190.0, 0.02)

    def __init__(self, id: str, wl_nm: float, res_km: float) -> None:
        self._value_ = id
        self.wl_nm = wl_nm
        self.res_km = res_km

    def __str__(self) -> str:
        """Return a simple representation of the band."""
        return f"{self.name}: {self.wl_nm}nm, {self.res_km}km"
