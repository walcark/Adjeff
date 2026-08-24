"""Define bands for different sensors to facilitate indexing in ImageDict."""

from enum import Enum


class SensorBand(Enum):
    """Base Enum for a SensorBand."""

    def __init__(self, id: str, wl_nm: float) -> None:
        self.id = id
        self.wl_nm = wl_nm

    def __repr__(self) -> str:
        """Return information about the band."""
        return "{}.{}({}nm)".format(
            self.__class__.__name__,
            self.id,
            self.wl_nm,
        )

    def __str__(self) -> str:
        """Return a simple representation of the band."""
        return f"{self.__class__.__name__}.{self.id}"

    def __cache_token__(self) -> str:
        """Return a stable, serialisable identity for cache fingerprints.

        An Enum member is not JSON-serialisable, so a cache handed one as
        context cannot tell when it changes.  The class name and the id
        pin the band down: two members never share both.
        """
        return f"{type(self).__name__}.{self.id}"


class S2Band(SensorBand):
    """Sentinel-2 spectral bands.

    Wavelengths (``wl_nm``) are in nanometres.
    """

    B01 = ("B01", 443.0)
    B02 = ("B02", 490.0)
    B03 = ("B03", 560.0)
    B04 = ("B04", 665.0)
    B05 = ("B05", 705.0)
    B06 = ("B06", 740.0)
    B07 = ("B07", 783.0)
    B08 = ("B08", 842.0)
    B8A = ("B8A", 865.0)
    B09 = ("B09", 945.0)
    B10 = ("B10", 1375.0)
    B11 = ("B11", 1610.0)
    B12 = ("B12", 2190.0)
