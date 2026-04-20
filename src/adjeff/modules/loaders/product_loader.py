"""Base class to load configuration from earth observation products."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import xarray as xr

from adjeff.core import SensorBand
from adjeff.exceptions import ConfigurationError
from adjeff.utils import CacheStore

from ..scene_source import SceneSource

if TYPE_CHECKING:
    from adjeff.core import ImageDict


class GeometryMixin(ABC):
    """Mixin for loaders that provide per-band viewing and sun angles.

    Contributes ``vza``, ``vaa``, ``sza``, ``saa`` to :attr:`output_vars`.
    """

    _output_vars_contribution: ClassVar[list[str]] = [
        "vza",
        "vaa",
        "sza",
        "saa",
    ]

    @abstractmethod
    def _vza_vaa(self, band: SensorBand) -> tuple[xr.DataArray, xr.DataArray]:
        """Return (Viewing Zenith Angle, Viewing Azimuth Angle) for *band*."""

    @abstractmethod
    def _sza_saa(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Return (Sun Zenith Angle, Sun Azimuth Angle) for the product."""


class AtmosphereMixin(ABC):
    """Mixin for loaders that provide atmospheric state (AOT, humidity).

    Contributes ``aot``, ``rh`` and ``href`` to :attr:`output_vars`.
    """

    _output_vars_contribution: ClassVar[list[str]] = ["aot", "rh", "href"]

    def __init__(self, href: float = 2.0) -> None:
        self.href = href

    @abstractmethod
    def _aot(self, ref: xr.DataArray) -> xr.DataArray:
        """Return the Aerosol Optical Thickness, as map or scalar."""

    @abstractmethod
    def _rh(self) -> xr.DataArray:
        """Return the relative humidity in percent."""

    def _href(self) -> xr.DataArray:
        """Return the reference height of the aerosols as a DataArray."""
        return xr.DataArray(
            [self.href],
            dims="href",
            coords=dict(href=[self.href]),
        )


class ElevationMixin(ABC):
    """Mixin for loaders that provide surface elevation from an external DEM.

    Contributes ``h`` to :attr:`output_vars`.
    """

    _output_vars_contribution: ClassVar[list[str]] = ["h"]

    @abstractmethod
    def _h(self, ref: xr.DataArray) -> xr.DataArray:
        """Return the surface elevation, as map or scalar (km)."""


class ProductLoader(SceneSource, ABC):
    """Load configurations from earth observation products.

    Subclasses must implement :meth:`reflectance`.  Optionally, they can
    mix in :class:`GeometryMixin`, :class:`AtmosphereMixin`, and/or
    :class:`ElevationMixin` to declare which additional quantities they
    provide.  :meth:`_compute` will populate the corresponding variables
    only when the relevant mixin is present.

    Parameters
    ----------
    product_path : Path
        The folder containing the product data.
    bands : list[SensorBand]
        Bands to load from the product.
    res : float | list[float]
        Target spatial resolution in km (e.g. 0.12 for 120 m).
    as_map : bool [default=False]
        When ``True``, load 2-D parameters as full spatial maps instead of
        spatially-averaged scalars.
    cache : CacheStore | None [default=None]
        Optional on-disk cache for a next session.
    """

    _BASE_OUTPUT_VARS: ClassVar[list[str]] = ["rho_s"]

    @property
    def output_vars(self) -> list[str]:  # type: ignore[override]
        """Aggregate output variables declared by active mixins."""
        result = list(self._BASE_OUTPUT_VARS)
        for cls in type(self).__mro__:
            contrib: list[str] = cls.__dict__.get(
                "_output_vars_contribution", []
            )
            result.extend(contrib)
        return result

    def __init__(
        self,
        product_path: Path,
        bands: list[SensorBand],
        res: float | list[float],
        as_map: bool = False,
        cache: CacheStore | None = None,
    ) -> None:
        if not product_path.is_dir():
            raise FileNotFoundError(
                f"Path {str(product_path)} does not exist."
            )
        self.ensure_correct_folder(product_path)
        self.product_path = product_path
        self._build_band_to_res(bands, res)
        self.extract_metadata()
        self.as_map = as_map
        super().__init__(bands=bands, cache=cache)

    def _build_band_to_res(
        self,
        bands: list[SensorBand],
        res: float | list[float],
    ) -> None:
        """Build a mapping from SensorBand to resolution in metres."""
        res_li: list[float] = res if isinstance(res, list) else [res]

        if len(res_li) == 1:
            res_li = res_li * len(bands)

        if len(res_li) != len(bands):
            raise ConfigurationError(
                "Size of `res` should be the same as `bands` "
                f", got {len(res_li)} != {len(bands)}"
            )

        self.bands_to_res: dict[SensorBand, float] = {
            band: r * 1000.0 for band, r in zip(bands, res_li)
        }

    def _compute(self, scene: "ImageDict") -> "ImageDict":
        """Load all variables from the product into *scene*."""
        _sza_saa: tuple[xr.DataArray, xr.DataArray] | None = None
        _rh: xr.DataArray | None = None

        if isinstance(self, GeometryMixin):
            _sza_saa = self._sza_saa()
        if isinstance(self, AtmosphereMixin):
            _rh = self._rh()

        for band in scene.bands:
            rho_s = self.reflectance(band=band)
            scene[band]["rho_s"] = rho_s

            if isinstance(self, GeometryMixin):
                assert _sza_saa is not None
                sza, saa = _sza_saa
                vza, vaa = self._vza_vaa(band)
                scene[band]["vza"] = vza
                scene[band]["vaa"] = vaa
                scene[band]["sza"] = sza
                scene[band]["saa"] = saa

            if isinstance(self, AtmosphereMixin):
                assert _rh is not None
                try:
                    scene[band]["aot"] = self._aot(ref=rho_s)
                except xr.CoordinateValidationError as e:
                    raise ConfigurationError(
                        "Shape of input scene not consistent "
                        "with product 2D output format."
                    ) from e
                scene[band]["rh"] = _rh
                scene[band]["href"] = self._href()

            if isinstance(self, ElevationMixin):
                try:
                    scene[band]["h"] = self._h(ref=rho_s)
                except xr.CoordinateValidationError as e:
                    raise ConfigurationError(
                        "Shape of input scene not consistent "
                        "with product 2D output format."
                    ) from e

        return scene

    def ensure_correct_folder(self, path: Path) -> None:
        """Check that a product folder is well formatted."""

    def extract_metadata(self) -> None:
        """Extract metadata from the folder."""

    @abstractmethod
    def reflectance(self, band: SensorBand) -> xr.DataArray:
        """Load the surface reflectance for *band* at :attr:`res`.

        Returns a 2-D ``(y, x)`` DataArray with pixel-spaced coordinates
        expressed in km.
        """
