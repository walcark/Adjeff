"""Tests for adjeff.modules.loaders (ProductLoader, MajaLoader)."""

import numpy as np
import pytest
import xarray as xr

from adjeff.exceptions import ImageIOError


def _make_ref(res_km: float = 0.12, n: int = 3) -> xr.DataArray:
    """Return a minimal (n x n) DataArray with km-spaced x/y coords."""
    coords = np.arange(n, dtype=np.float32) * res_km
    return xr.DataArray(
        np.ones((n, n), dtype=np.float32),
        dims=["y", "x"],
        coords={"y": coords, "x": coords},
    )


class TestMissingInputs:
    """What a loader says when a product file is absent."""

    def test_product_loader_path_not_exists(self, tmp_path):
        """ProductLoader raises ImageIOError for a non-existent product path."""
        from adjeff.core import S2Band
        from adjeff.modules.loaders.product_loader import ProductLoader

        class MinimalLoader(ProductLoader):
            def reflectance(self, band):  # type: ignore[override]
                return _make_ref()

        with pytest.raises(ImageIOError, match="does not exist"):
            MinimalLoader(tmp_path / "nonexistent", bands=[S2Band.B02], res=0.12)

    def test_maja_aot_file_missing_raises(self, tmp_path):
        from adjeff.modules.loaders.maja_loader import MajaLoader

        loader = MajaLoader.__new__(MajaLoader)
        loader.product_path = tmp_path  # real empty dir, no *ATB_R2.tif file
        loader.as_map = False

        with pytest.raises(ImageIOError, match="AOT"):
            loader._aot(_make_ref())

    def test_maja_dem_file_missing_raises(self, tmp_path):
        from adjeff.modules.loaders.maja_loader import MajaLoader

        loader = MajaLoader.__new__(MajaLoader)
        loader.mnt_path = tmp_path  # real empty dir, no DEM file
        loader.mtd = {"tile": "T31TGK"}
        loader.as_map = False

        with pytest.raises(ImageIOError, match="DEM"):
            loader._h(_make_ref())

    def test_maja_vza_not_in_xml_raises(self, tmp_path):
        """MajaLoader._vza_vaa raises ImageIOError when the band is absent in XML."""
        from adjeff.core import S2Band
        from adjeff.modules.loaders.maja_loader import MajaLoader

        xml_file = tmp_path / "meta.xml"
        xml_file.write_text("<root></root>")

        loader = MajaLoader.__new__(MajaLoader)
        loader.mtd = {"xml_path": xml_file}

        with pytest.raises(ImageIOError, match="Viewing angles"):
            loader._vza_vaa(S2Band.B02)

    def test_maja_sza_not_in_xml_raises(self, tmp_path):
        """MajaLoader._sza_saa raises ImageIOError when sun angles are absent."""
        from adjeff.modules.loaders.maja_loader import MajaLoader

        xml_file = tmp_path / "meta.xml"
        xml_file.write_text("<root></root>")

        loader = MajaLoader.__new__(MajaLoader)
        loader.mtd = {"xml_path": xml_file}

        with pytest.raises(ImageIOError, match="Sun angles"):
            loader._sza_saa()
