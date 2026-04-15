from adjeff.core import S2Band
from adjeff.atmosphere import SpectralConfig, AtmoConfig, GeoConfig
from adjeff.utils import ConfigBundle

import xarray as xr



spec: SpectralConfig = SpectralConfig(wl=xr.DataArray([550.0, 783.0], dims=["wl"]), band_type=S2Band)
atm: AtmoConfig = AtmoConfig(
    aot=xr.DataArray([0.0, 0.2, 0.2, 0.6, 0.8], dims=["aot"]), 
    rh=xr.DataArray([50.0, 90.0], dims=["rh"]),
    h=xr.DataArray([0.0, 0.1, 0.1, 0.2, 0.3], dims=["h"]), 
    href=2.0, 
    species={"sulphate": 1.0}
)
geo: GeoConfig = GeoConfig(
    sza=xr.DataArray([0.0], dims=["sza"]),
    vza=0.0,
    saa=0.0,
    vaa=0.0,
)

print(atm)

bundle: ConfigBundle = ConfigBundle(
    configs=[spec, atm, geo],
    scalars=["rh", "h"],
    vectors=["aot", "href", "sza", "vza", "saa", "vaa"],
    deduplicate_dims=None
)

