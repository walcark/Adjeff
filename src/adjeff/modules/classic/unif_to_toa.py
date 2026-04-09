"""Compute rho_toa from rho_unif."""

from typing import ClassVar

import xarray as xr

from adjeff.core import ImageDict

from ..scene_module import SceneModule


class Unif2Toa(SceneModule):
    """Compute top-of-atmosphere reflectance from known uniform reflectance.

    The uniform reflectance is computed with the 5S model, assuming the
    environment reflectance is equal to the surface reflectance.
    """

    required_vars: ClassVar[list[str]] = [
        "rho_unif",
        "tdir_up",
        "tdif_up",
        "tdir_down",
        "tdif_down",
        "rho_atm",
        "sph_alb",
    ]
    output_vars: ClassVar[list[str]] = ["rho_toa"]

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Use the 5S model, assuming rho_s=rho_env=rho_unif."""
        for band in scene.bands:
            ds: xr.Dataset = scene[band]
            t_up = ds["tdir_up"] + ds["tdif_up"]
            t_down = ds["tdir_down"] + ds["tdif_down"]
            frac = ds["rho_unif"] / (1 - ds["sph_alb"] * ds["rho_unif"])
            ds["rho_toa"] = ds["rho_atm"] + t_up * t_down * frac
        return scene
