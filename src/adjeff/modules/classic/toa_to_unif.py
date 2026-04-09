"""Compute rho_unif from rho_toa."""

from typing import ClassVar

from adjeff.core import ImageDict

from ..scene_module import SceneModule


class Toa2Unif(SceneModule):
    """Compute uniform reflectance from top-of-atmosphere reflectance.

    The uniform reflectance is computed with the 5S model, assuming the
    environment reflectance is equal to the surface reflectance.
    """

    required_vars: ClassVar[list[str]] = [
        "rho_toa",
        "tdir_up",
        "tdif_up",
        "tdir_down",
        "tdif_down",
        "rho_atm",
        "sph_alb",
    ]
    output_vars: ClassVar[list[str]] = ["rho_unif"]

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Invert the 5S model, assuming rho_s=rho_env=rho_unif."""
        for band in scene.bands:
            ds = scene[band]
            rho_toa_star = ds["rho_toa"] - ds["rho_atm"]
            t_up = ds["tdir_up"] + ds["tdif_up"]
            t_down = ds["tdir_down"] + ds["tdif_down"]
            ds["rho_unif"] = rho_toa_star / (
                ds["sph_alb"] * rho_toa_star + t_up * t_down
            )
        return scene
