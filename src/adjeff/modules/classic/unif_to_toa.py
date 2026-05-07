"""Apply the 5S radiative transfer forward model to compute TOA reflectance."""

from typing import ClassVar

import xarray as xr

from adjeff.core import ImageDict

from ..scene_module import SceneModule


class Unif2Toa(SceneModule):
    """Forward 5S model: compute TOA reflectance from uniform reflectance.

    Assumes the environment reflectance equals the surface reflectance
    (``rho_s = rho_env = rho_unif``).  This is the *forward* direction;
    use :class:`~adjeff.modules.classic.Toa2Unif` to invert.

    The forward formula applied per band is::

        t_up = tdir_up + tdif_up
        t_down = tdir_down + tdif_down
        rho_toa = rho_atm + t_up * t_down * rho_unif / (1 - sph_alb * rho_unif)

    Required variables (per band): ``rho_unif``, ``rho_atm``,
    ``tdir_up``, ``tdif_up``, ``tdir_down``, ``tdif_down``, ``sph_alb``.

    Produced variable: ``rho_toa``.
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
