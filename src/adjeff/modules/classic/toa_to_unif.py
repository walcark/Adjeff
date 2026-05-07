"""Invert the 5S radiative transfer model to retrieve uniform reflectance."""

from typing import ClassVar

from adjeff.core import ImageDict

from ..scene_module import SceneModule


class Toa2Unif(SceneModule):
    """Invert the 5S model: compute uniform reflectance from TOA reflectance.

    Assumes the environment reflectance equals the surface reflectance
    (``rho_s = rho_env = rho_unif``).  This is the *inverse* of
    :class:`~adjeff.modules.classic.Unif2Toa`.

    The inversion formula applied per band is::

        rho_toa_star = rho_toa - rho_atm
        t_up = tdir_up + tdif_up
        t_down = tdir_down + tdif_down
        rho_unif = rho_toa_star / (sph_alb * rho_toa_star + t_up * t_down)

    Required variables (per band): ``rho_toa``, ``rho_atm``,
    ``tdir_up``, ``tdif_up``, ``tdir_down``, ``tdif_down``, ``sph_alb``.

    Produced variable: ``rho_unif``.
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
