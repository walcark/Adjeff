"""5S forward model, from uniform surface to TOA reflectance.

Classes
-------
    Unif2Toa
        Computes ``rho_toa`` from ``rho_unif`` and the six 5S terms.
"""

from typing import ClassVar

import xarray as xr

from adjeff.core import ImageDict

from ..scene_module import SceneModule


class Unif2Toa(SceneModule):
    """5S forward model, for a uniform Lambertian surface.

        rho_toa = rho_atm + T_up T_down rho_unif / (1 - sph_alb rho_unif)

    with ``T = tdir + tdif``.  Inverse of :class:`Toa2Unif`.
    """

    _required_vars: ClassVar[list[str]] = [
        "rho_unif",
        "tdir_up",
        "tdif_up",
        "tdir_down",
        "tdif_down",
        "rho_atm",
        "sph_alb",
    ]
    _output_vars: ClassVar[list[str]] = ["rho_toa"]

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Use the 5S model, assuming rho_s=rho_env=rho_unif."""
        for band in scene.bands:
            ds: xr.Dataset = scene[band]
            at = self._slot
            t_up = ds[at("tdir_up")] + ds[at("tdif_up")]
            t_down = ds[at("tdir_down")] + ds[at("tdif_down")]
            rho_unif = ds[at("rho_unif")]
            frac = rho_unif / (1 - ds[at("sph_alb")] * rho_unif)
            ds[at("rho_toa")] = ds[at("rho_atm")] + t_up * t_down * frac
        return scene
