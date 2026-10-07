"""5S inverse model, from TOA to uniform surface reflectance.

Classes
-------
    Toa2Unif
        Computes ``rho_unif`` from ``rho_toa`` and the six 5S terms.
"""

from typing import ClassVar

from adjeff.core import ImageDict

from ..scene_module import SceneModule


class Toa2Unif(SceneModule):
    """5S inverse model, for a uniform Lambertian surface.

        rho_unif = r / (sph_alb r + T_up T_down),   r = rho_toa - rho_atm

    with ``T = tdir + tdif``.  Inverse of :class:`Unif2Toa`.
    """

    _required_vars: ClassVar[list[str]] = [
        "rho_toa",
        "tdir_up",
        "tdif_up",
        "tdir_down",
        "tdif_down",
        "rho_atm",
        "sph_alb",
    ]
    _output_vars: ClassVar[list[str]] = ["rho_unif"]

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Invert the 5S model, assuming rho_s=rho_env=rho_unif."""
        for band in scene.bands:
            ds = scene[band]
            at = self._slot
            rho_toa_star = ds[at("rho_toa")] - ds[at("rho_atm")]
            t_up = ds[at("tdir_up")] + ds[at("tdif_up")]
            t_down = ds[at("tdir_down")] + ds[at("tdif_down")]
            ds[at("rho_unif")] = rho_toa_star / (
                ds[at("sph_alb")] * rho_toa_star + t_up * t_down
            )
        return scene
