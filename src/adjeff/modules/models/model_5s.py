"""Define all the links between reflectance variables in the 5S model.

Implemented links:
1) Toa2Unif : compute uniform reflectance from TOA reflectance.
2) Unif2Toa : inverse operation.
3) Surface2Env : compute environment reflectance from surface reflectance.
"""

from typing import ClassVar

import torch
import xarray as xr

from adjeff.core import ImageDict, PSFDict, extend_analytical
from adjeff.utils import CacheStore, fft_convolve_2D

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


class Surface2Env(SceneModule):
    """Compute Environmental reflectance from surface reflectance.

    Uses a point-spread function to perform the convolution.
    """

    required_vars: ClassVar[list[str]] = ["rho_s"]
    output_vars: ClassVar[list[str]] = ["rho_env"]

    def __init__(
        self,
        psf_dict: PSFDict,
        cache: CacheStore | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self._psf_dict = psf_dict
        self._device = device
        super().__init__(cache=cache)

    def _config_dict(self) -> dict[str, object]:
        """Return frozen configuration for cache keying."""
        return {"psf_dict": self._psf_dict._cache_dict()}

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Perform convolution of `rho_s` with the PSF kernel for each band."""
        for band in scene.bands:
            ds: xr.Dataset = scene[band]
            kernel = self._psf_dict.kernel(band)

            if ds["rho_s"].adjeff.is_analytical():
                n = ds["rho_s"].sizes["y"]
                k = kernel.sizes["y_psf"]
                rho_s = extend_analytical(ds["rho_s"], n + k - 1)
                rho_env = fft_convolve_2D(
                    rho_s,
                    kernel,
                    padding="constant",
                    conv_type="valid",
                    device=self._device,
                )
                ds["rho_env"] = rho_env.assign_coords(ds["rho_s"].coords)
            else:
                ds["rho_env"] = fft_convolve_2D(
                    ds["rho_s"],
                    kernel,
                    padding="reflect",
                    conv_type="same",
                    device=self._device,
                )

        return scene
