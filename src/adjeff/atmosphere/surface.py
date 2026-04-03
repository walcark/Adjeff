"""Methods to instantiate Smart-G surface objects from a ground image.

Ground images in adjeff are either arbitrary (real image, complex scene)
or analytical (gaussian, disk) shapes. The following methods instantiate
both the Smart-G `Environment` and `Surface` from this knowledge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from smartg.water import Albedo_cst

if TYPE_CHECKING:
    from smartg.smartg import Entity, Environment, LambSurface


class SurfaceFactory:
    """Class that computes Smart-G surface-related objects."""

    def __init__(self) -> None:
        pass

    def entity(self, arr: xr.Dataset) -> Entity:
        """Return an entity object based on input image coordinates."""
        from smartg.smartg import Entity

        return Entity()

    def surface(self, arr: xr.Dataset) -> LambSurface:
        """Return a Lambertian Surface object based on the input image."""
        from smartg.smartg import LambSurface

        kind = arr["rho_s"].adjeff.kind()
        if kind == "analytical":
            rho_max = arr["rho_s"].adjeff.params().get("rho_max")
            return LambSurface(Albedo_cst(rho_max))
        elif kind == "arbitrary":
            rho_s_avg: float = float(np.mean(arr["rho_s"]))
            return LambSurface(Albedo_cst(rho_s_avg))
        else:
            raise ValueError(f"Wrong kind of surface: {kind}")

    def environment(self, arr: xr.Dataset) -> Environment:
        """Return an Environment object based on the input image."""
        from smartg.smartg import Environment

        kind = arr["rho_s"].adjeff.kind()
        if kind == "analytical":
            params = arr["rho_s"].adjeff.params()
            model = arr["rho_s"].adjeff.model()
            return analytical_environment(model, params)
        elif kind == "arbitrary":
            return arbitrary_environment(arr)
        else:
            return Environment()


def analytical_environment(
    model: str, params: dict[str, float]
) -> Environment:
    """Return the Environment for an analytical surface."""
    from smartg.smartg import Environment

    if model == "gauss":
        return Environment(
            ENV=2,
            ENV_SIZE=2 * params["sigma"] ** 2,
            ALB=Albedo_cst(params["rho_min"]),
        )

    if model == "disk":
        return Environment(
            ENV=1,
            ENV_SIZE=params["radius"],
            ALB=Albedo_cst(params["rho_min"]),
        )

    else:
        raise NotImplementedError(f"Model {model} not handled.")


def arbitrary_environment(arr: xr.Dataset) -> Environment:
    """Return the Environment for an arbitrary surface."""
    from smartg.smartg import Environment

    return Environment()
