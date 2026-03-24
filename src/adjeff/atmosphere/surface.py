"""Methods to instanciate Smart-G surface objects from a ground image.

Ground images in adjeff are either artibrary (real image, complex scene)
or analytical (gaussian, disk) shapes. The following methods instanciate
both the Smart-G `Environment` and `Surface` from this knowledge.
"""

import numpy as np
import xarray as xr
from smartg.smartg import Entity, Environment, LambSurface
from smartg.water import Albedo_cst

from adjeff import AdjeffAccessorError


class SurfaceFactory:
    """Class that computes Smart-G surface-related objects."""

    def __init__(self) -> None:
        pass

    def entity(self, arr: xr.DataArray) -> Entity:
        """Return an entity object based on input image coordinates."""
        return Entity()

    def surface(self, arr: xr.DataArray) -> LambSurface:
        """Return an Lambertian Surface object based on the input image."""
        if arr.adjeff.kind == "analytical":
            return LambSurface(Albedo_cst(arr.adjeff.params["rho_max"]))
        elif arr.adjeff.kind == "arbitrary":
            mean_alb: float = float(np.mean(arr.values))
            return LambSurface(Albedo_cst(mean_alb))
        else:
            raise AdjeffAccessorError(f"Wrong adjeff kind: {arr.adjeff.kind}")

    def environment(self, arr: xr.DataArray) -> Environment:
        """Return an Environment object based on the input image."""
        if arr.adjeff.kind == "analytical":
            return analytical_environment(arr)
        elif arr.adjeff.kind == "arbitrary":
            return arbitrary_environment(arr)
        else:
            raise AdjeffAccessorError(f"Wrong adjeff kind: {arr.adjeff.kind}")


def analytical_environment(arr: xr.DataArray) -> Environment:
    """Return the Environment for an analytical surface."""
    return Environment()


def arbitrary_environment(arr: xr.DataArray) -> Environment:
    """Return the Environment for an analytical surface."""
    return Environment()
