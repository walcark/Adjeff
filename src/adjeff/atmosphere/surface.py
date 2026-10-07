"""Smart-G surface and environment of an adjeff scene.

An analytical scene (Gaussian, disc) maps to a built-in Smart-G
environment. An arbitrary one is encoded as an ``AlbedoMap``.

Classes
-------
    SurfaceFactory
        Builds the ``LambSurface`` and ``Environment`` of a scene.

Functions
---------
    analytical_environment
        Built-in Smart-G environment of a Gaussian or disc surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from adjeff.exceptions import ConfigurationError

if TYPE_CHECKING:
    from smartg.objects3d import Entity
    from smartg.surface import Environment, LambSurface


class SurfaceFactory:
    """Builds the Smart-G surface and environment of a scene.

    Parameters
    ----------
    rho_background : float or {"mean", "min", "zero"}, optional
        Reflectance outside the ``AlbedoMap`` of an arbitrary scene: a
        value, or the field's mean, its minimum, or zero.  Ignored for an
        analytical scene.  ``"mean"`` by default.
    """

    def __init__(
        self,
        rho_background: float | Literal["mean", "min", "zero"] = "mean",
    ) -> None:
        self._rho_background = rho_background

    def surface(self, arr: xr.Dataset) -> LambSurface:
        """Return a Lambertian Surface object based on the input image."""
        from smartg.albedo import AlbedoCst
        from smartg.surface import LambSurface

        kind = arr["rho_s"].adjeff.kind()
        if kind == "analytical":
            rho_max = arr["rho_s"].adjeff.params().get("rho_max")
            return LambSurface(AlbedoCst(rho_max))
        elif kind == "arbitrary":
            if isinstance(self._rho_background, float):
                rho: float = self._rho_background
            elif self._rho_background == "mean":
                rho = float(np.mean(arr["rho_s"].values))
            elif self._rho_background == "min":
                rho = float(np.min(arr["rho_s"].values))
            else:  # "zero"
                rho = 0.0
            return LambSurface(AlbedoCst(rho))
        else:
            raise ConfigurationError(f"Wrong kind of surface: {kind}")

    def environment(self, arr: xr.Dataset) -> Environment:
        """Return an Environment object based on the input image.

        For analytical surfaces (gaussian, disk) a lightweight Smart-G
        built-in environment is used.  For arbitrary surfaces the full 2D
        albedo map is encoded via :meth:`custom_environment`.
        """
        kind = arr["rho_s"].adjeff.kind()
        if kind == "analytical":
            params = arr["rho_s"].adjeff.params()
            model = arr["rho_s"].adjeff.model()
            return analytical_environment(model, params)
        elif kind == "arbitrary":
            return self.custom_environment(arr)
        else:
            from smartg.surface import Environment

            return Environment()

    def custom_environment(
        self,
        arr: xr.Dataset,
        n_alb: int = 1000,
    ) -> Environment:
        """Return an ``AlbedoMap`` environment for an arbitrary scene.

        The reflectance is quantised to the nearest of *n_alb* levels
        spread evenly between its minimum (clipped at 0) and maximum.

        Parameters
        ----------
        arr : xr.Dataset
            Scene holding ``"rho_s"`` on dims ``(y, x)``.
        n_alb : int, optional
            Number of reflectance levels, 1000 by default.
        """
        from smartg.albedo import AlbedoCst, AlbedoMap
        from smartg.surface import Environment

        da = arr["rho_s"]
        # adjeff stores (y, x); Smart-G AlbedoMap expects (x, y) ordering
        rho_s_vals: np.ndarray = da.values.T.astype(np.float64)  # (nx, ny)
        x_coords: np.ndarray = da.coords["x"].values
        y_coords: np.ndarray = da.coords["y"].values
        nx, ny = rho_s_vals.shape

        mini = max(0.0, float(np.min(rho_s_vals)))
        maxi = float(np.max(rho_s_vals))
        albs_vals = np.linspace(mini, maxi, n_alb)
        # Nearest-neighbour quantisation on a uniform linspace
        step = (maxi - mini) / (n_alb - 1) if n_alb > 1 else 1.0
        idx = np.clip(
            np.round((rho_s_vals.ravel() - mini) / step).astype(int),
            0,
            n_alb - 1,
        )
        # rhos_idx[0, :] and rhos_idx[:, 0] stay at -1 (outside boundary)
        rhos_idx = -np.ones((nx + 1, ny + 1), dtype=np.float64)
        rhos_idx[1:, 1:] = idx.reshape(nx, ny).astype(np.float64)

        albs_list = [AlbedoCst(float(np.round(a, 4))) for a in albs_vals]

        res_x = float(x_coords[1] - x_coords[0])
        res_y = float(y_coords[1] - y_coords[0])
        x_edges = np.append(x_coords - res_x / 2, 1e8)
        y_edges = np.append(y_coords - res_y / 2, 1e8)

        alb_map = AlbedoMap(rhos_idx, x_edges, y_edges, albs_list)
        return Environment(env=5, alb=alb_map)


def analytical_environment(model: str, params: dict[str, float]) -> Environment:
    """Return the built-in Smart-G environment of an analytical surface.

    Parameters
    ----------
    model : {"gauss", "disk"}
        Surface model.
    params : dict[str, float]
        Its parameters: ``sigma`` (Gaussian) or ``radius`` (disc), and
        ``rho_min``.

    Raises
    ------
    NotImplementedError
        For any other model.
    """
    from smartg.albedo import AlbedoCst
    from smartg.surface import Environment

    if model == "gauss":
        return Environment(
            env=2,
            env_size=2 * params["sigma"] ** 2,
            alb=AlbedoCst(params["rho_min"]),
        )

    if model == "disk":
        return Environment(
            env=1,
            env_size=params["radius"],
            alb=AlbedoCst(params["rho_min"]),
        )

    else:
        raise NotImplementedError(f"Model {model} not handled.")
