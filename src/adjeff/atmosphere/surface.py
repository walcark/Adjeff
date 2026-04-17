"""Methods to instantiate Smart-G surface objects from a ground image.

Ground images in adjeff are either arbitrary (real image, complex scene)
or analytical (gaussian, disk) shapes. The following methods instantiate
both the Smart-G `Environment` and `Surface` from this knowledge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from smartg.water import Albedo_cst

if TYPE_CHECKING:
    from smartg.smartg import Entity, Environment, LambSurface


class SurfaceFactory:
    """Class that computes Smart-G surface-related objects.

    Parameters
    ----------
    rho_background : float | "mean" | "min" | "zero"
        Reflectance passed to ``LambSurface`` for arbitrary fields — used by
        Smart-G for photons that leave the ``Albedo_map`` region.  A float
        sets it explicitly; the string options derive it from the field:
        ``"mean"`` (spatial average), ``"min"`` (background value),
        ``"zero"`` (absorbing boundary).  Default is ``"mean"``.
        Ignored for analytical surfaces (their ``rho_min`` is used instead).
    """

    def __init__(
        self,
        rho_background: float | Literal["mean", "min", "zero"] = "mean",
    ) -> None:
        self._rho_background = rho_background

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
            if isinstance(self._rho_background, float):
                rho: float = self._rho_background
            elif self._rho_background == "mean":
                rho = float(np.mean(arr["rho_s"].values))
            elif self._rho_background == "min":
                rho = float(np.min(arr["rho_s"].values))
            else:  # "zero"
                rho = 0.0
            return LambSurface(Albedo_cst(rho))
        else:
            raise ValueError(f"Wrong kind of surface: {kind}")

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
            from smartg.smartg import Environment

            return Environment()

    def custom_environment(
        self,
        arr: xr.Dataset,
        n_alb: int = 1000,
    ) -> Environment:
        """Return an ``Albedo_map`` Environment from an arbitrary 2D surface.

        The reflectance field is quantised into *n_alb* discrete levels
        (linspace from min to max).  Each pixel is mapped to the index of
        its nearest level, producing the index grid consumed by Smart-G's
        ``Albedo_map``.

        Quantisation is vectorised via :func:`numpy.searchsorted` so it
        scales to large images (e.g. 1999 × 1999) without a Python loop.

        Parameters
        ----------
        arr : xr.Dataset
            Scene dataset containing the ``"rho_s"`` variable with ``x``
            and ``y`` spatial coordinates (adjeff convention: dims ``(y, x)``).
        n_alb : int
            Number of discrete albedo levels (default 1000).

        Returns
        -------
        Environment
            Smart-G Environment with ``ENV=5`` and an ``Albedo_map``.
        """
        from smartg.smartg import Albedo_map, Environment

        da = arr["rho_s"]
        # adjeff stores (y, x); Smart-G Albedo_map expects (x, y) ordering
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

        albs_list = [Albedo_cst(float(np.round(a, 4))) for a in albs_vals]

        res_x = float(x_coords[1] - x_coords[0])
        res_y = float(y_coords[1] - y_coords[0])
        x_edges = np.append(x_coords - res_x / 2, 1e8)
        y_edges = np.append(y_coords - res_y / 2, 1e8)

        alb_map = Albedo_map(rhos_idx, x_edges, y_edges, albs_list)
        return Environment(ENV=5, ALB=alb_map)


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
