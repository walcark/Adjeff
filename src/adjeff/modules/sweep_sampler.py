"""SceneModule whose physics runs through an xsweep batched sweep."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import numpy as np
import xarray as xr
from xsweep import Sweeper, SweepPolicy

from adjeff.utils import CacheStore
from adjeff.utils._config import ConfigProtocol

from .scene_module import SceneModule

if TYPE_CHECKING:
    from adjeff.core import ImageDict


class SweepSampler(SceneModule):
    """SceneModule whose physics runs through an xsweep batched sweep.

    Subclasses declare:

    - ``contract``: the xsweep contract of one call, atmospheric and
      geometric parameters in ``batch(...)``, wavelength in ``vec(...)``,
      e.g. ``"batch(aot, rh, h, href, sza) vec(wl) -> tdir_down(wl)"``;
    - ``point_fn``: the physics, a function of :mod:`._smartg`;
    - :meth:`_get_configs`: the configs the swept values come from.

    The inherited :meth:`_compute` runs the sweep and writes the output
    into every band of ``spectral_config``.

    Parameters
    ----------
    cache, rename : optional
        See :class:`SceneModule`.
    batch_size : int, optional
        States per Smart-G call, 64 by default.  Changes the cost only.
    dedup : bool, optional
        Merge identical states before calling.  Worth it for spatial
        maps, overhead otherwise.
    """

    contract: ClassVar[str]
    point_fn: ClassVar[Callable[..., xr.DataArray]]

    def __init__(
        self,
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
        rename: dict[str, str] | None = None,
    ) -> None:
        super().__init__(cache, rename=rename)
        # Public, so that _config_dict reads them.
        self.batch_size = batch_size
        self.dedup = dedup

    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        """Return the config instances the sweep space is drawn from."""
        raise NotImplementedError

    def _statics(self) -> dict[str, Any]:
        """Return the keywords forwarded verbatim to every call."""
        return {}

    def _space(self) -> xr.Dataset:
        """Collect the swept parameters from the configs into one Dataset.

        Coordinates are dropped: two arrays sharing a spatial dim would
        otherwise claim conflicting labels for it.
        """
        wanted = set(self._contract.inputs)
        arrays: dict[str, xr.DataArray] = {}
        for config in self._get_configs():
            for name, array in config._arrays.items():
                if name in wanted and name not in arrays:
                    arrays[name] = array.drop_vars(list(array.coords), errors="ignore")
        return xr.Dataset(arrays)

    @property
    def _contract(self) -> Any:
        """Return the parsed contract, built once per class."""
        from xsweep.contract import coerce

        return coerce(type(self).contract)

    @staticmethod
    def _restore_coords(arr: xr.DataArray, source: xr.Dataset) -> xr.DataArray:
        """Give *arr* the coordinates *source* holds for its dims.

        xsweep labels the dims it does not sweep (``y``, ``x``) by
        position; assigned as they are, they would misalign with the
        scene and leave NaN.
        """
        shared = {
            str(dim): source.coords[dim]
            for dim in arr.dims
            if dim in source.coords and dim in source.dims
        }
        return arr.assign_coords(shared) if shared else arr

    def _sweep(self, **bound: Any) -> xr.DataArray:
        """Run the sweep and return its output.

        *bound* is passed to the physics as keywords, outside xsweep's
        fingerprint: it is scene data, keyed by the input hashes.
        """
        func = type(self).point_fn
        if bound:
            func = functools.partial(func, **bound)
        sweeper = Sweeper(type(self).contract, func)
        space = self._space()
        statics = self._statics()

        # Announce the cost before paying it.
        spectral = getattr(self, "spectral_config", None)
        plan: dict[str, Any] = {
            "states": int(np.prod([space.sizes[d] for d in space.dims]) or 1),
            "n_ph": statics.get("n_ph"),
            "batch_size": self.batch_size,
            "dedup": self.dedup,
        }
        if spectral is not None:
            plan["bands"] = len(spectral.bands)
        self._log.info("sweep.plan", **plan)
        result = sweeper(
            space,
            policy=SweepPolicy(batch_size=self.batch_size, dedup=self.dedup),
            **statics,
        )
        out: xr.DataArray = result[self._contract.outputs[0]]
        return out

    def _compute(self, scene: "ImageDict") -> "ImageDict":
        """Run the sweep and write its output into every declared band."""
        name = self._contract.outputs[0]
        arr = self._sweep()
        for band in self.spectral_config.bands:  # type: ignore[attr-defined]
            if band not in scene.bands:
                scene[band] = xr.Dataset()
            scene[band][self._slot(name)] = arr.sel(wl=band.wl_nm)
        return scene
