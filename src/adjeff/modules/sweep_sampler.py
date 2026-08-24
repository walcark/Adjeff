"""SceneModule whose physics runs through an xsweep batched sweep."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar

import xarray as xr
from xsweep import Sweeper, SweepPolicy

from adjeff.utils import CacheStore, ConfigProtocol

from .scene_module import SceneModule

if TYPE_CHECKING:
    from adjeff.core import ImageDict


class SweepSampler(SceneModule):
    """Extension of :class:`SceneModule` for Smart-G parameter sweeps.

    A sampler runs one physics function over every atmospheric state the
    caller asks for.  Smart-G is expensive per call and takes many states
    at once, so the states travel in groups: xsweep's ``batch`` clause
    keeps them sweep axes — deduplicated, resumable, addressable in a
    store — while handing the engine a whole group per call.

    Declaring a sampler
    -------------------
    Two class variables and one method:

    ``contract`` — the xsweep contract for a single call.  Atmospheric
    and geometric parameters go in ``batch(...)``, wavelength in
    ``vec(...)`` since Smart-G is vectorised over it and the output
    carries it::

        contract = "batch(aot, rh, h, href, sza) vec(wl) -> tdir_down(wl)"

    ``point_fn`` — the physics, a function of :mod:`._smartg`.  It
    receives the batched parameters as 1-D arrays over ``point``, the
    vector ones whole, and the statics as keywords.

    :meth:`_get_configs` — the config objects the space is drawn from.

    ``_compute`` is then inherited: it builds the space, runs the sweep
    and writes the result into every band of the scene.

    Parameters
    ----------
    cache : CacheStore or None
        Disk cache for computed outputs.  ``None`` disables caching.
    batch_size : int
        How many atmospheric states one Smart-G call receives.  A cost
        decision only: the values it produces do not depend on it.
    dedup : bool
        Collapse repeated states before calling.  Worth it when the
        parameters are spatial maps, where many pixels share a state;
        pure overhead on a sweep where every state is distinct.
    """

    contract: ClassVar[str]
    point_fn: ClassVar[Callable[..., xr.DataArray]]

    def __init__(
        self,
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
    ) -> None:
        super().__init__(cache)
        # Public names: SceneModule._config_dict() reads __init__ params
        # off same-named attributes.  batch_size cannot change a value and
        # dedup cannot either, but both are cheap to hash and leaving them
        # out would mean explaining why, every time someone reads this.
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

        Coordinates are dropped on the way in.  A config auto-assigns each
        array its own values as a coordinate, which collides as soon as two
        arrays share a spatial dim (``aot(x, y)`` and ``rh(x, y)`` would
        each claim different labels for ``x``), and xsweep reads its axes
        from the data rather than from coordinates anyway.
        """
        wanted = set(self._contract.inputs)
        arrays: dict[str, xr.DataArray] = {}
        for config in self._get_configs():
            for name, array in config._arrays.items():
                if name in wanted and name not in arrays:
                    arrays[name] = array.drop_vars(
                        list(array.coords), errors="ignore"
                    )
        return xr.Dataset(arrays)

    @property
    def _contract(self) -> Any:
        """Return the parsed contract, built once per class."""
        from xsweep.contract import coerce

        return coerce(type(self).contract)

    def _sweep(self) -> xr.DataArray:
        """Run the sweep and return the single declared output."""
        sweeper = Sweeper(type(self).contract, type(self).point_fn)
        result = sweeper(
            self._space(),
            policy=SweepPolicy(batch_size=self.batch_size, dedup=self.dedup),
            **self._statics(),
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
            scene[band][name] = arr.sel(wl=band.wl_nm)
        return scene
