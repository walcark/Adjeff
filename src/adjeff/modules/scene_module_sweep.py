"""SceneModuleSweep: SceneModule with a SweepBundle-driven parameter sweep."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import xarray as xr

from adjeff.sweep import SweepBundle, UniqueIndex
from adjeff.sweep.bundle import _aggregate
from adjeff.utils import CacheStore, ConfigProtocol

from .scene_module import SceneModule

if TYPE_CHECKING:
    from adjeff.core import ImageDict


class SceneModuleSweep(SceneModule):
    """Extension of :class:`SceneModule` for Smart-G parameter-space sweeps.

    A *sweep* is the operation of running the underlying physics function
    (typically a Smart-G simulation) over every combination of atmospheric
    parameters requested by the caller, then stacking all results into a
    multi-dimensional :class:`xr.DataArray`.

    :class:`SceneModuleSweep` encapsulates this sweep logic so that
    subclasses only need to declare *which* parameters to iterate over
    and *how* to call the physics function — the loop, coordinate
    alignment, chunking, and optional spatial deduplication are all
    handled here.

    Scalar vs. vector dimensions
    ----------------------------
    Parameters are split into two groups declared as class variables:

    ``scalar_dims`` — iterated **one-by-one** as a Cartesian product.
    At each step a single value is selected and forwarded to the physics
    function.  Use this for parameters that require a separate Smart-G
    call per value (e.g. ``"aot"``, ``"rh"``, ``"sza"``).

    ``vector_dims`` — passed **whole** (or in chunks) to the physics
    function at every scalar step.  Use this for parameters that
    Smart-G handles in a single vectorised call (e.g. ``"wl"``).

    Example: with ``scalar_dims = ["aot", "rh"]``,
    ``vector_dims = ["wl"]``, ``aot ∈ {0.1, 0.3}``, ``rh ∈ {50, 80}``
    and ``wl = [440, 560, 665]``, the physics function is called 4 times
    (2 × 2), each time receiving all 3 wavelengths.  The result has
    shape ``(aot=2, rh=2, wl=3)``.

    Spatial deduplication
    ----------------------
    When atmospheric inputs are 2-D spatial maps (one ``aot``/``rh``
    value per pixel), the number of *unique* ``(aot, rh)`` pairs is
    often much smaller than the full grid.  Passing
    ``deduplicate_dims=["x", "y"]`` activates
    :class:`~adjeff.sweep.UniqueIndex`: the grid is collapsed to its
    unique rows before the sweep, and the full spatial result is
    restored afterwards.  Without deduplication, Smart-G would be
    called once per pixel, which is prohibitively expensive.

    How to subclass
    ---------------
    1. Declare ``scalar_dims``, ``vector_dims``, ``required_vars`` and
       ``output_vars`` as :class:`~typing.ClassVar` attributes.
    2. Store config objects in ``__init__`` and forward ``cache``,
       ``sweep_chunks`` and ``deduplicate_dims`` to ``super().__init__``.
    3. Implement :meth:`_get_configs` — return a tuple of config objects
       (:class:`~adjeff.atmosphere.AtmoConfig`,
       :class:`~adjeff.atmosphere.GeoConfig`,
       :class:`~adjeff.atmosphere.SpectralConfig`) from which the bundle
       extracts the named arrays.
    4. Implement :meth:`_compute` — call
       :meth:`_apply_bundle` and write the result into the scene bands.

    Minimal example::

        class MyQuantitySampler(SceneModuleSweep):
            required_vars: ClassVar[list[str]] = []
            output_vars: ClassVar[list[str]] = ["my_var"]
            scalar_dims: ClassVar[list[str]] = ["aot", "rh"]
            vector_dims: ClassVar[list[str]] = ["wl"]

            def __init__(self, atmo_config, spectral_config, **kw):
                self.atmo_config = atmo_config
                self.spectral_config = spectral_config
                super().__init__(**kw)

            def _get_configs(self):
                return (self.atmo_config, self.spectral_config)

            def _compute(self, scene):
                arr = self._apply_bundle(smartg_core_fn)
                for band in self.spectral_config.bands:
                    scene[band]["my_var"] = arr.sel(wl=band.wl_nm)
                return scene

    Parameters
    ----------
    cache : CacheStore or None
        Disk cache for computed outputs.  ``None`` disables caching.
    sweep_chunks : dict[str, int] or None
        Maximum slice size per vector dimension forwarded to the physics
        function, e.g. ``{"wl": 4}``.  Useful to limit GPU memory
        usage.  ``None`` passes the whole vector at once.
    deduplicate_dims : list[str] or None
        Spatial dimensions to collapse before sweeping.  Activates
        :class:`~adjeff.sweep.UniqueIndex` when set.  ``None`` disables
        deduplication.
    """

    scalar_dims: ClassVar[list[str]] = []
    vector_dims: ClassVar[list[str]] = []

    def __init__(
        self,
        cache: CacheStore | None = None,
        sweep_chunks: dict[str, int] | None = None,
        deduplicate_dims: list[str] | None = None,
    ) -> None:
        super().__init__(cache)
        self._sweep_chunks = sweep_chunks
        self._deduplicate_dims = deduplicate_dims

    @abstractmethod
    def _get_configs(self) -> tuple[ConfigProtocol, ...]:
        """Return the config instances to aggregate into the bundle."""

    @abstractmethod
    def _compute(self, scene: "ImageDict") -> "ImageDict":
        """Run the core transform using :meth:`_apply_bundle`."""

    def _make_bundle(self) -> tuple[SweepBundle, UniqueIndex | None]:
        """Build a :class:`~adjeff.sweep.SweepBundle` from the current configs.

        Collects the DataArrays named in ``scalar_dims`` and
        ``vector_dims`` from the configs returned by
        :meth:`_get_configs`.  If ``deduplicate_dims`` is set, applies
        :class:`~adjeff.sweep.UniqueIndex` deduplication before building
        the bundle, then validates that no scalar retains more than one
        dimension.

        Returns
        -------
        tuple[SweepBundle, UniqueIndex or None]
            The bundle ready to be passed to
            :meth:`~adjeff.sweep.SweepBundle.apply`, and the
            deduplication index (``None`` if deduplication is inactive).

        Raises
        ------
        ValueError
            If a scalar field remains multi-dimensional after
            deduplication — add its extra dimensions to
            ``deduplicate_dims``.
        """
        all_names = self.scalar_dims + self.vector_dims
        das, _ = _aggregate(list(self._get_configs()), all_names)

        dedup: UniqueIndex | None = None
        if self._deduplicate_dims:
            dedup, das = UniqueIndex.build(das, self._deduplicate_dims)

        for name in self.scalar_dims:
            if name in das and das[name].ndim > 1:
                raise ValueError(
                    f"Scalar '{name}' has shape {das[name].shape} after "
                    "deduplication. Add its dimensions to deduplicate_dims."
                )

        return SweepBundle(
            scalars={k: das[k] for k in self.scalar_dims if k in das},
            vectors={k: das[k] for k in self.vector_dims if k in das},
            sweep_chunks=self._sweep_chunks,
        ), dedup

    def _apply_bundle(
        self,
        func: Callable[..., xr.DataArray],
        **kwargs: Any,
    ) -> xr.DataArray:
        """Build the bundle, apply *func*, and expand deduplication.

        This is the single entry point that subclasses should call from
        their :meth:`_compute` implementation.  It wraps
        :meth:`_make_bundle` and :meth:`~adjeff.sweep.SweepBundle.apply`
        and transparently re-expands the spatial grid when deduplication
        was active.

        Parameters
        ----------
        func : callable
            Core physics function.  Receives keyword arguments drawn
            from the bundle (one scalar value per ``scalar_dims`` field
            at the current step, plus vector chunks for ``vector_dims``
            fields) filtered to the names it explicitly declares.  Must
            return an unnamed :class:`xr.DataArray`.
        **kwargs
            Extra fixed arguments forwarded verbatim to every call of
            *func* (e.g. ``species``, ``afgl_type``, ``n_ph``).

        Returns
        -------
        xr.DataArray
            Result stacked over the full parameter space.  If
            deduplication was active, the ``"index"`` dimension is
            expanded back to the original spatial grid before returning.
        """
        bundle, dedup = self._make_bundle()
        arr = bundle.apply(func, **kwargs)
        if dedup is not None:
            arr = dedup.expand(arr)
        return arr
