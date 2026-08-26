"""Base classes for all adjeff scene transformation modules."""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import joblib  # type: ignore[import-untyped]
import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from adjeff.exceptions import ComputationError, ConfigurationError
from adjeff.utils import CacheStore
from adjeff.utils._config import _Config

from .._logging import get_logger, run_context, timed

if TYPE_CHECKING:
    from adjeff.core import ImageDict
    from adjeff.core._psf import PSFModule
    from adjeff.core.bands import SensorBand

logger = get_logger(__name__)


class SceneModule:
    """Base class for all adjeff scene transforms.

    Operates on :class:`~adjeff.core.ImageDict` — one
    :class:`xr.Dataset` per sensor band.

    Subclasses must declare two :class:`~typing.ClassVar` attributes:

    ``required_vars`` — variable names that *must* be present in every
    band Dataset of the input scene.  Validated before ``_compute`` is
    called; raises :class:`~adjeff.exceptions.MissingVariableError` on
    any missing variable.

    Example: with ``required_vars = ["rho_s"]``, this input is valid::

        ImageDict(
            {
                S2Band.B02: Dataset(["rho_s", "rho_toa"]),
                S2Band.B03: Dataset(["rho_s", "rho_toa"]),
            }
        )

    but this one is not (``rho_s`` is absent)::

        ImageDict(
            {
                S2Band.B02: Dataset(["rho_unif", "rho_toa"]),
                S2Band.B03: Dataset(["rho_unif", "rho_toa"]),
            }
        )

    ``output_vars`` — variable names that the module *writes* into the
    output scene.  Used to key the disk cache: only these variables are
    saved and restored across calls.

    Example: with ``output_vars = ["rho_toa"]``, an input containing
    ``rho_s`` becomes (``rho_s`` is preserved, ``rho_toa`` is added)::

        # input
        ImageDict({S2Band.B02: Dataset(["rho_s"]), ...})
        # output
        ImageDict({S2Band.B02: Dataset(["rho_s", "rho_toa"]), ...})

    Execution flow (handled by :meth:`forward`)
    --------------------------------------------
    1. Shallow-copy the input so the caller's data is never mutated.
    2. Validate ``required_vars`` against every band Dataset.
    3. Compute a cache key from the module config and input hashes.
    4. Return cached outputs if the key is found; otherwise call
       :meth:`_compute`.
    5. Stamp provenance metadata and persist ``output_vars`` to cache.
    6. Replace in-memory arrays with lazy Zarr-backed views to limit
       peak RAM usage.  **Only when a cache is configured**: with
       ``cache=None`` every output stays in RAM, which a large sweep will
       exhaust without warning.

    Parameters
    ----------
    cache : CacheStore or None
        Disk cache for computed outputs.  ``None`` disables caching, and
        with it the lazy Zarr views of step 6: outputs are then held in
        memory for as long as the caller keeps the scene.
    """

    _required_vars: ClassVar[list[str]] = []
    _output_vars: ClassVar[list[str]] = []
    #: Variables the module consumes when the scene carries them and
    #: computes itself when it does not.  They enter the cache key only
    #: when present: declaring them in ``required_vars`` would forbid the
    #: standalone call that produces them, while leaving them out
    #: entirely would let two different inputs share one entry.
    _optional_vars: ClassVar[list[str]] = []

    def __init__(
        self,
        cache: CacheStore | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._cache = cache if cache is not None else CacheStore()
        self._log = logger.bind(module=type(self).__name__)
        self.rename = dict(rename or {})
        roles = {*self._required_vars, *self._output_vars, *self._optional_vars}
        unknown = sorted(set(self.rename) - roles)
        if unknown:
            known = ", ".join(sorted(roles)) or "none"
            raise ConfigurationError(
                f"{type(self).__name__} has no role named {unknown!r}; "
                f"it declares: {known}."
            )

    def _slot(self, role: str) -> str:
        """Return the Dataset name *role* is read from or written to."""
        return self.rename.get(role, role)

    @property
    def required_vars(self) -> list[str]:
        """Slot names this module reads."""
        return [self._slot(role) for role in self._required_vars]

    @property
    def output_vars(self) -> list[str]:
        """Slot names this module writes."""
        return [self._slot(role) for role in self._output_vars]

    @property
    def optional_vars(self) -> list[str]:
        """Slot names this module reuses when the scene carries them."""
        return [self._slot(role) for role in self._optional_vars]

    def __call__(self, scene: "ImageDict") -> "ImageDict":
        """Apply the module to *scene*."""
        return self.forward(scene)

    def forward(self, scene: "ImageDict") -> "ImageDict":
        """Apply the module to *scene* and return the enriched scene.

        Parameters
        ----------
        scene : ImageDict
            Input scene.  Shallow-copied internally so the caller's
            data is never mutated.

        Returns
        -------
        ImageDict
            Scene enriched with ``output_vars`` (computed or from
            cache).

        Raises
        ------
        MissingVariableError
            If any band Dataset is missing a variable in
            ``required_vars``.
        """
        scene = scene.shallow_copy()
        scene.require_vars(self.required_vars)

        key = self._cache_key(scene)
        log = self._log.bind(key=key[:8])

        # The module name goes into the context, not only onto `log`:
        # a warning raised by a helper three frames down carries it too,
        # and those are the lines whose origin is hardest to guess.
        with (
            run_context(module=type(self).__name__),
            timed(log, "module", bands=len(scene.bands)) as outcome,
        ):
            cached = self._cache.load_vars(key, scene.bands, self._output_vars)
            if cached is not None:
                self._write_roles(scene, cached)
                outcome["cached"] = True
                return scene

            outcome["cached"] = False
            scene = self._compute(scene)
            self._reject_non_finite(scene)
            self._stamp_provenance(scene, key)
            self._cache.save_vars(key, self._role_view(scene), self._output_vars)
            # Replace in-memory arrays with lazy Zarr-backed views so large
            # outputs (e.g. rho_toa at all atmospheric combos) are not kept
            # fully in RAM when the caller stores multiple scenes.
            lazy = self._cache.load_vars(key, scene.bands, self._output_vars)
            if lazy is not None:
                self._write_roles(scene, lazy)
            return scene

    def _reject_non_finite(self, scene: "ImageDict") -> None:
        """Raise when an output holds NaN or infinity, before it is cached.

        Smart-G returns NaN rather than raising when it cannot run,
        whether the GPU is busy or its auxiliary data is not where
        ``SMARTG_DIR_AUXDATA`` says.  Cached, that result becomes permanent: every later
        run reads it back and fails somewhere far away, on an
        interpolation or a solver, with nothing pointing at a simulation
        that ran minutes or days earlier.  Checking here costs one pass
        over each output and turns a silent poisoning into an error at
        the place that caused it.

        Raises
        ------
        ComputationError
            If any output variable of any band holds a non-finite value.
        """
        for band in scene.bands:
            ds = scene[band]
            for role in self._output_vars:
                slot = self._slot(role)
                if slot not in ds:
                    continue
                values = np.asarray(ds[slot].values)
                if values.dtype.kind not in "fc":
                    continue
                finite = np.isfinite(values)
                if bool(finite.all()):
                    continue
                bad = int(values.size - finite.sum())
                raise ComputationError(
                    f"{type(self).__name__} produced {bad} non-finite "
                    f"value(s) out of {values.size} in {slot!r} for band "
                    f"{band}.  Nothing was cached.  Smart-G returns NaN "
                    "instead of raising when it cannot run, so check "
                    "that SMARTG_DIR_AUXDATA points at the auxiliary "
                    "data and that no other process is holding the GPU."
                )

    def _role_view(self, scene: "ImageDict") -> "ImageDict":
        """Return *scene*'s outputs under their role names.

        The cache is keyed by role, so it must be filled by role too:
        two runs that differ only by where they put their result share
        one entry, and either of them can read it back.
        """
        from adjeff.core import ImageDict

        return ImageDict(
            {
                band: xr.Dataset(
                    {
                        role: scene[band][self._slot(role)]
                        for role in self._output_vars
                        if self._slot(role) in scene[band]
                    }
                )
                for band in scene.bands
            }
        )

    def _write_roles(
        self, scene: "ImageDict", by_band: dict[Any, dict[str, xr.DataArray]]
    ) -> None:
        """Write role-named arrays into the slots this instance uses."""
        for band, var_map in by_band.items():
            ds = scene[band]
            for role, da in var_map.items():
                ds[self._slot(role)] = da

    @abstractmethod
    def _compute(self, scene: "ImageDict") -> "ImageDict":
        """Run the core transform."""

    def _cache_key(self, scene: "ImageDict") -> str:
        """Return a joblib hash of module type, config, and input hashes."""
        return str(
            joblib.hash(
                {
                    "module": type(self).__name__,
                    "config": self._config_dict(),
                    "inputs": self._input_hashes(scene),
                }
            )
        )

    # Parameters excluded from auto-detection: they're infrastructure, not
    # computation config (don't affect the output value for given inputs).
    _INFRA_PARAMS: ClassVar[frozenset[str]] = frozenset(
        ("self", "cache", "chunks", "rename")
    )

    def _config_dict(self) -> dict[str, object]:
        """Return frozen configuration for cache keying.

        Auto-detects public ``__init__`` parameters stored as same-named
        instance attributes, excluding infrastructure params listed in
        ``_INFRA_PARAMS`` (``cache``, ``chunks``) that do not affect
        output values.

        Subclasses with privately-stored params (e.g. ``_psfs``)
        must override this method.

        Raises
        ------
        ConfigurationError
            If a parameter is neither excluded nor readable as a
            same-named attribute.  Auto-detection is an implicit
            contract: storing a parameter under a private name silently
            drops it from the key, and two runs that differ only by that
            parameter then collide on the same cache entry.  Failing at
            the first lookup turns that into a development error rather
            than a wrong result read back from disk months later.
        """
        sig = inspect.signature(type(self).__init__)
        wanted = [name for name in sig.parameters if name not in self._INFRA_PARAMS]
        missing = [name for name in wanted if not hasattr(self, name)]
        if missing:
            raise ConfigurationError(
                f"{type(self).__name__} does not expose {missing!r} as "
                "attributes, so they cannot enter the cache key. Store "
                "them under their own name, add them to _INFRA_PARAMS if "
                "they cannot change the output, or override _config_dict."
            )
        raw = {name: getattr(self, name) for name in wanted}
        return {
            k: v._stable_hash_repr if isinstance(v, _Config) else v
            for k, v in raw.items()
        }

    def _input_hashes(self, scene: "ImageDict") -> dict[str, str]:
        """Return a stable hash per ``(band, variable)`` pair in *scene*.

        Covers ``required_vars`` plus whichever ``optional_vars`` the
        scene happens to carry.  Uses the DataArray's provenance key when
        available to avoid re-hashing large arrays; falls back to
        ``joblib.hash`` of the raw values.
        """
        hashes: dict[str, str] = {}
        for band in scene.bands:
            ds = scene[band]
            present = [role for role in self._optional_vars if self._slot(role) in ds]
            for role in [*self._required_vars, *present]:
                # Keyed by role, read by slot: two runs that differ only
                # by where they put their result compute the same thing
                # and must share one entry.
                hashes[f"{band}.{role}"] = self._var_hash(ds[self._slot(role)])
        return hashes

    @staticmethod
    def _var_hash(da: xr.DataArray) -> str:
        """Return the provenance key of *da*, or a hash of its values."""
        provenance_key: str | None = da.attrs.get("_adjeff_provenance", {}).get("key")
        return str(
            provenance_key if provenance_key is not None else joblib.hash(da.values)
        )

    def _stamp_provenance(self, scene: "ImageDict", key: str) -> None:
        """Tag each output DataArray with module name and cache key."""
        provenance = {"module": type(self).__name__, "key": key}
        for band in scene.bands:
            ds = scene[band]
            for var in self.output_vars:
                if var in ds:
                    ds[var].attrs["_adjeff_provenance"] = provenance


class TrainableSceneModule(nn.Module, SceneModule):
    """Abstract SceneModule with a differentiable per-band forward pass.

    Inherits from both :class:`torch.nn.Module` (for parameter registration
    and gradient flow) and :class:`SceneModule` (for the xarray pipeline
    contract).  When called as ``model(scene)``, the ``nn.Module.__call__``
    machinery is used (hooks fire, then ``forward`` is dispatched).

    Subclasses must implement :meth:`forward_band` and expose their
    per-band PSF modules via :attr:`psf_modules`.

    These two additions form the contract consumed by
    :func:`~adjeff.optim.fit`.
    """

    def __init__(
        self,
        cache: CacheStore | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        SceneModule.__init__(self, cache=cache, rename=rename)

    def forward(self, scene: "ImageDict") -> "ImageDict":
        """Delegate to :meth:`SceneModule.forward` (resolves MRO ambiguity)."""
        return SceneModule.forward(self, scene)

    @property
    @abstractmethod
    def psf_modules(self) -> "dict[str, PSFModule]":
        """Mapping of band IDs to PSF modules."""

    @abstractmethod
    def forward_band(
        self,
        band: "SensorBand",
        *,
        kernel: torch.Tensor | None = None,
        **inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable per-band forward pass for the training loop.

        *kernel* overrides the band's own PSF for one call, without
        installing it in the model.  It is part of the contract because
        that is what evaluating a candidate costs: mapping a loss
        surface would otherwise have to reach into the model's private
        state, or reimplement its forward pass.
        """
