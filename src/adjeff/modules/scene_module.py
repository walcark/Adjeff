"""Base classes of every scene module.

Classes
-------
    SceneModule
        Step reading ``required_vars`` and writing ``output_vars``, with
        caching and provenance.
    TrainableSceneModule
        SceneModule that is also a ``torch.nn.Module`` holding PSFs.
"""

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
    """Step reading variables from an ImageDict and writing new ones.

    Subclasses declare ``_required_vars`` (read, must exist in every
    band), ``_output_vars`` (written, and cached) and optionally
    ``_optional_vars`` (read when present, computed otherwise), then
    implement :meth:`_compute`.

    :meth:`forward` copies the scene, checks the inputs, returns the
    cached outputs when the cache key matches, and otherwise computes,
    rejects non-finite values, stamps provenance and caches.  With a
    cache, outputs are then replaced by lazy Zarr views; without one,
    they stay in RAM.

    Parameters
    ----------
    cache : CacheStore or None, optional
        Disk cache of the outputs.  ``None`` disables it.
    rename : dict[str, str] or None, optional
        Variable each role is read from or written to, when not its own
        name, e.g. ``{"rho_toa": "rho_toa_smartg"}``.
    """

    _required_vars: ClassVar[list[str]] = []
    _output_vars: ClassVar[list[str]] = []
    #: Read when present and then part of the cache key; computed otherwise.
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
        """Return a copy of *scene* with the output variables added.

        Raises
        ------
        MissingVariableError
            If a band lacks a required variable.
        ComputationError
            If an output holds NaN or infinity.
        """
        scene = scene.shallow_copy()
        scene.require_vars(self.required_vars)

        key = self._cache_key(scene)
        log = self._log.bind(key=key[:8])

        # In the context, so that warnings from helpers carry it too.
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
            # Swap the outputs for lazy Zarr views, to free the RAM.
            lazy = self._cache.load_vars(key, scene.bands, self._output_vars)
            if lazy is not None:
                self._write_roles(scene, lazy)
            return scene

    def _reject_non_finite(self, scene: "ImageDict") -> None:
        """Raise if an output holds NaN or infinity, before it is cached.

        Smart-G returns NaN instead of raising when it cannot run; cached,
        that result would poison every later run.

        Raises
        ------
        ComputationError
            Naming the variable, the band and the count.
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
        """Return the configuration the cache key depends on.

        Every ``__init__`` parameter not in ``_INFRA_PARAMS`` is read
        from the same-named attribute.  Override when a parameter is
        stored under another name.

        Raises
        ------
        ConfigurationError
            If a parameter has no same-named attribute, which would
            otherwise drop it from the key silently.
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
    """SceneModule that is also a ``torch.nn.Module`` holding PSFs.

    Subclasses implement :attr:`psf_modules` and :meth:`forward_band`,
    which is what :func:`~adjeff.optim.fit` uses.
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
        """Differentiable forward pass of one band, on tensors.

        *kernel*, when given, replaces the band's PSF for this call only.
        """
