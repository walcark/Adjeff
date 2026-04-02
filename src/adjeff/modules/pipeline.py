"""Generic pipeline for chaining SceneModuleSweep instances."""

from __future__ import annotations

from adjeff.core import ImageDict

from .scene_module import SceneModule


class Pipeline:
    """Ordered sequence of Modules instances applied to an ImageDict.

    Validates at construction that the required_vars of each module are
    satisfied by the output_vars of the preceding modules.

    Parameters
    ----------
    modules : list[SceneModule]
        List of modules to chain in the pipeline.
    """

    def __init__(self, modules: list[SceneModule]) -> None:
        self._modules = list(modules)
        self._validate_chain()

    def _validate_chain(self) -> None:
        available: set[str] = set()
        for mod in self._modules:
            missing = set(mod.required_vars) - available
            if missing:
                raise ValueError(
                    f"{type(mod).__name__} requires {missing}, "
                    f"but only {available} are available."
                )
            available.update(mod.output_vars)

    def __call__(self, scene: ImageDict) -> ImageDict:
        """Apply all modules in order."""
        for mod in self._modules:
            scene = mod(scene)
        return scene

    @property
    def output_vars(self) -> list[str]:
        """All variables produced by the pipeline, in order."""
        result = []
        for mod in self._modules:
            result.extend(mod.output_vars)
        return result
