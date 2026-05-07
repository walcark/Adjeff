"""Scene transformation modules for the adjeff radiative pipeline.

The module system is organized in four abstraction levels, from base
classes to domain-specific implementations:

**1. Base classes**

- :class:`SceneModule` — core contract: validates inputs against
  ``required_vars``, checks the disk cache, calls ``_compute``, and
  stamps provenance on outputs.  Every module inherits from it.
- :class:`SceneSource` — variant of :class:`SceneModule` that creates a
  scene from scratch (no required input variables).
- :class:`TrainableSceneModule` — extends :class:`SceneModule` with
  :class:`torch.nn.Module` for gradient-tracked PSF optimisation.
- :class:`SceneModuleSweep` — extends :class:`SceneModule` with a
  :class:`~adjeff.sweep.SweepBundle`-driven parameter sweep, used by
  all Smart-G samplers.

**2. Pipeline**

- :class:`Pipeline` — chains :class:`SceneModule` instances and
  validates at construction time that each module's ``required_vars``
  are produced by its predecessors.

**3. Sub-packages** (in order of complexity)

- :mod:`adjeff.modules.classic` — analytical 5S formulas, no GPU:
  :class:`~adjeff.modules.classic.Toa2Unif`,
  :class:`~adjeff.modules.classic.Unif2Toa`.
- :mod:`adjeff.modules.models` — trainable model:
  :class:`~adjeff.modules.models.Unif2Surface` (5S formula + learnable
  PSF convolution).
- :mod:`adjeff.modules.samplers` — Smart-G Monte-Carlo samplers
  (GPU required): :class:`~adjeff.modules.samplers.RadiativePipeline`
  and the individual transmittance / reflectance samplers.
"""

from .pipeline import Pipeline
from .scene_module import SceneModule, TrainableSceneModule
from .scene_module_sweep import SceneModuleSweep
from .scene_source import SceneSource
from .test_module import TestModule

__all__ = [
    # Base classes
    "SceneModule",
    "SceneSource",
    "TrainableSceneModule",
    "SceneModuleSweep",
    # Pipeline
    "Pipeline",
    # Utilities
    "TestModule",
]
