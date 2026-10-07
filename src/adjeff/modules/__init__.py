"""Scene modules: read variables from an ImageDict and add new ones.

Classes
-------
    SceneModule
        Base step: checks inputs, caches outputs, stamps provenance.
    SceneSource
        Step creating a scene from nothing (no required input).
    TrainableSceneModule
        Step holding trainable PSFs, as a ``torch.nn.Module``.
    SweepSampler
        Step whose physics runs through an xsweep batched sweep.
    Pipeline
        Chain of steps, checked at construction.

Sub-packages
------------
    classic
        Closed-form 5S forward and inverse models.
    models
        Trainable PSF-convolution models.
    samplers
        Smart-G Monte-Carlo samplers of the 5S terms (GPU).
    loaders
        Earth observation products read into an ImageDict.
"""

from .pipeline import Pipeline
from .scene_module import SceneModule, TrainableSceneModule
from .scene_source import SceneSource
from .sweep_sampler import SweepSampler

__all__ = [
    # Base classes
    "SceneModule",
    "SceneSource",
    "TrainableSceneModule",
    "SweepSampler",
    # Pipeline
    "Pipeline",
]
