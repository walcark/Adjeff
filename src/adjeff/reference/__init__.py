"""Published methods adjeff is compared against, one module per paper.

These are other people's models, reimplemented for a like-for-like
comparison and kept apart from :mod:`adjeff.modules` for that reason.

Classes
-------
    WuPsfSampler
        Monte-Carlo sampled PSF of Wu et al. (2024), without
        earth-atmosphere coupling.
"""

from .wu2024 import WuPsfSampler

__all__ = ["WuPsfSampler"]
