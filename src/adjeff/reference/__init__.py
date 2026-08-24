"""Published methods adjeff is compared against, one module per paper.

Nothing here is adjeff's own work.  These are other people's models,
reimplemented so that a comparison can be made on equal footing, and
kept apart from :mod:`adjeff.modules` for exactly that reason: reading
``adjeff.modules.samplers`` should tell you what adjeff computes, not
what someone else published.

Each module is named after its reference and carries the citation in its
docstring, so the provenance is visible in the file tree.

- :mod:`.wu2024` — :class:`WuPsfSampler`, the Monte-Carlo sampled PSF of
  Wu et al. (2024), without earth-atmosphere coupling.

A method belongs here when it is someone else's, whether or not it
shares adjeff's machinery: :class:`WuPsfSampler` is an ordinary
:class:`~adjeff.modules.SweepSampler`.  What sets it apart is
provenance, not shape.
"""

from .wu2024 import WuPsfSampler

__all__ = ["WuPsfSampler"]
