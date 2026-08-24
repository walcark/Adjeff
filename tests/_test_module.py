"""Minimal SceneModule used as a fixture by the test suite.

It lives here rather than in the package: it is a test double, and
exporting it from adjeff.modules put a pytest collection warning on
every run.
"""

from typing import ClassVar

import xarray as xr

from adjeff.core import ImageDict
from adjeff.modules import SceneModule


class TestModule(SceneModule):
    """Simple test module that returns shifted values."""

    required_vars: ClassVar[list[str]] = ["rho_s"]
    output_vars: ClassVar[list[str]] = ["rho_toa"]

    def _compute(self, scene: ImageDict) -> ImageDict:
        """Shifting input values."""
        for band in scene.bands:
            ds: xr.Dataset = scene[band]
            ds["rho_toa"] = ds["rho_s"] + 0.05
        return scene
