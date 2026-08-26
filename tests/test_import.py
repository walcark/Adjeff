"""Importing adjeff must not require a Smart-G installation.

Smart-G raises at import time when ``SMARTG_DIR_AUXDATA`` is unset, so an
eager import anywhere in the package makes ``import adjeff`` fail for
anyone who only wants the parts that never touch the radiative transfer:
the accessor, the image generators, the PSF models, the radial analysis.

The check runs in a subprocess because adjeff is already imported in this
one, and because the variable has to be missing from the environment
rather than merely empty.
"""

import os
import subprocess
import sys

SCRIPT = """
import adjeff
from adjeff.core import GaussPSF, PSFGrid, S2Band
from adjeff.utils import fft_convolve_2D

psf = GaussPSF(PSFGrid(0.1, 21), S2Band.B03, sigma=0.3)
assert psf.to_dataarray().shape == (21, 21)
print("ok")
"""


def test_import_without_smartg():
    """`import adjeff` works with SMARTG_DIR_AUXDATA absent."""
    env = {k: v for k, v in os.environ.items() if k != "SMARTG_DIR_AUXDATA"}
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )

    assert proc.returncode == 0, (
        f"importing adjeff without SMARTG_DIR_AUXDATA failed:\n{proc.stderr[-2000:]}"
    )
    assert "ok" in proc.stdout
