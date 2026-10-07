"""Surface reflectance from uniform reflectance, through a learnable PSF.

Classes
-------
    Unif2Surface
        Convolves ``rho_unif`` into ``rho_env``, then solves 5S for ``rho_s``.
"""

from typing import Any, ClassVar

from .psf_conv_module import PSFConvModule


def _rho_s_from_rho_env(
    rho_unif: Any,
    sph_alb: Any,
    tdir_up: Any,
    tdif_up: Any,
    rho_env: Any,
) -> Any:
    """Return ``rho_s`` from the 5S formula, given ``rho_env``."""
    frac = (1 - rho_env * sph_alb) / (1 - rho_unif * sph_alb)
    return (rho_unif * (tdir_up + tdif_up) * frac - rho_env * tdif_up) / tdir_up


class Unif2Surface(PSFConvModule):
    """``rho_s`` from ``rho_unif``, through a learnable PSF.

    ``rho_env`` is ``rho_unif`` convolved with the PSF, then::

        rho_s = (rho_unif T_up f - rho_env tdif_up) / tdir_up
        f     = (1 - rho_env sph_alb) / (1 - rho_unif sph_alb)

    Parameters
    ----------
    See :class:`PSFConvModule`.
    """

    _required_vars: ClassVar[list[str]] = [
        "rho_unif",
        "tdir_up",
        "tdif_up",
        "sph_alb",
    ]
    _output_vars: ClassVar[list[str]] = ["rho_s"]
    _conv_input: ClassVar[str] = "rho_unif"
    _formula: ClassVar[Any] = staticmethod(_rho_s_from_rho_env)
