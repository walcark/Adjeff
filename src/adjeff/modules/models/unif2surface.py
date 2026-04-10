"""Unif2Surface: estimate rho_s from rho_unif via a learnable PSF."""

from typing import Any, ClassVar

from adjeff.core import PSFDict

from .psf_conv_module import PSFConvModule


def _rho_s_from_rho_env(
    rho_unif: Any,
    sph_alb: Any,
    tdir_up: Any,
    tdif_up: Any,
    rho_env: Any,
) -> Any:
    """5S formula given a pre-computed rho_env (result of PSF convolution)."""
    frac = (1 - rho_env * sph_alb) / (1 - rho_unif * sph_alb)
    return (
        rho_unif * (tdir_up + tdif_up) * frac - rho_env * tdif_up
    ) / tdir_up


class Unif2Surface(PSFConvModule):
    """Estimate rho_s from rho_unif via a learnable PSF.

    Parameters
    ----------
    psf_dict : PSFDict
        Trainable or frozen PSFDict.  Use :func:`~adjeff.core.init_psf_dict`
        to create a trainable instance for optimisation.
    cache : CacheStore or None, optional
        Cache backend for the xarray inference path.
    device : torch.device or str, optional
        Device for convolutions (default ``"cuda"``).
    """

    required_vars: ClassVar[list[str]] = [
        "rho_unif",
        "tdir_up",
        "tdif_up",
        "sph_alb",
    ]
    output_vars: ClassVar[list[str]] = ["rho_s"]
    _conv_input: ClassVar[str] = "rho_unif"
    _formula: ClassVar[Any] = staticmethod(_rho_s_from_rho_env)

    def to_psf_dict(self) -> PSFDict:
        """Export current kernels to a frozen :class:`~adjeff.core.PSFDict`."""
        return self._psf_dict.to_frozen()
