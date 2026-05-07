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
    """Return surface reflectance from the 5S formula given *rho_env*.

    *rho_env* is the PSF-convolved version of *rho_unif*, representing
    the effective environment reflectance seen by the sensor::

        frac = (1 - rho_env * sph_alb) / (1 - rho_unif * sph_alb)
        rho_s = (
            rho_unif * (tdir_up + tdif_up) * frac - rho_env * tdif_up
        ) / tdir_up
    """
    frac = (1 - rho_env * sph_alb) / (1 - rho_unif * sph_alb)
    return (
        rho_unif * (tdir_up + tdif_up) * frac - rho_env * tdif_up
    ) / tdir_up


class Unif2Surface(PSFConvModule):
    """Estimate surface reflectance from ``rho_unif`` via a learnable PSF.

    The forward pass combines two steps:

    1. **PSF convolution** — ``rho_unif`` is convolved with the PSF
       kernel to produce ``rho_env``, the effective environment
       reflectance seen by the sensor.
    2. **5S formula** — ``rho_s`` is recovered from ``rho_unif`` and
       ``rho_env``::

           frac = (1 - rho_env * sph_alb) / (1 - rho_unif * sph_alb)
           rho_s = (
               rho_unif * (tdir_up + tdif_up) * frac - rho_env * tdif_up
           ) / tdir_up

    Required variables (per band): ``rho_unif``, ``tdir_up``,
    ``tdif_up``, ``sph_alb``.

    Produced variable: ``rho_s``.

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
