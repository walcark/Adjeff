"""Unif2Surface: estimate rho_s from rho_unif via a learnable PSF."""

from typing import Any, ClassVar

import torch

from adjeff.core import PSFDict

from .psf_conv_module import PSFConvModule

# ---------------------------------------------------------------------------
# Shared algebraic formula — duck-typed, works on tensors and xarray alike
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Public tensor path — autograd-compatible, used by the training loop
# ---------------------------------------------------------------------------


def unif2s_tensors(
    rho_unif: torch.Tensor,
    sph_alb: torch.Tensor,
    tdir_up: torch.Tensor,
    tdif_up: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Estimate rho_s from rho_unif using the 5S formula (tensor path).

    All inputs must be 2-D ``(H, W)`` tensors on the same device.
    Intended for the training loop via :class:`~adjeff.optim.Loss` — the
    autograd graph is preserved when *kernel* has ``requires_grad=True``.

    For multi-dimensional inference use :class:`Unif2Surface` instead.
    """
    from adjeff.utils import fft_convolve_2D_torch

    rho_env = fft_convolve_2D_torch(
        rho_unif, kernel, padding="reflect", conv_type="same"
    )
    return _rho_s_from_rho_env(  # type: ignore[no-any-return]
        rho_unif, sph_alb, tdir_up, tdif_up, rho_env
    )


# ---------------------------------------------------------------------------
# Unif2Surface
# ---------------------------------------------------------------------------


class Unif2Surface(PSFConvModule):
    """Estimate rho_s from rho_unif via a learnable PSF.

    Parameters
    ----------
    psfs : list[PSFModule]
        One PSFModule (``nn.Module`` subclass) per band.
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
        return PSFDict(
            [psf for psf in self._psfs.values()]  # type: ignore[misc]
        )
