"""Bounded trainable parameters, and radial weights and masks.

Classes
-------
    Transform
        Protocol of an increasing bijection from the real line.
    ExpTransform
        Real line to ``(0, inf)``.
    SigmoidTransform
        Real line to ``(a, b)``.
    ConstrainedParameter
        Trainable parameter kept within bounds.

Functions
---------
    radial_weights
        Weights giving every radius the same total weight.
    radial_mask
        Pixels within a fraction of a field's radial energy.
"""

from typing import Protocol, cast

import torch
import torch.nn as nn
from torch.distributions import transforms as _transforms

from .._logging import get_logger

logger = get_logger(__name__)


class Transform(Protocol):
    """Strictly increasing bijection from the real line to a parameter's domain."""

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained parameter to its constrained form."""
        ...

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Map a constrained parameter back to the unconstrained space."""
        ...


class ExpTransform:
    """``exp``: the real line to ``(0, inf)``."""

    def __init__(self) -> None:
        self._t = _transforms.ExpTransform()

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Return ``exp(p)``, strictly positive."""
        return cast(torch.Tensor, self._t(p))

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Return ``log(theta)``."""
        return cast(torch.Tensor, self._t.inv(theta))


class SigmoidTransform:
    """Scaled sigmoid: the real line to ``(a, b)``.

    Parameters
    ----------
    a, b : float
        Bounds.
    eps : float, optional
        Margin from the bounds when inverting, 1e-6 by default.  Coarser
        than machine precision on purpose, see :meth:`inverse`.
    """

    def __init__(self, a: float, b: float, eps: float = 1e-6) -> None:
        self.a = a
        self.b = b
        self.eps = eps
        self._t = _transforms.ComposeTransform(
            [
                _transforms.SigmoidTransform(),
                _transforms.AffineTransform(a, b - a),
            ]
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Return the value in ``(a, b)`` that *p* maps to."""
        return cast(torch.Tensor, self._t(p))

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the unconstrained parameter *theta* comes from."""
        inside = torch.clamp(theta, self.a + self.eps, self.b - self.eps)
        return cast(torch.Tensor, self._t.inv(inside))


class ConstrainedParameter(nn.Module):
    """Trainable parameter kept in ``[min_val, max_val]`` through a transform.

    The optimiser moves an unconstrained value ``p``; :attr:`value` is
    ``transform(p)``.

    Parameters
    ----------
    init_value : torch.Tensor
        Initial value, in the domain; clamped into bounds with a warning.
    transform : Transform
        Increasing bijection, finite at both bounds.
    min_val, max_val : float
        Bounds.
    requires_grad : bool, optional
        Whether the parameter is trained, True by default.
    name : str or None, optional
        Name used in logs and errors.
    """

    p_min: torch.Tensor
    p_max: torch.Tensor

    def __init__(
        self,
        init_value: torch.Tensor,
        transform: "Transform",
        min_val: float,
        max_val: float,
        requires_grad: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.name = name or "param"
        self.min_val = min_val
        self.max_val = max_val

        # Bounds in unconstrained space; buffers, so they follow .to(device).
        self.register_buffer(
            "p_min", transform.inverse(torch.tensor(min_val)), persistent=False
        )
        self.register_buffer(
            "p_max", transform.inverse(torch.tensor(max_val)), persistent=False
        )
        if not (
            torch.isfinite(self.p_min)
            and torch.isfinite(self.p_max)
            and self.p_min < self.p_max
        ):
            raise ValueError(
                f"{self.name}: transform {type(transform).__name__} maps "
                f"[{min_val}, {max_val}] to [{float(self.p_min)}, "
                f"{float(self.p_max)}], which is not a usable interval. "
                "A transform must be finite and strictly increasing over "
                "the parameter's bounds."
            )
        with torch.no_grad():
            raw = transform.inverse(init_value)
            p0 = torch.clamp(raw, self.p_min, self.p_max)
            if not torch.equal(p0, raw):
                logger.warning(
                    "parameter.clamped",
                    parameter=self.name,
                    requested=float(init_value),
                    used=float(transform.forward(p0)),
                    bounds=(min_val, max_val),
                )

        self.p = nn.Parameter(p0, requires_grad=requires_grad)

    def forward(self) -> torch.Tensor:
        """Return the constrained parameter within bounds."""
        p = torch.clamp(self.p, self.p_min, self.p_max)
        return self.transform.forward(p)

    @torch.no_grad()
    def project(self) -> None:
        """Clamp ``p`` into bounds; call after every optimiser step.

        Past a bound, ``clamp`` in :meth:`forward` has a zero gradient
        and ``p`` would never come back.
        """
        self.p.clamp_(self.p_min, self.p_max)

    @property
    def value(self) -> torch.Tensor:
        """Return the current constrained value (theta), grad and all."""
        return self.forward()

    @property
    def scalar(self) -> float:
        """Constrained value, as a float detached from the graph."""
        return float(self.value.detach())

    @torch.no_grad()
    def set(self, theta: torch.Tensor) -> None:
        """Set parameter from constrained value."""
        p = self.transform.inverse(theta)
        p = torch.clamp(p, self.p_min, self.p_max)
        self.p.copy_(p)


def radial_weights(dists: torch.Tensor) -> torch.Tensor:
    """Return ``1 / (2 pi r)`` per pixel, so that every radius weighs the same.

    The centre takes the weight of the nearest pixel beyond 1e-3.
    """
    dists_non_zero = dists[dists > 0]
    if dists_non_zero.numel() == 0:
        raise ValueError("dists must contain at least one non-zero value.")
    threshold = 1e-3
    min_dist = dists_non_zero[dists_non_zero > threshold].min()
    perimeter = 2 * torch.pi * torch.maximum(min_dist, dists)
    return 1.0 / perimeter  # type: ignore[no-any-return, unused-ignore]


def radial_mask(
    tensor: torch.Tensor, rr: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Return the pixels within *threshold* of the radial energy of ``|tensor|``.

    *rr* gives the distance of each pixel; the mask has the shape of
    *tensor*.
    """
    with torch.no_grad():
        values: torch.Tensor = tensor.flatten()
        rr_cp: torch.Tensor = rr.to(values.device).flatten()

        # Order values with increasing rr and compute cumsum
        idx = torch.argsort(rr_cp)
        sorted_values: torch.Tensor = values[idx].abs()
        energy: torch.Tensor = sorted_values.cumsum(dim=0)

        # Mask values for CDF > threshold
        cutoff: torch.Tensor = energy[-1] * threshold
        mask: torch.Tensor = energy <= cutoff

        # Invert sorting to invert the mask
        idx_inv: torch.Tensor = torch.empty_like(idx)
        idx_inv[idx] = torch.arange(idx.numel(), device=idx.device)
        mask = mask[idx_inv]

        # Return the reshaped mask
        return mask.to(tensor.device).reshape(*tensor.shape)
