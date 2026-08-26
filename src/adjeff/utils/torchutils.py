"""PyTorch utilities: constrained parameters, transforms, radial helpers."""

from typing import Protocol, cast

import torch
import torch.nn as nn
from torch.distributions import transforms as _transforms

from .._logging import get_logger

logger = get_logger(__name__)


class Transform(Protocol):
    """Map an unconstrained parameter to a physical one, and back.

    An optimiser walks the unconstrained space; the model reads the
    constrained value.  Any strictly increasing bijection will do, which
    is what :class:`ConstrainedParameter` checks for.
    """

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained parameter to its constrained form."""
        ...

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Map a constrained parameter back to the unconstrained space."""
        ...


class ExpTransform:
    """Map the whole line onto the strictly positive half of it.

    Use this when the parameter must be positive and has no upper limit
    of its own.  The mapping itself is
    :class:`torch.distributions.transforms.ExpTransform`.
    """

    def __init__(self) -> None:
        self._t = _transforms.ExpTransform()

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Return ``exp(p)``, strictly positive."""
        return cast(torch.Tensor, self._t(p))

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Return ``log(theta)``."""
        return cast(torch.Tensor, self._t.inv(theta))


class SigmoidTransform:
    """Map the whole line onto the open interval ``(a, b)``.

    Composes :class:`torch.distributions.transforms.SigmoidTransform`
    with an affine rescaling onto ``(a, b)``.

    Parameters
    ----------
    a : float
        Lower bound.
    b : float
        Upper bound.
    eps : float, optional
        Margin kept away from either bound when inverting, since the
        inverse diverges there.  The default is deliberately far coarser
        than the machine epsilon torch would use: see :meth:`inverse`.
    """

    def __init__(self, a: float, b: float, eps: float = 1e-6) -> None:
        self.a = a
        self.b = b
        self.eps = eps
        self._t = _transforms.ComposeTransform(
            [_transforms.SigmoidTransform(), _transforms.AffineTransform(a, b - a)]
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Return the value in ``(a, b)`` that *p* maps to."""
        return cast(torch.Tensor, self._t(p))

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the unconstrained parameter *theta* comes from.

        *theta* is pulled *eps* inside the interval first.  Inverting at
        the bound itself is infinite, and inverting near it is worse than
        useless: torch clamps at the machine epsilon, which puts the
        lower bound of a ``(1, 5)`` interval at ``p = -87``, where the
        sigmoid derivative is ``1e-38``.  A parameter projected there is
        as dead as one left outside the interval, which is the very
        failure :meth:`ConstrainedParameter.project` exists to prevent.
        At ``eps = 1e-6`` the bound sits at ``p = -15.2``, where the
        derivative is still ``2e-7`` and float32 can work with it.
        """
        inside = torch.clamp(theta, self.a + self.eps, self.b - self.eps)
        return cast(torch.Tensor, self._t.inv(inside))


class ConstrainedParameter(nn.Module):
    """Trainable parameter constrained via Sigmoid or Log transforms.

    Guarantees that the parameter stays within specified bounds in
    the optimization space.

    Parameters
    ----------
    init_value : torch.Tensor
        Initial value in constrained space.
    transform : Transform
        Any strictly increasing bijection whose inverse is finite at
        *min_val* and *max_val*.
    min_val : float
        Minimum allowed value.
    max_val : float
        Maximum allowed value.
    requires_grad : bool
        Whether the parameter is trainable.
    name : str, optional
        Parameter name for logging/debug.
    """

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

        # The bounds of the unconstrained space are the images of the
        # physical ones, which only means anything for a transform that
        # is strictly increasing.  Rather than admit a fixed list of
        # transforms, check the property the bounds actually need.
        self.p_min = transform.inverse(torch.tensor(min_val))
        self.p_max = transform.inverse(torch.tensor(max_val))
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
        """Bring the raw parameter back onto its domain.

        Call this after every optimiser step.  Clamping inside
        :meth:`forward` bounds the *value* but not the raw parameter, and
        a step large enough to send the raw parameter far past a bound
        leaves it there for good: the derivative of ``clamp`` is zero
        outside the interval, so the gradient dies and no later step can
        bring it back.  The constrained value looks perfectly plausible
        the whole time, which is what makes it worth guarding against.

        Projecting keeps the raw parameter *on* the boundary instead of
        behind it, where the transform is still differentiable and a
        descent direction still exists.
        """
        self.p.clamp_(self.p_min, self.p_max)

    @property
    def value(self) -> torch.Tensor:
        """Return the current constrained value (theta), grad and all."""
        return self.forward()

    @property
    def scalar(self) -> float:
        """Return the current constrained value as a plain number.

        Reading :attr:`value` into a `float` detaches implicitly and
        torch warns about it, rightly: it is the point where a value
        leaves the graph, and doing it by accident inside a training loop
        is a real mistake.  Anything that only wants the number says so
        here.
        """
        return float(self.value.detach())

    @torch.no_grad()
    def set(self, theta: torch.Tensor) -> None:
        """Set parameter from constrained value."""
        p = self.transform.inverse(theta)
        p = torch.clamp(p, self.p_min, self.p_max)
        self.p.copy_(p)


def radial_weights(dists: torch.Tensor) -> torch.Tensor:
    """Compute inverse-perimeter radial weights.

    Each pixel is assigned weight ``1 / (2π · max(r_min, r))`` so that
    integrating over the image gives equal importance to every radial
    distance.  The centre pixel (r=0) receives the same weight as the
    nearest non-zero-distance pixel to avoid division by zero.

    Parameters
    ----------
    dists : torch.Tensor
        Per-pixel radial distances, any shape.

    Returns
    -------
    torch.Tensor
        Weights tensor, same shape as *dists*.
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
    """Return a boolean mask retaining pixels within a radial CDF threshold.

    Pixels are sorted by increasing distance from the centre.  The cumulative
    sum of ``|tensor|`` is computed radially; the mask keeps all pixels whose
    cumulative contribution is below *threshold* of the total energy.

    Parameters
    ----------
    tensor : torch.Tensor
        2D (or flat) field whose energy distribution drives the mask.
    rr : torch.Tensor
        Per-pixel radial distances, same shape as *tensor*.
    threshold : float
        CDF fraction to retain (e.g. ``0.99`` keeps 99 % of the energy).

    Returns
    -------
    torch.Tensor
        Boolean tensor, same shape as *tensor*.
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
