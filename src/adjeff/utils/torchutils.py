"""Define useful classes and methods for PyTorch usage."""

import torch
import torch.nn as nn


class Transform:
    """Define a base class for parameter transformations.

    A Transform maps an unconstrained trainable parameter (p) to a
    constrained physical parameter (theta), and provides the inverse
    mapping for proper initialization.
    """

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained parameter to its constrained form.

        Parameters
        ----------
        p : torch.Tensor
            Unconstrained trainable parameter.

        Returns
        -------
        torch.Tensor
            Constrained parameter.
        """
        raise NotImplementedError

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Map a constrained parameter to its unconstrained form.

        Use this method to initialize trainable parameters from
        physically meaningful values.

        Parameters
        ----------
        theta : torch.Tensor
            Constrained parameter.

        Returns
        -------
        torch.Tensor
            Unconstrained parameter.
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    """Define an identity transformation (no constraint)."""

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Return the parameter unchanged."""
        return p

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the parameter unchanged."""
        return theta


class ExpTransform(Transform):
    """Define an exponential transformation to enforce positivity.

    Use this transform when the parameter must be strictly positive.
    """

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Map to a strictly positive parameter using exponential."""
        return torch.exp(p)

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Map a positive parameter back using logarithm."""
        return torch.log(theta)


class SigmoidTransform(Transform):
    """Define a sigmoid-based transformation for bounded parameters.

    Map parameters to a bounded interval [a, b] with a numerical stability
    margin ``eps``.

    Parameters
    ----------
    a : float
        Lower bound.
    b : float
        Upper bound.
    eps : float, optional
        Small value to avoid numerical issues at the boundaries.
    """

    def __init__(self, a: float, b: float, eps: float = 1e-6):
        self.a = a
        self.b = b
        self.eps = eps

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Map to the interval [a, b] using a sigmoid."""
        return self.a + (self.b - self.a) * torch.sigmoid(p)

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """Map a bounded parameter back to the unconstrained space."""
        theta = torch.clamp(theta, self.a + self.eps, self.b - self.eps)
        x = (theta - self.a) / (self.b - self.a)
        return torch.log(x / (1 - x))


class ConstrainedParameter(nn.Module):
    """Trainable parameter constrained via Sigmoid or Log transforms.

    Guarantees that the parameter stays within specified bounds in
    the optimization space.

    Parameters
    ----------
    init_value : torch.Tensor
        Initial value in constrained space.
    transform : Transform
        Must be SigmoidTransform or ExpTransform (for log-style positive).
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

        if not isinstance(transform, (SigmoidTransform, ExpTransform)):
            raise ValueError("Only supports Sigmoid or Exp Transforms.")

        self.transform = transform
        self.name = name or "param"
        self.min_val = min_val
        self.max_val = max_val

        # compute initial p in optimization space
        self.p_min = transform.inverse(torch.tensor(min_val))
        self.p_max = transform.inverse(torch.tensor(max_val))
        with torch.no_grad():
            p0 = transform.inverse(init_value)
            p0 = torch.clamp(p0, self.p_min, self.p_max)

        self.p = nn.Parameter(p0, requires_grad=requires_grad)

    def forward(self) -> torch.Tensor:
        """Return the constrained parameter within bounds."""
        p = torch.clamp(self.p, self.p_min, self.p_max)
        return self.transform.forward(p)

    @property
    def value(self) -> torch.Tensor:
        """Return the current constrained value (theta)."""
        return self.forward()

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
    min_dist = dists_non_zero.min()
    perimeter = 2 * torch.pi * torch.maximum(min_dist, dists)
    return 1.0 / perimeter  # type: ignore[no-any-return, unused-ignore]


def radial_mask(
    tensor: torch.Tensor, rr: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Compute a radial mask for all values under threshold."""
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
