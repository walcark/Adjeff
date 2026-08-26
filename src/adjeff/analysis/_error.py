"""Compare a retrieved field with the truth, on DataArrays.

``Metric`` works on tensors, which is what a training loop needs and
what an analysis does not: every caller outside the loop repeated the
same four ``.adjeff.to_tensor().to(device)`` conversions, and paid for a
shape mistake with a silent broadcast rather than an error.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import xarray as xr

from adjeff.optim.metrics import Metric

__all__ = ["bias", "mae", "rmse"]

_RADIAL = {
    "mae": Metric.MAE_RAD,
    "mse": Metric.MSE_RAD,
    "rmse": Metric.RMSE_RAD,
}
_PLAIN = {"mae": Metric.MAE, "mse": Metric.MSE, "rmse": Metric.RMSE}


def _score(
    kind: Literal["mae", "mse", "rmse"],
    predicted: xr.DataArray,
    truth: xr.DataArray,
    mask: xr.DataArray | float | None,
    radial: bool,
    device: str,
) -> float:
    """Return one metric of *predicted* against *truth*."""
    if predicted.shape != truth.shape:
        raise ValueError(
            f"shape mismatch: predicted {predicted.shape}, truth "
            f"{truth.shape}.  Select and tidy them the same way first."
        )
    dists = truth.adjeff.dists.to(device)

    mask_tensor: torch.Tensor | None
    if mask is None:
        mask_tensor = None
    elif isinstance(mask, xr.DataArray):
        if mask.shape != truth.shape:
            raise ValueError(
                f"shape mismatch: mask {mask.shape}, truth {truth.shape}."
            )
        mask_tensor = mask.adjeff.to_tensor().to(device)
    else:
        mask_tensor = dists <= float(mask)

    metric = (_RADIAL if radial else _PLAIN)[kind]
    return float(
        metric(
            predicted.adjeff.to_tensor().to(device),
            truth.adjeff.to_tensor().to(device),
            dists,
            mask_tensor,
        )
    )


def rmse(
    predicted: xr.DataArray,
    truth: xr.DataArray,
    *,
    mask: xr.DataArray | float | None = None,
    radial: bool = False,
    device: str = "cpu",
) -> float:
    """Return the root mean square error of *predicted* against *truth*.

    Parameters
    ----------
    predicted, truth : xr.DataArray
        Two fields of the same shape.  A mismatch raises rather than
        broadcasting, since a leftover singleton dimension would yield a
        number that means nothing.
    mask : xr.DataArray, float or None, optional
        Which pixels take part.  A field keeps the pixels within its 99%
        radial energy, a number keeps a disc of that radius, ``None``
        keeps everything.
    radial : bool, optional
        Weight each pixel by the inverse of its circular perimeter, so
        that every radius carries the same total weight.  Appropriate for
        a radially symmetric scene, where each radius is one independent
        sample; misleading otherwise.
    device : str, optional
        Torch device the comparison runs on.

    Returns
    -------
    float
        The error, on residuals expressed in units of the reference
        amplitude.
    """
    return _score("rmse", predicted, truth, mask, radial, device)


def mae(
    predicted: xr.DataArray,
    truth: xr.DataArray,
    *,
    mask: xr.DataArray | float | None = None,
    radial: bool = False,
    device: str = "cpu",
) -> float:
    """Return the mean absolute error of *predicted* against *truth*.

    See :func:`rmse` for the parameters.
    """
    return _score("mae", predicted, truth, mask, radial, device)


def bias(
    predicted: xr.DataArray,
    truth: xr.DataArray,
    *,
    mask: xr.DataArray | float | None = None,
) -> float:
    """Return the mean signed error of *predicted* against *truth*.

    Unlike :func:`rmse` and :func:`mae`, this keeps the sign, which is
    what tells a systematic offset from scattered noise.

    Parameters
    ----------
    predicted, truth : xr.DataArray
        Two fields of the same shape.
    mask : xr.DataArray, float or None, optional
        A number keeps a disc of that radius; a field keeps the pixels
        where it is non-zero; ``None`` keeps everything.

    Returns
    -------
    float
        The mean of ``predicted - truth`` over the retained pixels.
    """
    if predicted.shape != truth.shape:
        raise ValueError(
            f"shape mismatch: predicted {predicted.shape}, truth "
            f"{truth.shape}.  Select and tidy them the same way first."
        )
    residual = (predicted - truth).values
    if mask is None:
        return float(np.mean(residual))
    if isinstance(mask, xr.DataArray):
        keep = np.asarray(mask.values) != 0
    else:
        keep = truth.adjeff.dists.numpy() <= float(mask)
    if not keep.any():
        return float("nan")
    return float(np.mean(residual[keep]))
