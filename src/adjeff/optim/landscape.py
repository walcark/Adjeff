"""Loss landscape and encircled-energy utilities for PSF parameter spaces.

Both :func:`loss_landscape` and :func:`energy_radius_landscape` accept any
list of :class:`~adjeff.core._psf.PSFModule` instances — Gaussian, King,
Voigt, Moffat, or any custom subclass.  The caller is responsible for
building the parameter grid and reshaping the returned 1-D arrays.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm  # type: ignore[import-untyped]

from adjeff.core import SensorBand
from adjeff.core._psf import PSFModule
from adjeff.modules.models.unif2surface import _rho_s_from_rho_env
from adjeff.utils import fft_convolve_2D_torch

from .loss import Loss
from .training_set import (
    TrainingImages,
    iterate_broadcasted_dims,
    training_set,
)


def _make_forward_fn(
    kernel: torch.Tensor,
    device: torch.device,
) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    """Unif2Surface forward closure for a fixed kernel."""

    def _fwd(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        d = device
        rho_unif = inputs["rho_unif"].to(d)
        rho_env = fft_convolve_2D_torch(
            rho_unif,
            kernel.to(d),
            padding="reflect",
            conv_type="same",
        )
        return _rho_s_from_rho_env(  # type: ignore[no-any-return]
            rho_unif=rho_unif,
            sph_alb=inputs["sph_alb"].to(d),
            tdir_up=inputs["tdir_up"].to(d),
            tdif_up=inputs["tdif_up"].to(d),
            rho_env=rho_env,
        )

    return _fwd


def loss_landscape(
    train_images: TrainingImages,
    band: SensorBand,
    psf_modules: list[PSFModule],
    loss: Loss,
    device: str = "cpu",
) -> np.ndarray:
    """Evaluate the loss for each PSF in *psf_modules*.

    For each PSF, its kernel is applied as a Unif2Surface convolution on
    the training images and the loss is computed.  Loss values are averaged
    over all atmospheric parameter combinations found in *train_images*.

    Parameters
    ----------
    train_images : TrainingImages
        Pre-computed scenes (must contain ``rho_unif``, ``tdir_up``,
        ``tdif_up``, ``sph_alb``, and ``rho_s``).
    band : SensorBand
        Band to evaluate.
    psf_modules : list[PSFModule]
        Any :class:`~adjeff.core._psf.PSFModule` instances
        (e.g. :class:`~adjeff.core.GaussGeneralPSF`,
        :class:`~adjeff.core.KingPSF`, …).
    loss : Loss
        Loss function instance.
    device : str
        Torch device for convolutions (default ``"cpu"``).

    Returns
    -------
    np.ndarray, shape (len(psf_modules),)
        Mean loss across all atmospheric combos for each PSF.
        The caller is responsible for reshaping to a parameter grid.
    """
    input_names = ["rho_unif", "tdir_up", "tdif_up", "sph_alb"]
    target_name = "rho_s"
    dev = torch.device(device)

    combos = list(
        iterate_broadcasted_dims(train_images, input_names, target_name, band)
    )
    training_sets = [
        training_set(
            train_images,
            input_names,
            target_name,
            band,
            device=device,
            **p,
        )
        for p in combos
    ]
    n_combos = max(len(training_sets), 1)

    result = np.zeros(len(psf_modules), dtype=np.float32)

    with torch.no_grad():
        for i, psf in tqdm(enumerate(psf_modules), total=len(psf_modules)):
            kernel = psf.forward()
            fwd = _make_forward_fn(kernel, dev)
            total = sum(float(loss(fwd, ts).item()) for ts in training_sets)
            result[i] = total / n_combos

    return result


def energy_radius_landscape(
    psf_modules: list[PSFModule],
    fractions: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute encircled-energy radii for each PSF in *psf_modules*.

    For each PSF, the radial CDF of its kernel is used to find the radius
    encircling *fractions* of the total energy.

    Parameters
    ----------
    psf_modules : list[PSFModule]
        Any :class:`~adjeff.core._psf.PSFModule` instances.
    fractions : list[float] or None
        CDF fractions to evaluate.  Defaults to ``[0.10, 0.50, 0.99]``.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are ``"EE10%"``, ``"EE50%"``, ``"EE99%"``
        (or matching *fractions*).
        Values are 1-D arrays of length ``len(psf_modules)``.
        The caller is responsible for reshaping to a parameter grid.
    """
    if fractions is None:
        fractions = [0.10, 0.50, 0.99]

    keys = [f"EE{int(f * 100)}%" for f in fractions]
    ee: dict[str, list[float]] = {k: [] for k in keys}

    with torch.no_grad():
        for psf in tqdm(psf_modules, total=len(psf_modules)):
            coords = psf.grid.as_coords()
            kernel = psf.forward()
            da = xr.DataArray(
                kernel.detach().cpu().numpy(),
                dims=["y_psf", "x_psf"],
                coords=coords,
            )
            cdf = da.adjeff.radial_cdf()
            r = cdf.coords["r"].values
            cdf_vals = cdf.values

            for frac, key in zip(fractions, keys):
                idx = int(np.searchsorted(cdf_vals, frac))
                idx = min(idx, len(r) - 1)
                ee[key].append(float(r[idx]))

    return {k: np.array(v) for k, v in ee.items()}
