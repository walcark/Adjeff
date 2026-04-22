"""Loss landscape and encircled-energy utilities for PSF parameter spaces.

Both :func:`loss_landscape` and :func:`energy_radius_landscape` accept any
list of :class:`~adjeff.core._psf.PSFModule` instances — Gaussian, King,
Voigt, Moffat, or any custom subclass.  The caller is responsible for
building the parameter grid and reshaping the returned 1-D arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from tqdm import tqdm  # type: ignore[import-untyped]

from adjeff.core import SensorBand
from adjeff.core._psf import PSFModule
from adjeff.modules.models.unif2surface import _rho_s_from_rho_env
from adjeff.utils import fft_convolve_2D_torch

from .loss import Loss
from .training_set import (
    TrainingImages,
    TrainingSample,
    iterate_broadcasted_dims,
    training_set,
)


def _make_forward_fn(
    kernel: torch.Tensor,
) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    """Unif2Surface forward closure for a pre-moved kernel."""

    def _fwd(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        rho_unif = inputs["rho_unif"]
        rho_env = fft_convolve_2D_torch(
            rho_unif,
            kernel,
            padding="reflect",
            conv_type="same",
        )
        return _rho_s_from_rho_env(  # type: ignore[no-any-return]
            rho_unif=rho_unif,
            sph_alb=inputs["sph_alb"],
            tdir_up=inputs["tdir_up"],
            tdif_up=inputs["tdif_up"],
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

    # Transfer all training data to the target device exactly once, before
    # the PSF loop.  TrainingSet.__iter__ does .to() on every call, which
    # would otherwise cause N_psf redundant host↔device transfers per tensor.
    prefetched: list[list[TrainingSample]] = []
    for p in combos:
        ts = training_set(
            train_images,
            input_names,
            target_name,
            band,
            device=device,
            **p,
        )
        prefetched.append(list(ts))

    n_combos = max(len(prefetched), 1)
    result = np.zeros(len(psf_modules), dtype=np.float32)

    with torch.no_grad():
        for i, psf in tqdm(enumerate(psf_modules), total=len(psf_modules)):
            kernel = psf.forward().to(dev)
            fwd = _make_forward_fn(kernel)
            total = 0.0
            for samples in prefetched:
                combo_losses = []
                for sample in samples:
                    pred = fwd(sample.inputs)
                    mask = (
                        sample.inputs.get("rho_unif")
                        if loss.mask_on == "rho_unif"
                        else None
                    )
                    combo_losses.append(
                        loss.metric(pred, sample.target, sample.dist, mask)
                        * sample.weight
                    )
                total += float(torch.stack(combo_losses).sum().item())
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

    if not psf_modules:
        return {k: np.array([], dtype=np.float32) for k in keys}

    # All PSFs in a landscape scan share the same grid.  Compute the radial
    # binning structure (pixel→bin mapping, bin counts, area weights) once
    # and reuse it for every kernel, avoiding O(N_psf) redundant meshgrid
    # and xarray operations.
    grid = psf_modules[0].grid
    n = grid.n
    npix = max(int((n - 1) / math.sqrt(2)) - 1, 2)

    half = (n // 2) * grid.res
    coords_1d = np.linspace(-half, half, n, dtype=np.float32)
    XX, YY = np.meshgrid(coords_1d, coords_1d)
    rr = torch.from_numpy(np.sqrt(XX**2 + YY**2).ravel())

    bins = torch.linspace(0.0, float(rr.max()), npix + 1)
    inds = (torch.bucketize(rr, bins, right=False) - 1).clamp(0, npix - 1)
    counts = torch.bincount(inds, minlength=npix).float()
    bin_mask = counts > 0

    r_centers = (0.5 * (bins[:-1] + bins[1:])).clone()
    r_centers[0] = 0.0
    dr = r_centers[1:] - r_centers[:-1]
    edges = torch.empty(npix + 1, dtype=r_centers.dtype)
    edges[1:-1] = 0.5 * (r_centers[:-1] + r_centers[1:])
    edges[0] = r_centers[0] - 0.5 * dr[0]
    edges[-1] = r_centers[-1] + 0.5 * dr[-1]
    area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

    r_np = r_centers.numpy()
    ee: dict[str, list[float]] = {k: [] for k in keys}

    with torch.no_grad():
        for psf in tqdm(psf_modules, total=len(psf_modules)):
            vv = psf.forward().detach().cpu().ravel().clamp(min=0.0)
            sum_vals = torch.bincount(inds, weights=vv, minlength=npix)
            mean_vals = torch.zeros(npix, dtype=torch.float32)
            mean_vals[bin_mask] = sum_vals[bin_mask] / counts[bin_mask]
            cdf = torch.cumsum(mean_vals * area, dim=0)
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1]
            cdf_np = cdf.numpy()

            for frac, key in zip(fractions, keys):
                idx = min(int(np.searchsorted(cdf_np, frac)), npix - 1)
                ee[key].append(float(r_np[idx]))

    return {k: np.array(v, dtype=np.float32) for k, v in ee.items()}
