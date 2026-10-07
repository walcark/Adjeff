"""Loss and encircled energy over a list of PSFs, e.g. a parameter grid.

The caller builds the PSFs and reshapes the 1-D results onto its grid.

Functions
---------
    loss_landscape
        Loss of each PSF, averaged over the atmospheric combos.
    energy_radius_landscape
        Encircled-energy radii of each PSF.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from tqdm import tqdm  # type: ignore[import-untyped]

from adjeff.analysis import encircled_radii
from adjeff.core import SensorBand
from adjeff.core._psf import PSFModule
from adjeff.modules.models.unif2surface import _rho_s_from_rho_env
from adjeff.modules.scene_module import TrainableSceneModule
from adjeff.utils import fft_convolve_2D_torch

from .._logging import get_logger, timed
from .training_set import (
    TrainingImages,
    TrainingSample,
    iterate_broadcasted_dims,
    training_set,
)

logger = get_logger(__name__)

LossFn = Callable[
    [Callable[[dict[str, torch.Tensor]], torch.Tensor], list[TrainingSample]],
    torch.Tensor,
]


def _model_forward_fn(
    model: TrainableSceneModule, band: SensorBand, kernel: torch.Tensor
) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    """Return the per-band forward pass of *model*, on a given kernel."""

    def _fwd(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return model.forward_band(band, kernel=kernel, **inputs)

    return _fwd


def _make_forward_fn(
    kernel: torch.Tensor,
) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    """Return the Unif2Surface forward pass for a pre-moved kernel.

    Used when :func:`loss_landscape` is given no model, which keeps the
    call short for the common case.
    """

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
    loss: LossFn,
    model: TrainableSceneModule | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Evaluate the loss for each PSF in *psf_modules*.

    For each PSF, its kernel is applied to the training images and the
    loss is computed.  Loss values are averaged over all atmospheric
    parameter combinations found in *train_images*.

    Parameters
    ----------
    train_images : TrainingImages
        Pre-computed scenes.  They must carry whatever the model
        declares in ``required_vars`` and ``output_vars``, which for
        :class:`~adjeff.modules.models.Unif2Surface` means ``rho_unif``,
        ``tdir_up``, ``tdif_up``, ``sph_alb`` and ``rho_s``.
    band : SensorBand
        Band to evaluate.
    psf_modules : list[PSFModule]
        Any :class:`~adjeff.core._psf.PSFModule` instances
        (e.g. :class:`~adjeff.core.GeneralizedGaussianPSF`,
        :class:`~adjeff.core.KingPSF`, …).
    loss : Loss or callable
        Anything with the signature ``loss(forward_fn, samples)``.  The
        built-in :class:`Loss` satisfies it, and so does a custom
        callable, which is what makes it possible to map a surface for a
        metric this package does not ship.
    model : TrainableSceneModule or None, optional
        Model whose forward pass the kernels feed.  ``None`` keeps the
        :class:`~adjeff.modules.models.Unif2Surface` convolution, which
        is what the manuscript's figures use.  When given, the variable
        names are read off the model rather than assumed.
    device : str
        Torch device for convolutions (default ``"cpu"``).

    Returns
    -------
    np.ndarray, shape (len(psf_modules),)
        Mean loss across all atmospheric combos for each PSF.
        The caller is responsible for reshaping to a parameter grid.
    """
    if model is None:
        input_names = ["rho_unif", "tdir_up", "tdif_up", "sph_alb"]
        target_name = "rho_s"
    else:
        input_names = list(model.required_vars)
        target_name = model.output_vars[0]
    dev = torch.device(device)

    combos = list(
        iterate_broadcasted_dims(train_images, input_names, target_name, band)
    )

    # Transfer all training data to the target device exactly once, before
    # the PSF loop. TrainingSet.__iter__ does .to() on every call, which
    # would otherwise cause N_psf redundant host <-> device transfers per tensor.
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

    with (
        torch.no_grad(),
        timed(
            logger,
            "landscape.scan",
            kernels=len(psf_modules),
            combos=n_combos,
            band=band.id,
            device=device,
        ),
    ):
        for i, psf in tqdm(enumerate(psf_modules), total=len(psf_modules)):
            kernel = psf.forward().to(dev)
            if model is None:
                forward = _make_forward_fn(kernel)
            else:
                forward = _model_forward_fn(model, band, kernel)
            total = 0.0
            for samples in prefetched:
                # The loss is called exactly as `fit` calls it, which is
                # what lets any callable stand in for the built-in one.
                total += float(loss(forward, samples).item())
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
    if not psf_modules:
        return encircled_radii([], n=0, res=1.0, fractions=fractions)
    grid = psf_modules[0].grid
    with torch.no_grad():
        kernels = [psf.forward() for psf in tqdm(psf_modules)]
    return encircled_radii(kernels, n=grid.n, res=grid.res, fractions=fractions)
