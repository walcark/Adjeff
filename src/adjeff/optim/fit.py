"""Fit a model's PSF over every atmospheric combo of a training set."""

from __future__ import annotations

import uuid
from pathlib import Path

import xarray as xr

from adjeff.core.bands import SensorBand
from adjeff.core.psf_tree import psf_tree, write_band
from adjeff.modules.scene_module import TrainableSceneModule

from .._logging import get_logger, run_context, timed
from ._combo_stage import _ComboStage, restore_all_params, save_all_params
from ._config import OptimizerConfig
from .adam_optimizer import AdamConfig, AdamStage
from .lbfgs_optimizer import LBFGSConfig, LBFGSStage
from .loss import Loss
from .metrics import Metric
from .training_set import (
    TrainingImages,
    iterate_broadcasted_dims,
    training_set,
)

logger = get_logger(__name__)

__all__ = ["fit", "default_stages"]


def default_stages(loss: Loss | None = None) -> list[OptimizerConfig]:
    """Return the default Adam warm-up followed by L-BFGS refinement.

    Parameters
    ----------
    loss : Loss or None, optional
        Loss both stages minimise.  Defaults to ``Loss(Metric.RMSE_RAD)``.

    Returns
    -------
    list of OptimizerConfig
        Stage configurations, in the order they are run.
    """
    loss = loss if loss is not None else Loss(Metric.RMSE_RAD)
    return [
        AdamConfig(
            min_steps=5,
            max_steps=20,
            loss_relative_tolerance=1e-4,
            loss=loss,
            lr=1e-2,
        ),
        LBFGSConfig(
            min_steps=5,
            max_steps=30,
            loss_relative_tolerance=1e-6,
            loss=loss,
        ),
    ]


def fit(
    model: TrainableSceneModule,
    train_images: TrainingImages,
    *,
    loss: Loss | None = None,
    stages: list[OptimizerConfig] | None = None,
    device: str = "cuda",
    store: str | Path | None = None,
) -> xr.DataTree:
    """Optimise every PSF of *model* and return a frozen PSF tree.

    One independent optimisation runs per ``(atmospheric combo, band)``
    pair found in *train_images*.  Each band is fitted on its own so that
    wavelength-dependent effects are captured, and the model is reset to
    its initial parameters before every run.

    Parameters
    ----------
    model : TrainableSceneModule
        Model holding live PSF modules, e.g. built by
        :func:`~adjeff.api.make_model`.
    train_images : TrainingImages
        Reference scenes the PSF is fitted against.
    loss : Loss or None, optional
        Loss for the default stages.  Ignored when *stages* is given,
        since each stage config carries its own loss.
    stages : list of OptimizerConfig or None, optional
        Stage configurations run in order within each combo.  Defaults to
        :func:`default_stages`, i.e. an Adam warm-up then L-BFGS.
    device : str, optional
        Torch device for the training loop.
    store : str or Path or None, optional
        When given, each band's stacked kernel is written to zarr as soon
        as it is complete and freed from RAM; the returned tree is then
        backed by that store.  ``None`` keeps the kernels in memory.

    Returns
    -------
    xr.DataTree
        One group per band, holding the kernel stacked over every combo
        plus the fitted parameters.
    """
    runs = _stages_from_configs(stages if stages is not None else default_stages(loss))
    zpath = Path(store) if store is not None else None
    inputs = model.required_vars
    target = model.output_vars[0]
    bands: list[SensorBand] = [psf.band for psf in model.psf_modules.values()]

    combos = list(iterate_broadcasted_dims(train_images, inputs, target, bands[0]))
    initial = save_all_params(model)

    # Kernels are captured as each combo finishes rather than replayed
    # from a parameter snapshot afterwards: the snapshot only existed to
    # rebuild what the model already held at that moment.
    kernels: dict[SensorBand, list[tuple[dict[str, float], xr.DataArray]]] = {
        band: [] for band in bands
    }
    params: dict[SensorBand, dict[str, list[tuple[dict[str, float], float]]]] = {
        band: {} for band in bands
    }

    total = len(combos) * len(bands)
    done = 0
    with run_context(run_id=uuid.uuid4().hex[:8]):
        logger.info(
            "fit.start",
            combos=len(combos),
            bands=len(bands),
            optimisations=total,
            stages=" ".join(type(stage).__name__ for stage in runs),
            device=device,
        )
        for combo in combos:
            for band in bands:
                done += 1
                label = "  ".join(f"{k}={v:.3g}" for k, v in combo.items())
                with (
                    run_context(band=band.id, combo=f"{done}/{total}"),
                    timed(logger, "fit.combo", params=label or "-") as outcome,
                ):
                    data = training_set(
                        train_images, inputs, target, band, device=device, **combo
                    )
                    restore_all_params(model, initial)

                    best = float("inf")
                    steps = 0
                    for stage in runs:
                        stage._reset_state()
                        stage._run_combo(model, band, data, label)
                        best = min(best, stage.best_loss)
                        steps += stage.nloop
                    del data

                    psf = model.psf_modules[band.id]
                    kernels[band].append((combo, psf.to_dataarray()))
                    for name, value in psf.param_dict().items():
                        params[band].setdefault(name, []).append((combo, value))

                    outcome["best_loss"] = round(best, 6)
                    outcome["steps"] = steps

    stacked: dict[SensorBand, xr.DataArray] = {}
    stacked_params: dict[SensorBand, dict[str, xr.DataArray]] = {}
    for band in bands:
        kernel = _stack(kernels[band], name="kernel")
        band_params = {
            name: _stack_scalars(values) for name, values in params[band].items()
        }
        if zpath is not None:
            write_band(zpath / band.id, xr.Dataset({"kernel": kernel, **band_params}))
            del kernel
        else:
            stacked[band] = kernel
            if band_params:
                stacked_params[band] = band_params

    logger.info(
        "optimisation complete",
        n_combos=len(combos),
        n_bands=len(bands),
        bands=[str(b) for b in bands],
    )
    if zpath is not None:
        tree: xr.DataTree = xr.open_datatree(zpath, engine="zarr")
        return tree
    return psf_tree(stacked, params=stacked_params or None)


def _stages_from_configs(
    configs: list[OptimizerConfig],
) -> list[_ComboStage]:
    """Instantiate the stage matching each configuration type."""
    out: list[_ComboStage] = []
    for config in configs:
        if isinstance(config, AdamConfig):
            out.append(AdamStage(config))
        elif isinstance(config, LBFGSConfig):
            out.append(LBFGSStage(config))
        else:
            raise TypeError(
                f"Unknown stage configuration {type(config).__name__!r}; "
                "expected AdamConfig or LBFGSConfig."
            )
    return out


def _stack(
    pieces: list[tuple[dict[str, float], xr.DataArray]], *, name: str
) -> xr.DataArray:
    """Stack per-combo arrays into one, keyed by the combo coordinates."""
    datasets = []
    for combo, array in pieces:
        for dim, value in combo.items():
            array = array.expand_dims({dim: [value]})
        datasets.append(array.to_dataset(name=name))
    combined: xr.DataArray = xr.combine_by_coords(datasets, combine_attrs="drop")[name]
    return combined


def _stack_scalars(
    values: list[tuple[dict[str, float], float]],
) -> xr.DataArray:
    """Stack per-combo scalar parameter values into one array."""
    return _stack([(combo, xr.DataArray(value)) for combo, value in values], name="p")
