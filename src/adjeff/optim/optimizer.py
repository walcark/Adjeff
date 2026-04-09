"""Base optimizer and L-BFGS implementation for PSF training."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import cast

import structlog
import torch
import torch.nn as nn
import xarray as xr

from adjeff.core import PSFDict
from adjeff.core.bands import SensorBand
from adjeff.modules.scene_module import TrainableSceneModule

from .loss import Loss
from .training_set import (
    TrainingImages,
    TrainingSet,
    iterate_broadcasted_dims,
    training_set,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class OptimizerConfig:
    """Shared configuration for all PSF optimizers.

    Parameters
    ----------
    min_steps : int
        Minimum steps before early stopping is allowed.
    max_steps : int
        Hard upper bound on the number of steps.
    loss_relative_tolerance : float
        Stop when ``|Δloss / previous_loss| ≤ loss_relative_tolerance``.
    loss : Loss
        Loss function instance.
    """

    min_steps: int
    max_steps: int
    loss_relative_tolerance: float
    loss: Loss


# ---------------------------------------------------------------------------
# Base optimizer
# ---------------------------------------------------------------------------


class _Optimizer(abc.ABC):
    """Base class for PSF optimizers.

    Subclasses implement :meth:`_run_combo`.  This class handles all
    generic logic: training-set construction, parameter snapshots,
    the outer combo loop, early-stopping helpers, and PSFDict assembly.

    Parameters
    ----------
    train_images : TrainingImages
        Collection of training scenes.
    config : OptimizerConfig
        Shared optimizer settings.
    device : str
        PyTorch device for tensor operations.
    """

    def __init__(
        self,
        train_images: TrainingImages,
        config: OptimizerConfig,
        device: str = "cuda",
    ) -> None:
        self.train_images = train_images
        self.config = config
        self.device = device

        self.loss_history: list[float] = []
        self.params_history: list[dict[str, dict[str, torch.Tensor]]] = []
        self.nloop: int = 0
        self.previous_loss: float = float("inf")
        self.best_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, model: TrainableSceneModule) -> PSFDict:
        """Optimise all PSFs in *model* and return a :class:`PSFDict`.

        One independent call to :meth:`_run_combo` is made per atmospheric
        combo found in *train_images*.  The model is reset to its initial
        parameter state before each combo.

        Parameters
        ----------
        model : TrainableSceneModule
            Model whose PSFs are to be optimised.

        Returns
        -------
        PSFDict
            Kernels stacked over all optimised atmospheric combos.
        """
        logger.info("Building combo training sets.")
        combo_sets = self._build_combo_sets(model)
        n_combos = len(combo_sets)
        logger.info("Combo training sets ready.", n_combos=n_combos)

        initial_params = self._save_all_params(model)

        kernel_pieces: dict[
            str, list[tuple[dict[str, float], xr.DataArray]]
        ] = {band_id: [] for band_id in model.psf_modules}
        param_pieces: dict[
            str, dict[str, list[tuple[dict[str, float], float]]]
        ] = {band_id: {} for band_id in model.psf_modules}

        for combo_idx, (combo, band_sets) in enumerate(combo_sets):
            combo_str = "  ".join(f"{k}={v:.3g}" for k, v in combo.items())
            logger.info(
                "Starting combo optimisation.",
                combo=combo_str,
                progress=f"{combo_idx + 1}/{n_combos}",
            )

            self._restore_all_params(model, initial_params)
            self._reset_state()

            self._run_combo(model, band_sets, combo_str)

            for band_id, psf in model.psf_modules.items():
                da = psf.to_dataarray()
                kernel_pieces[band_id].append((combo, da))
                for pname, pval in psf.param_dict().items():
                    param_pieces[band_id].setdefault(pname, []).append(
                        (combo, pval)
                    )

            logger.info(
                "Combo optimisation finished.",
                combo=combo_str,
                best_loss=f"{self.best_loss:.6g}",
                steps=self.nloop,
            )

        stacked: dict[SensorBand, xr.DataArray] = {}
        stacked_params: dict[SensorBand, dict[str, xr.DataArray]] = {}
        for band_id, pieces in kernel_pieces.items():
            band: SensorBand = model.psf_modules[band_id].band
            stacked[band] = self._stack_kernels(pieces)
            if n_combos > 1 and param_pieces[band_id]:
                stacked_params[band] = {
                    pname: self._stack_param(combo_vals)
                    for pname, combo_vals in param_pieces[band_id].items()
                }

        logger.info(
            "All combos optimised.",
            n_combos=n_combos,
            bands=[str(b) for b in stacked],
        )
        return PSFDict.from_kernels(
            stacked,
            params=stacked_params if stacked_params else None,
        )

    # ------------------------------------------------------------------
    # Abstract — implemented by each concrete optimizer
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _run_combo(
        self,
        model: TrainableSceneModule,
        band_sets: list[tuple[SensorBand, TrainingSet]],
        combo_str: str,
    ) -> None:
        """Run one optimisation for a single atmospheric combo.

        Called after the model has been reset to initial parameters and
        :meth:`_reset_state` has been called.  Should update the model
        in-place and set ``self.best_loss`` and ``self.nloop`` via
        :meth:`record` and :meth:`improved_loss_or_under_min_steps`.
        """

    # ------------------------------------------------------------------
    # Loss aggregation
    # ------------------------------------------------------------------

    def _total_loss(
        self,
        model: TrainableSceneModule,
        band_sets: list[tuple[SensorBand, TrainingSet]],
    ) -> torch.Tensor:
        """Sum of losses over all bands for a single atmospheric combo."""
        losses = []
        for band, ts in band_sets:
            _band = band

            def _fwd(
                inputs: dict[str, torch.Tensor],
                _b: SensorBand = _band,
            ) -> torch.Tensor:
                return model.forward_band(_b, **inputs)

            losses.append(self.config.loss(_fwd, ts))
        return torch.stack(losses).sum()

    # ------------------------------------------------------------------
    # Parameter save / restore
    # ------------------------------------------------------------------

    def _save_all_params(
        self, model: TrainableSceneModule
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Return a snapshot of all PSF unconstrained parameters."""
        return {
            band_id: {
                name: p.data.clone()
                for name, p in cast(nn.Module, psf).named_parameters()
            }
            for band_id, psf in model.psf_modules.items()
        }

    def _restore_all_params(
        self,
        model: TrainableSceneModule,
        saved: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Restore all PSF parameters from a snapshot."""
        with torch.no_grad():
            for band_id, psf in model.psf_modules.items():
                for name, p in cast(nn.Module, psf).named_parameters():
                    p.copy_(saved[band_id][name])

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset per-combo counters before each run."""
        self.loss_history = []
        self.params_history = []
        self.nloop = 0
        self.previous_loss = float("inf")
        self.best_loss = float("inf")

    def record(
        self,
        loss: float,
        params: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Append current loss and parameter snapshot to history."""
        self.loss_history.append(loss)
        self.params_history.append(params)

    def improved_loss_or_under_min_steps(self, loss: float) -> bool:
        """Return True if training should continue."""
        if self.nloop < self.config.min_steps:
            return True
        rel = (self.previous_loss - loss) / max(abs(self.previous_loss), 1e-9)
        return abs(rel) > self.config.loss_relative_tolerance

    # ------------------------------------------------------------------
    # Training-set construction
    # ------------------------------------------------------------------

    def _build_combo_sets(
        self, model: TrainableSceneModule
    ) -> list[tuple[dict[str, float], list[tuple[SensorBand, TrainingSet]]]]:
        """Pre-build one (band, TrainingSet) group per atmospheric combo."""
        input_names = model.required_vars
        target_name = model.output_vars[0]
        bands: list[SensorBand] = [
            psf.band for psf in model.psf_modules.values()
        ]
        combos = list(
            iterate_broadcasted_dims(
                self.train_images, input_names, target_name, bands[0]
            )
        )
        return [
            (
                combo,
                [
                    (
                        band,
                        training_set(
                            self.train_images,
                            input_names,
                            target_name,
                            band,
                            device=self.device,
                            **combo,
                        ),
                    )
                    for band in bands
                ],
            )
            for combo in combos
        ]

    # ------------------------------------------------------------------
    # PSFDict assembly helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_param(
        combo_vals: list[tuple[dict[str, float], float]],
    ) -> xr.DataArray:
        """Stack per-combo scalar param values into a multi-dim DataArray."""
        pieces_ds = []
        for combo, val in combo_vals:
            da: xr.DataArray = xr.DataArray(val)
            for dim, dval in combo.items():
                da = da.expand_dims({dim: [dval]})
            pieces_ds.append(da.to_dataset(name="p"))
        return xr.combine_by_coords(pieces_ds)["p"]

    @staticmethod
    def _stack_kernels(
        pieces: list[tuple[dict[str, float], xr.DataArray]],
    ) -> xr.DataArray:
        """Stack per-combo 2D kernels into a multi-dimensional DataArray."""
        if len(pieces) == 1:
            return pieces[0][1]
        datasets = []
        for combo, da in pieces:
            for dim, val in combo.items():
                da = da.expand_dims({dim: [val]})
            datasets.append(da.to_dataset(name="kernel"))
        return xr.combine_by_coords(datasets, combine_attrs="drop")["kernel"]
