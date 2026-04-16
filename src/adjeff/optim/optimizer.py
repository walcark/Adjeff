"""Outer optimization loop and pipeline orchestrators."""

from __future__ import annotations

import abc

import structlog
import xarray as xr

from adjeff.core import PSFDict
from adjeff.core.bands import SensorBand
from adjeff.modules.scene_module import TrainableSceneModule

from ._combo_stage import _ComboStage, restore_all_params, save_all_params
from ._config import OptimizerConfig
from .training_set import (
    TrainingImages,
    TrainingSet,
    iterate_broadcasted_dims,
    training_set,
)

logger = structlog.get_logger(__name__)


class _Optimizer(abc.ABC):
    """Outer optimization loop over atmospheric combos.

    Handles combo building, parameter snapshots, kernel stacking, and
    logging.  Subclasses implement :meth:`_run_combo` to specify what
    happens within each combo.

    Parameters
    ----------
    train_images : TrainingImages
        Collection of training scenes.
    config : OptimizerConfig
        Configuration (used for logging and pipeline synthesis).
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
        self.best_loss: float = float("inf")
        self.nloop: int = 0

    def _reset_state(self) -> None:
        """Reset per-combo logging counters."""
        self.best_loss = float("inf")
        self.nloop = 0

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

        initial_params = save_all_params(model)

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

            restore_all_params(model, initial_params)
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
            if param_pieces[band_id]:
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

    @abc.abstractmethod
    def _run_combo(
        self,
        model: TrainableSceneModule,
        band_sets: list[tuple[SensorBand, TrainingSet]],
        combo_str: str,
    ) -> None:
        """Run one optimisation for a single atmospheric combo.

        Must set ``self.best_loss`` and ``self.nloop`` for logging.
        """

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
        datasets = []
        for combo, da in pieces:
            for dim, val in combo.items():
                da = da.expand_dims({dim: [val]})
            datasets.append(da.to_dataset(name="kernel"))
        return xr.combine_by_coords(datasets, combine_attrs="drop")["kernel"]


class SingleStageOptimizer(_Optimizer):
    """Optimizer wrapping a single :class:`_ComboStage`.

    Parameters
    ----------
    stage : _ComboStage
        The optimization stage to run for each combo.
    train_images : TrainingImages
        Collection of training scenes.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        stage: _ComboStage,
        train_images: TrainingImages,
        device: str = "cuda",
    ) -> None:
        super().__init__(train_images, stage.config, device=device)
        self.stage = stage

    def _run_combo(
        self,
        model: TrainableSceneModule,
        band_sets: list[tuple[SensorBand, TrainingSet]],
        combo_str: str,
    ) -> None:
        """Delegate to the wrapped stage."""
        self.stage._reset_state()
        self.stage._run_combo(model, band_sets, combo_str)
        self.best_loss = self.stage.best_loss
        self.nloop = self.stage.nloop


class OptimizerPipeline(_Optimizer):
    """Chains multiple :class:`_ComboStage` instances within each combo.

    Each stage is reset and run in order.  The pipeline's ``best_loss`` is
    the minimum across all stages; ``nloop`` is the total step count.

    Parameters
    ----------
    stages : list[_ComboStage]
        Stages to run sequentially per combo.
    train_images : TrainingImages
        Collection of training scenes.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        stages: list[_ComboStage],
        train_images: TrainingImages,
        device: str = "cuda",
    ) -> None:
        config = OptimizerConfig(
            min_steps=sum(s.config.min_steps for s in stages),
            max_steps=sum(s.config.max_steps for s in stages),
            loss_relative_tolerance=stages[-1].config.loss_relative_tolerance,
            loss=stages[-1].config.loss,
        )
        super().__init__(train_images, config, device=device)
        self.stages = stages

    def _run_combo(
        self,
        model: TrainableSceneModule,
        band_sets: list[tuple[SensorBand, TrainingSet]],
        combo_str: str,
    ) -> None:
        """Run all stages sequentially."""
        total_nloop = 0
        total_best_loss = float("inf")
        for stage in self.stages:
            stage._reset_state()
            stage._run_combo(model, band_sets, combo_str)
            total_nloop += stage.nloop
            total_best_loss = min(total_best_loss, stage.best_loss)
        self.best_loss = total_best_loss
        self.nloop = total_nloop
