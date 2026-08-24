"""Outer optimization loop and pipeline orchestrators."""

from __future__ import annotations

import abc
from pathlib import Path

import structlog
import torch
import xarray as xr

from adjeff.core.bands import SensorBand
from adjeff.core.psf_tree import psf_tree, write_band
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

    def run(
        self,
        model: TrainableSceneModule,
        zarr_path: str | Path | None = None,
    ) -> xr.DataTree:
        """Optimise all PSFs in *model* and return a frozen PSF tree.

        One independent call to :meth:`_run_combo` is made per
        (atmospheric combo × band) pair found in *train_images*.  Each
        band's PSF is optimised independently so that wavelength-dependent
        effects are captured correctly.  The model is reset to its initial
        state before each (combo, band) run.

        Parameters
        ----------
        model : TrainableSceneModule
            Model whose PSFs are to be optimised.
        zarr_path : str or Path or None
            When provided, each band's stacked kernel is written to zarr
            as it is reconstructed and freed from RAM.  The returned
            tree is backed by zarr on disk (lazy, no kernel data in
            RAM).  When ``None`` (default), kernels are kept in memory.

        Returns
        -------
        xr.DataTree
            One group per band, holding the kernel stacked over every
            optimised atmospheric combo plus the fitted parameters.
        """
        zpath: Path | None = Path(zarr_path) if zarr_path is not None else None

        input_names = model.required_vars
        target_name = model.output_vars[0]
        bands: list[SensorBand] = [
            psf.band for psf in model.psf_modules.values()
        ]

        atmo_combos = self._build_combos(model)
        n_atmo = len(atmo_combos)
        n_combos = n_atmo * len(bands)

        initial_params = save_all_params(model)

        # Snapshots keyed by (band_id, atmo_idx) for kernel reconstruction.
        param_snapshots: dict[
            tuple[str, int], dict[str, dict[str, torch.Tensor]]
        ] = {}
        param_pieces: dict[
            str, dict[str, list[tuple[dict[str, float], float]]]
        ] = {band_id: {} for band_id in model.psf_modules}

        combo_counter = 0
        for atmo_idx, combo in enumerate(atmo_combos):
            for band in bands:
                combo_counter += 1
                combo_str = "  ".join(f"{k}={v:.3g}" for k, v in combo.items())
                logger.info(
                    f"combo {combo_counter}/{n_combos}",
                    params=combo_str or "—",
                    band=str(band),
                )

                band_sets: list[tuple[SensorBand, TrainingSet]] = [
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
                ]

                restore_all_params(model, initial_params)
                self._reset_state()
                self._run_combo(model, band_sets, combo_str)
                del band_sets

                param_snapshots[(band.id, atmo_idx)] = save_all_params(model)
                for pname, pval in (
                    model.psf_modules[band.id].param_dict().items()
                ):
                    param_pieces[band.id].setdefault(pname, []).append(
                        (combo, pval)
                    )

                logger.info(
                    f"combo {combo_counter}/{n_combos} done",
                    best_loss=f"{self.best_loss:.4g}",
                    steps=self.nloop,
                    band=str(band),
                )

        # Reconstruct kernels one band at a time so peak memory is
        # n_combos * kernel_size rather than n_bands * n_combos * kernel_size.
        # When zarr_path is set, each band is flushed to disk immediately and
        # freed from RAM — the returned tree is fully lazy.
        stacked: dict[SensorBand, xr.DataArray] = {}
        stacked_params: dict[SensorBand, dict[str, xr.DataArray]] = {}
        for band_id, psf in model.psf_modules.items():
            b: SensorBand = psf.band
            kernel_pieces: list[tuple[dict[str, float], xr.DataArray]] = []
            for atmo_idx, combo in enumerate(atmo_combos):
                restore_all_params(model, param_snapshots[(band_id, atmo_idx)])
                kernel_pieces.append((combo, psf.to_dataarray()))
            stacked_kernel = self._stack_kernels(kernel_pieces)
            del kernel_pieces

            band_params: dict[str, xr.DataArray] | None = (
                {
                    pname: self._stack_param(combo_vals)
                    for pname, combo_vals in param_pieces[band_id].items()
                }
                if param_pieces[band_id]
                else None
            )

            if zpath is not None:
                ds_vars: dict[str, xr.DataArray] = {"kernel": stacked_kernel}
                if band_params:
                    ds_vars.update(band_params)
                write_band(zpath / b.id, xr.Dataset(ds_vars))
                del stacked_kernel
            else:
                stacked[b] = stacked_kernel
                if band_params:
                    stacked_params[b] = band_params

        logger.info(
            "optimisation complete",
            n_atmo_combos=n_atmo,
            n_bands=len(bands),
            n_combos_total=n_combos,
            bands=[str(b) for b in (bands if zpath else stacked)],
        )
        if zpath is not None:
            return xr.open_datatree(zpath, engine="zarr")
        return psf_tree(
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

    def _build_combos(
        self, model: TrainableSceneModule
    ) -> list[dict[str, float]]:
        """Return the list of atmospheric combos to iterate over."""
        input_names = model.required_vars
        target_name = model.output_vars[0]
        bands: list[SensorBand] = [
            psf.band for psf in model.psf_modules.values()
        ]
        return list(
            iterate_broadcasted_dims(
                self.train_images, input_names, target_name, bands[0]
            )
        )

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
