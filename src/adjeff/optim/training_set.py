"""Training data, from reference scenes to per-combo tensors.

Classes
-------
    TrainingImages
        Reference scenes and their loss weights.
    TrainingSet
        Tensors of every scene at one atmospheric combo.
    TrainingSample
        One scene of a TrainingSet.

Functions
---------
    iterate_broadcasted_dims
        Every combo of the non-spatial coordinates of the scenes.
    training_set
        TrainingSet of one band at one combo.
"""

from collections.abc import Generator
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable

import torch
import xarray as xr

from adjeff.core import ImageDict, SensorBand
from adjeff.exceptions import ConfigurationError


@dataclass(frozen=True)
class TrainingSample:
    """Tensors of one scene at one combo, with its loss weight."""

    inputs: dict[str, torch.Tensor]
    target: torch.Tensor
    dist: torch.Tensor
    weight: float


@dataclass(frozen=True)
class TrainingSet(Iterable["TrainingSample"]):
    """Tensors of every scene at one combo, moved to *device* on iteration.

    Parameters
    ----------
    inputs, targets, dists, weights : list
        One entry per scene: input tensors by name, target, distance to
        the centre, loss weight.
    params : dict[str, float], optional
        Combo the scenes were sliced at.
    device : str, optional
        Device the tensors are moved to, ``"cuda"`` by default.
    """

    inputs: list[dict[str, torch.Tensor]]
    targets: list[torch.Tensor]
    dists: list[torch.Tensor]
    weights: list[float]
    params: dict[str, float] = field(default_factory=dict)
    device: str = "cuda"

    def __iter__(self) -> Generator[TrainingSample, None, None]:
        """Iterate over the samples in the TrainingSet."""
        for ipt, tgt, d, w in zip(
            self.inputs,
            self.targets,
            self.dists,
            self.weights,
            strict=True,
        ):
            yield TrainingSample(
                inputs={k: v.to(device=self.device) for k, v in ipt.items()},
                target=tgt.to(device=self.device),
                dist=d.to(device=self.device),
                weight=w,
            )


@dataclass(frozen=True)
class TrainingImages:
    """Reference scenes and their loss weights.

    Parameters
    ----------
    images : list[ImageDict]
        Reference scenes.
    weights : list[float] or None, optional
        One weight per scene; equal weights for ``None``.
    """

    images: list[ImageDict]
    weights: list[float] | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if self.weights is not None and len(self.weights) != len(self.images):
            raise ConfigurationError(
                f"{len(self.weights)} weights for {len(self.images)} "
                "images: there must be one per image."
            )

    @property
    def per_image(self) -> list[float]:
        """Return one weight per image, uniform when none were given."""
        if self.weights is None:
            return [1.0] * len(self.images)
        return self.weights


def iterate_broadcasted_dims(
    train: TrainingImages,
    input_names: list[str],
    target_name: str,
    band: SensorBand,
) -> Generator[dict[str, float], None, None]:
    """Yield every combo of the non-spatial coordinates of the scenes.

    Inputs and target are broadcast together, scene by scene.

    Yields
    ------
    dict[str, float]
        ``{dim: value}`` for every non-spatial dimension.

    Raises
    ------
    ConfigurationError
        If scenes differ in dimensions or coordinates, or lack ``x``/``y``.
    """
    dims: list[str] = []
    coords: list[xr.DataArray] = []

    for idx, im in enumerate(train.images):
        arrays = [im[band][name] for name in [*input_names, target_name]]
        broadcasted, *_ = xr.broadcast(*arrays)

        current_dims = [str(d) for d in broadcasted.dims]
        dims = current_dims if not dims else dims

        if current_dims != dims:
            raise ConfigurationError(
                f"Dims of ImageDict {idx} ({current_dims}) not consistent "
                f"with the other ImageDict dims ({dims})"
            )

        current_coords = [broadcasted.coords[dim] for dim in dims]
        coords = current_coords if not coords else coords
        diff = [bool((c1 != c2).any()) for c1, c2 in zip(coords, current_coords)]
        if any(diff):
            raise ConfigurationError(
                "Mismatching coordinates between trained ImageDicts."
            )

    if "x" not in dims or "y" not in dims:
        raise ConfigurationError("Dimensions should contain (x, y)")

    extra_dims = [d for d in dims if d not in ("x", "y")]
    extra_coords = [coords[dims.index(d)] for d in extra_dims]

    for combo in product(*extra_coords):
        yield dict(zip(extra_dims, combo))


def _safe_sel(da: xr.DataArray, params: dict[str, float]) -> xr.DataArray:
    """Select only on dimensions that exist in *da*.

    Variables such as ``rho_s`` have only ``(y, x)`` dimensions and would
    raise if called with atmospheric selectors (``aot``, ``rh``, …).
    """
    existing = {k: v for k, v in params.items() if k in da.dims}
    return da.sel(existing, drop=True) if existing else da


def training_set(
    train: TrainingImages,
    input_names: list[str],
    target_name: str,
    band: SensorBand,
    device: str = "cuda",
    **params: float,
) -> TrainingSet:
    """Return the tensors of *band* at the combo *params*, for every scene.

    A variable is sliced only on the dimensions it has.
    """

    def tensor(da: xr.DataArray) -> torch.Tensor:
        out: torch.Tensor = _safe_sel(da, params).adjeff.to_tensor().to(device=device)
        return out

    targets = [_safe_sel(im[band][target_name], params) for im in train.images]

    return TrainingSet(
        inputs=[
            {name: tensor(im[band][name]) for name in input_names}
            for im in train.images
        ],
        targets=[t.adjeff.to_tensor().to(device=device) for t in targets],
        dists=[t.adjeff.dists.to(device=device) for t in targets],
        weights=train.per_image,
        params=dict(params),
        device=device,
    )
