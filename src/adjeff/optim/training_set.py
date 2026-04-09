"""Set of ImageDict instances used for optimisation."""

from collections.abc import Generator
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable

import torch
import xarray as xr

from adjeff.core import ImageDict, SensorBand


@dataclass(frozen=True)
class TrainingSample:
    """Store input, target and distance Tensors for a single comparison.

    Parameters
    ----------
    inputs : list[dict[str, torch.Tensor]]
        Input tensors for a training image.
    target : torch.Tensor
        Target tensor for a training image.
    dists : torch.Tensor
        Radial distance tensor for a training image.
    weight : float
        Loss weight for a training image.
    """

    inputs: dict[str, torch.Tensor]
    target: torch.Tensor
    dist: torch.Tensor
    weight: float


@dataclass(frozen=True)
class TrainingSet(Iterable["TrainingSample"]):
    """Store input, target and distance Tensors for a single training.

    Parameters
    ----------
    inputs : list[dict[str, torch.Tensor]]
        Input tensors for each training image.
    targets : list[torch.Tensor]
        Target tensors for each training image.
    dists : list[torch.Tensor]
        Radial distance tensors for each training image.
    weights : list[float]
        Loss weights for each training image.
    params : dict[str, float]
        Atmospheric and geometric parameters used to slice the ImageDicts
        when building this set (e.g. ``{"aot": 0.1, "rh": 50.0}``).
    device : str
        Device on which all tensors are moved when iterating.
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
    """Store multiple ImageDict instances for training purpose.

    Parameters
    ----------
    images : list[ImageDict]
        List of stored ImageDicts.
    weights : list[float]
        Weight associated to in ImageDict instance. This is used for
        the weighting in the loss computation process.
    """

    images: list[ImageDict]
    weights: list[float]


def iterate_broadcasted_dims(
    train: TrainingImages,
    input_names: list[str],
    target_name: str,
    band: SensorBand,
) -> Generator[dict[str, float], None, None]:
    """Iterate on the broadcasted dimensions of a TrainingImages.

    Broadcasts all input variables and the target together, then yields
    one selector dict per combination of non-spatial coordinates.

    Parameters
    ----------
    train : TrainingImages
        List of ImageDict used for the training.
    input_names : list[str]
        Input variable names (all are broadcast together with the target).
    target_name : str
        Target variable name.
    band : SensorBand
        Sensor band of interest.

    Yields
    ------
    dict[str, float]
        Mapping from extra-dimension name to coordinate value.

    Raises
    ------
    ValueError
        If broadcast dimensions are inconsistent across ImageDict instances.
    ValueError
        If ``"x"`` or ``"y"`` are absent from the broadcast dimensions.
    """
    dims: list[str] = []
    coords: list[xr.DataArray] = []

    for idx, im in enumerate(train.images):
        arrays = [im[band][name] for name in [*input_names, target_name]]
        broadcasted, *_ = xr.broadcast(*arrays)

        current_dims = [str(d) for d in broadcasted.dims]
        dims = current_dims if not dims else dims

        if current_dims != dims:
            raise ValueError(
                f"Dims of ImageDict {idx} ({current_dims}) not consistent "
                f"with the other ImageDict dims ({dims})"
            )

        current_coords = [broadcasted.coords[dim] for dim in dims]
        coords = current_coords if not coords else coords
        diff = [
            bool((c1 != c2).any()) for c1, c2 in zip(coords, current_coords)
        ]
        if any(diff):
            raise ValueError(
                "Mismatching coordinates between trained ImageDicts."
            )

    if "x" not in dims or "y" not in dims:
        raise ValueError("Dimensions should contain (x, y)")

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
    """Return the input and target tensors for the training loop.

    Selects a specific atmospheric combination (via ``**params``) from each
    ImageDict and converts to tensors.  Radial distances are derived from the
    spatial coordinates of the target array.  Only dimensions that exist in
    each variable are used for selection (see :func:`_safe_sel`).

    Parameters
    ----------
    train : TrainingImages
        List of ImageDict used for the training.
    input_names : list[str]
        Input variable names to extract and bundle as ``dict[str, Tensor]``.
    target_name : str
        Target variable name.
    band : SensorBand
        Sensor band of interest.
    **params : float
        Coordinate selectors for the atmospheric dimensions
        (e.g. ``aot=0.1, rh=80``).

    Returns
    -------
    TrainingSet
        The lists of input dicts, target tensors, distance tensors, weights
        and the ``params`` dict used for slicing.
    """
    ipts: list[dict[str, torch.Tensor]] = [
        {
            name: _safe_sel(im[band][name], params).adjeff.to_tensor()
            for name in input_names
        }
        for im in train.images
    ]
    tgts: list[torch.Tensor] = [
        _safe_sel(im[band][target_name], params).adjeff.to_tensor()
        for im in train.images
    ]
    dists: list[torch.Tensor] = [
        _safe_sel(im[band][target_name], params).adjeff.dists
        for im in train.images
    ]

    return TrainingSet(
        inputs=ipts,
        targets=tgts,
        dists=dists,
        weights=train.weights,
        params=dict(params),
        device=device,
    )
