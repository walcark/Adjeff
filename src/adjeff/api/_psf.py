"""Build, fit and apply a PSF model.

Functions
---------
    M
        See :mod:`adjeff.api`.
    make_model
        See :mod:`adjeff.api`.
    ARTICLE_TRAIN_RADII_KM
        See :mod:`adjeff.api`.
    ARTICLE_TRAIN_SIZE
        See :mod:`adjeff.api`.
    fit_psf
        See :mod:`adjeff.api`.
    apply_psf
        See :mod:`adjeff.api`.
    sample_psf_atm
        See :mod:`adjeff.api`.
    sample_psf_atm_from_scene
        See :mod:`adjeff.api`.
"""

from __future__ import annotations

from typing import TypeVar, cast

import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig
from adjeff.core import (
    ImageDict,  # noqa: E402
    PSFGrid,
    SensorBand,
    disk_image_dict,
    gaussian_image_dict,
    psf_params,
    psf_tree,
)
from adjeff.core._psf import PSFModule  # not in core.__init__
from adjeff.modules.models import Unif2Surface
from adjeff.modules.models.psf_conv_module import (
    PSFConvModule,
)  # not in models.__init__
from adjeff.optim import (
    Loss,
    Metric,
    OptimizerConfig,
    TrainingImages,
    default_stages,
    fit,
)
from adjeff.reference import WuPsfSampler
from adjeff.utils import CacheStore

from ._config import _res_from_scene, _to_scalar, load_config
from ._pipeline import run_forward_pipeline

M = TypeVar("M", bound=PSFConvModule)


def _build_psfs(
    psf_type: type[PSFModule],
    bands: list[SensorBand],
    res_km: float,
    n: int,
    init_parameters: dict[str, float] | dict[SensorBand, dict[str, float]],
) -> dict[SensorBand, PSFModule]:
    """Return one *psf_type* per band, on a common ``n × n`` grid.

    *init_parameters* is either shared, ``{"sigma": 0.1}``, or per band,
    ``{S2Band.B02: {"sigma": 0.1}, ...}``.
    """
    per_band = bool(init_parameters) and isinstance(
        next(iter(init_parameters)), SensorBand
    )
    grid = PSFGrid(res=res_km, n=n)
    psfs: dict[SensorBand, PSFModule] = {}
    for band in bands:
        params = (
            cast(dict[SensorBand, dict[str, float]], init_parameters)[band]
            if per_band
            else cast(dict[str, float], init_parameters)
        )
        psfs[band] = psf_type(grid=grid, band=band, **params)
    return psfs


def make_model(
    model_cls: type[M],
    psf_type: type[PSFModule],
    bands: list[SensorBand],
    res_km: float,
    n: int,
    init_parameters: dict[str, float] | dict[SensorBand, dict[str, float]],
    device: str = "cuda",
    cache: CacheStore | None = None,
) -> M:
    """Return a *model_cls* holding one live *psf_type* per band.

    Parameters
    ----------
    model_cls : type[PSFConvModule]
        Model, e.g. ``Unif2Surface``.
    psf_type : type[PSFModule]
        PSF, e.g. ``KingPSF``.
    init_parameters : dict
        Initial PSF parameters, shared ``{"sigma": 0.1}`` or per band
        ``{S2Band.B02: {"sigma": 0.1}, ...}``.
    cache : CacheStore or None, optional
        Shared disk cache.

    The other parameters are those of the module docstring.
    """
    return model_cls(
        psfs=_build_psfs(psf_type, bands, res_km, n, init_parameters),
        device=device,
        cache=cache,
    )


ARTICLE_TRAIN_RADII_KM: tuple[float, ...] = (1.0, 5.0, 50.0)

ARTICLE_TRAIN_SIZE: int = 1999

_DEFAULT_LOSS = Loss(Metric.RMSE_RAD)


def fit_psf(
    scene: ImageDict,
    bands: list[SensorBand],
    psf_type: type[PSFModule],
    init_parameters: dict[str, float],
    *,
    model_cls: type[PSFConvModule] = Unif2Surface,
    target_var: str | None = None,
    train_radii: list[float] | None = None,
    stages: list[OptimizerConfig] | None = None,
    res_km: float | None = None,
    n_train: int | None = None,
    cache: CacheStore | None = None,
    device: str = "cuda",
) -> xr.DataTree:
    """Fit a *psf_type* per band on disk scenes, and return the frozen tree.

    The disk scenes are simulated with the mean atmosphere and geometry
    of ``scene[bands[0]]``, then :func:`~adjeff.optim.fit` runs.

    Parameters
    ----------
    psf_type : type[PSFModule]
        PSF, e.g. :class:`~adjeff.core.KingPSF`.
    init_parameters : dict[str, float]
        Initial PSF parameters, shared by the bands.
    model_cls : type[PSFConvModule], optional
        Model fitted, ``Unif2Surface`` by default.
    target_var : str or None, optional
        Variable fitted; the model's output by default.
    train_radii : list[float] or None, optional
        Disk radii [km], those of the manuscript by default.
    stages : list[OptimizerConfig] or None, optional
        Optimiser stages, :func:`~adjeff.optim.default_stages` by default.
    res_km : float or None, optional
        Pixel size [km], that of the scene by default.
    n_train : int or None, optional
        Grid side of the disk scenes and PSFs, 1999 by default.

    The other parameters are those of the module docstring.
    """
    # Read off the class, so the role rather than any instance's slot.
    _target_var: str = target_var or model_cls._output_vars[0]
    _res_km: float = res_km or _res_from_scene(scene, bands[0])
    _stages = default_stages() if stages is None else stages
    _radii = train_radii or list(ARTICLE_TRAIN_RADII_KM)
    _n_train = n_train or ARTICLE_TRAIN_SIZE

    cfg = load_config(scene, bands[0], aggregate=True)

    disk_scenes = [
        disk_image_dict(
            radius=r,
            res_km=_res_km,
            rho_min=0.0,
            rho_max=0.5,
            bands=bands,
            var=_target_var,
            n=_n_train,
        )
        for r in _radii
    ]
    disk_scenes = run_forward_pipeline(disk_scenes, **cfg, cache=cache)
    train_images = TrainingImages(
        images=disk_scenes,
        weights=[1.0] * len(disk_scenes),
    )

    model = model_cls(
        psfs=_build_psfs(psf_type, bands, _res_km, _n_train, init_parameters),
        device=device,
        cache=cache,
    )

    return fit(model, train_images, stages=_stages, device=device)


def apply_psf(
    scene: ImageDict,
    tree: xr.DataTree,
    band: SensorBand,
    *,
    model_cls: type[PSFConvModule] = Unif2Surface,
    psf_type: type[PSFModule] | None = None,
    n: int | None = None,
    res_km: float | None = None,
    device: str = "cuda",
) -> ImageDict:
    """Return *scene* with the output of *model_cls* applied with *tree*.

    The kernels of *tree* are used as they are, or, given *n*, rebuilt
    as *psf_type* on an ``n × n`` grid from the parameters it stores.

    Parameters
    ----------
    tree : xr.DataTree
        Frozen PSF tree, e.g. from :func:`fit_psf`.
    model_cls : type[PSFConvModule], optional
        Model applied, ``Unif2Surface`` by default.
    psf_type : type[PSFModule] or None, optional
        PSF to rebuild; required with *n*.
    res_km : float or None, optional
        Pixel size of the rebuilt PSF [km], that of the scene by default.

    The other parameters are those of the module docstring.

    Raises
    ------
    ConfigurationError
        If *n* is given without *psf_type*.
    """
    from adjeff.exceptions import ConfigurationError

    if n is not None and psf_type is None:
        raise ConfigurationError("`psf_type` is required when `n` is provided.")

    if n is not None and psf_type is not None:
        params: dict[str, float] = {
            name: _to_scalar(value) for name, value in psf_params(tree, band).items()
        }
        _res_km: float = res_km or _res_from_scene(scene, band)
        model = model_cls(
            psfs=_build_psfs(psf_type, [band], _res_km, n, params),
            device=device,
        )
    else:
        model = model_cls(kernels=tree, device=device)

    model.eval()
    return model(scene)  # type: ignore[no-any-return]


def sample_psf_atm(
    bands: list[SensorBand],
    res_km: float,
    n: int,
    atmo_config: AtmoConfig,
    geo_config: GeoConfig,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    n_ph: int = int(1e6),
    cache: CacheStore | None = None,
) -> xr.DataTree:
    """Return the PSF tree of :class:`~adjeff.reference.WuPsfSampler` (GPU).

    Sampled on an ``n × n`` grid of pixel *res_km*, keeping the swept
    atmospheric dims.  Parameters are those of the module docstring.
    """
    scene = gaussian_image_dict(
        sigma=res_km * n,
        res_km=res_km,
        rho_min=0.5,
        rho_max=0.5,
        bands=bands,
        n=n,
    )
    sampler = WuPsfSampler(
        atmo_config=atmo_config,
        geo_config=geo_config,
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        n_ph=n_ph,
        cache=cache,
    )
    out = sampler(scene)
    return psf_tree({band: out[band]["psf_atm"] for band in out.bands})


def sample_psf_atm_from_scene(
    scene: ImageDict,
    band: SensorBand,
    *,
    n: int = 1999,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    n_ph: int = int(1e6),
    res_km: float | None = None,
    cache: CacheStore | None = None,
) -> xr.DataTree:
    """Return :func:`sample_psf_atm` of *band*, at the mean state of *scene*.

    *res_km* defaults to the pixel size of the scene; the other
    parameters are those of the module docstring.
    """
    cfg = load_config(scene, band, aggregate=True)
    _res_km: float = res_km or _res_from_scene(scene, band)

    return sample_psf_atm(
        bands=[band],
        res_km=_res_km,
        n=n,
        atmo_config=cfg["atmo_config"],
        geo_config=cfg["geo_config"],
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        n_ph=n_ph,
        cache=cache,
    )
