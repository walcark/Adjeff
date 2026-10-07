"""High-level API, composing the modules of adjeff.

Functions
---------
    load_scene, load_maja
        Load a product into a scene.
    make_full_config, load_config
        Configs from scalars, or from the fields of a scene.
    run_radiatives_from_scene, run_forward_pipeline
        Radiative terms, or the whole forward chain down to ``rho_unif``.
    make_model, fit_psf, apply_psf
        Build, fit and apply a PSF model.
    sample_psf_atm, sample_psf_atm_from_scene
        Atmospheric PSF of Wu et al. (2024).

Common parameters
-----------------
bands : list[SensorBand]
    Bands to process.
aot, rh, h, href : float, list or DataArray
    Aerosol optical thickness, relative humidity [%], ground elevation
    [km] and aerosol scale height [km]; a list is swept.
sza, vza, saa, vaa : float, list or DataArray
    Sun and viewing zenith and azimuth angles [°].
species : dict[str, float] or None
    Aerosol species fractions; ``{"sulphate": 1.0}`` by default.
atmo_config, geo_config, spectral_config : configs
    As returned by :func:`make_full_config`.
res_km : float
    Pixel size [km].
n : int
    Grid side in pixels, odd.
remove_rayleigh : bool
    Suppress Rayleigh scattering.
afgl_type : str
    AFGL atmosphere profile, ``"afgl_exp_h8km"`` by default.
n_ph : int
    Photons per Smart-G run.
n_bins : int or None
    Digitise ``aot`` and ``h`` to this many values, to cut Smart-G runs.
dedup : bool
    Merge repeated atmospheric states; worth it for spatial maps.
cache : CacheStore or None
    Shared disk cache.
device : str
    Torch device, ``"cuda"`` by default.

Example
-------
>>> cfg = make_full_config(bands=[S2Band.B03], aot=0.1, rh=50.0, sza=30.0, vza=0.0)
>>> model = make_model(
...     Unif2Surface,
...     KingPSF,
...     [S2Band.B03],
...     res_km=0.12,
...     n=1999,
...     init_parameters={"sigma": 0.1, "gamma": 1.0},
... )
>>> scene = run_forward_pipeline(rho_s_scene, **cfg)
>>> psf_tree = fit(model, train_images, loss=Loss(Metric.RMSE_RAD))
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, TypeVar, cast

import numpy as np
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
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
from adjeff.exceptions import MissingVariableError
from adjeff.modules import Pipeline
from adjeff.modules.classic import Toa2Unif
from adjeff.modules.loaders import MajaLoader, ProductLoader
from adjeff.modules.models import Unif2Surface
from adjeff.modules.models.psf_conv_module import (
    PSFConvModule,
)  # not in models.__init__
from adjeff.modules.samplers import RadiativePipeline, RhoToaSymSampler
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

__all__ = [
    "FullConfig",
    # Loading
    "load_scene",
    "load_maja",
    # Configuration
    "make_full_config",
    "load_config",
    # Pipelines
    "run_radiatives_from_scene",
    "run_forward_pipeline",
    # PSF
    "make_model",
    "fit_psf",
    "apply_psf",
    "sample_psf_atm",
    "sample_psf_atm_from_scene",
    # The manuscript's own choices, named so that a caller can tell them
    # apart from the library's defaults.
    "ARTICLE_TRAIN_RADII_KM",
    "ARTICLE_TRAIN_SIZE",
]


_Scalar = float | list[float] | xr.DataArray


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


def _da(val: _Scalar, dim: str) -> xr.DataArray:
    """Coerce a scalar/list/DataArray to a named-dim DataArray."""
    if isinstance(val, xr.DataArray):
        return val
    arr = np.atleast_1d(np.asarray(val, dtype=float))
    return xr.DataArray(arr, dims=[dim])


def _make_atmo_config(
    aot: _Scalar = 0.1,
    rh: _Scalar = 50.0,
    h: _Scalar = 0.0,
    href: _Scalar = 2.0,
    species: dict[str, float] | None = None,
) -> AtmoConfig:
    """Return an :class:`~adjeff.atmosphere.AtmoConfig`, with defaults."""
    if species is None:
        species = {"sulphate": 1.0}
    return AtmoConfig(
        aot=_da(aot, "aot"),
        rh=_da(rh, "rh"),
        h=_da(h, "h"),
        href=_da(href, "href"),
        species=species,
    )


def _make_geo_config(
    sza: _Scalar = 30.0,
    vza: _Scalar = 0.0,
    saa: _Scalar = 120.0,
    vaa: _Scalar = 120.0,
    sat_height: float = 786.0,
) -> GeoConfig:
    """Return a :class:`~adjeff.atmosphere.GeoConfig`, with defaults."""
    return GeoConfig(
        sza=_da(sza, "sza"),
        vza=_da(vza, "vza"),
        saa=_da(saa, "saa"),
        vaa=_da(vaa, "vaa"),
        sat_height=sat_height,
    )


class FullConfig(TypedDict):
    """The three configs, to unpack as ``**cfg`` into the samplers."""

    atmo_config: AtmoConfig
    geo_config: GeoConfig
    spectral_config: SpectralConfig


def make_full_config(
    bands: list[SensorBand],
    aot: _Scalar = 0.1,
    rh: _Scalar = 50.0,
    h: _Scalar = 0.0,
    href: _Scalar = 2.0,
    species: dict[str, float] | None = None,
    sza: _Scalar = 30.0,
    vza: _Scalar = 0.0,
    saa: _Scalar = 120.0,
    vaa: _Scalar = 120.0,
    sat_height: float = 786.0,
) -> FullConfig:
    """Return the atmosphere, geometry and spectral configs of *bands*.

    Parameters are those of the module docstring; *sat_height* is the
    satellite altitude [km].
    """
    return FullConfig(
        atmo_config=_make_atmo_config(aot=aot, rh=rh, h=h, href=href, species=species),
        geo_config=_make_geo_config(
            sza=sza, vza=vza, saa=saa, vaa=vaa, sat_height=sat_height
        ),
        spectral_config=SpectralConfig.from_bands(bands),
    )


M = TypeVar("M", bound=PSFConvModule)

#: One scene or a batch of them.  Constrained rather than bound, so that
#: a function taking a list is known to return a list.
SceneT = TypeVar("SceneT", ImageDict, list[ImageDict])


def _run_each(
    scene: ImageDict | list[ImageDict], run: Callable[[ImageDict], ImageDict]
) -> ImageDict | list[ImageDict]:
    """Apply *run* to one scene, or to each scene of a batch."""
    if isinstance(scene, list):
        return [run(s) for s in scene]
    return run(scene)


def _map_scenes(scene: SceneT, run: Callable[[ImageDict], ImageDict]) -> SceneT:
    """Apply *run* to *scene*, returning one scene or a list as given."""
    return cast(SceneT, _run_each(scene, run))


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


def run_forward_pipeline(
    scene: SceneT,
    atmo_config: AtmoConfig,
    geo_config: GeoConfig,
    spectral_config: SpectralConfig,
    cache: CacheStore | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    nr: int = 500,
    n_ph: int = int(1e5),
    batch_size: int = 64,
    rtls: tuple[float, float, float] | None = None,
    stream_dims: dict[str, int] | None = None,
) -> SceneT:
    """Add the radiative terms, ``rho_toa`` and ``rho_unif`` to *scene*.

    Chains :class:`~adjeff.modules.samplers.RadiativePipeline`,
    :class:`~adjeff.modules.samplers.RhoToaSymSampler` and
    :class:`~adjeff.modules.classic.Toa2Unif`; a list of scenes reuses
    the same modules.  Typically called as
    ``run_forward_pipeline(scene, **make_full_config(...))``.

    Parameters
    ----------
    scene : ImageDict or list[ImageDict]
        Scene(s) holding ``rho_s``.
    nr, n_ph : int, optional
        Radii and photons per sensor of ``rho_toa``.
    batch_size : int, optional
        Atmospheric states per Smart-G call; changes the cost only.
    rtls : tuple[float, float, float] or None, optional
        RTLS weights ``(k0, k1p, k2p)`` of the radiative terms; ``None``
        keeps them Lambertian.  ``rho_toa`` is unaffected.
    stream_dims : dict[str, int] or None, optional
        Run the chain on chunks of these dims, e.g. ``{"aot": 3}``.

    The other parameters are those of the module docstring.
    """
    radiative = RadiativePipeline(
        atmo_config=atmo_config,
        geo_config=geo_config,
        spectral_config=spectral_config,
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        cache=cache,
        batch_size=batch_size,
        rtls=rtls,
    )
    rho_toa = RhoToaSymSampler(
        atmo_config=atmo_config,
        geo_config=geo_config,
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        cache=cache,
        nr=nr,
        n_ph=n_ph,
    )
    pipeline = Pipeline(
        [radiative, rho_toa, Toa2Unif()],  # type: ignore[list-item]
        stream_dims=stream_dims,
    )

    return _map_scenes(scene, pipeline)


_SPECIES_ATTR = "adjeff:species"
_DEFAULT_SPECIES: dict[str, float] = {"sulphate": 1.0}
_DEFAULT_LOSS = Loss(Metric.RMSE_RAD)
ARTICLE_TRAIN_RADII_KM: tuple[float, ...] = (1.0, 5.0, 50.0)
ARTICLE_TRAIN_SIZE: int = 1999


def _res_from_scene(scene: ImageDict, band: SensorBand) -> float:
    """Return the pixel size [km] of *scene[band]*, from its ``y`` spacing."""
    y: xr.DataArray = scene[band].coords["y"]
    return float(abs(float(y[1]) - float(y[0])))


def _to_scalar(v: xr.DataArray | float) -> float:
    """Extract a Python float from a scalar, 0-d, or 1-element DataArray."""
    if isinstance(v, xr.DataArray):
        return float(v.values.flat[0])
    return float(v)


def load_scene(
    loader: ProductLoader,
    *,
    compute_radiatives: bool = False,
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    cache: CacheStore | None = None,
    dedup: bool = False,
) -> ImageDict:
    """Return the scene *loader* produces, with its aerosol species.

    The species, from *species*, else ``loader.species()``, else
    sulphate, are stored in ``attrs["adjeff:species"]`` of every band
    for :func:`load_config`.  With *compute_radiatives*, the radiative
    terms are added by :func:`run_radiatives_from_scene`.

    The other parameters are those of the module docstring.
    """
    scene = loader.forward()

    _species: dict[str, float] = (
        species
        or (loader.species() if hasattr(loader, "species") else None)
        or _DEFAULT_SPECIES
    )
    for band in scene.bands:
        scene[band].attrs[_SPECIES_ATTR] = _species

    if compute_radiatives:
        scene = run_radiatives_from_scene(
            scene,
            n_bins=n_bins,
            species=_species,
            remove_rayleigh=remove_rayleigh,
            afgl_type=afgl_type,
            cache=cache,
            dedup=dedup,
        )

    return scene


def load_maja(
    product_path: Path,
    bands: list[SensorBand],
    res: float | list[float],
    mnt_path: Path | None = None,
    href: float = 2.0,
    as_map: bool = False,
    cache: CacheStore | None = None,
    compute_radiatives: bool = False,
    n_bins: int | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    dedup: bool = False,
) -> ImageDict:
    """Return :func:`load_scene` of a :class:`~adjeff.modules.loaders.MajaLoader`.

    *product_path*, *res*, *mnt_path* (required), *href* and *as_map*
    are those of the loader; the rest are those of :func:`load_scene`.
    """
    if mnt_path is None:
        from adjeff.exceptions import ConfigurationError

        raise ConfigurationError(
            "load_maja requires mnt_path (path to the DEM folder). "
            "Example: mnt_path=Path('/data/dtm')."
        )
    loader = MajaLoader(
        product_path=product_path,
        bands=bands,
        res=res,
        mnt_path=mnt_path,
        href=href,
        as_map=as_map,
        cache=cache,
    )
    return load_scene(
        loader,
        compute_radiatives=compute_radiatives,
        n_bins=n_bins,
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        cache=cache,
        dedup=dedup,
    )


def run_radiatives_from_scene(
    scene: SceneT,
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    cache: CacheStore | None = None,
    dedup: bool = False,
) -> SceneT:
    """Add the six radiative terms to *scene*, configured from its own fields.

    The atmosphere and geometry are read from each band by
    :func:`load_config`, and a pipeline is run per band, since the
    viewing angles differ between bands.  *scene* may be a list.

    The other parameters are those of the module docstring.
    """

    def _run(s: ImageDict) -> ImageDict:
        s = s.shallow_copy()
        for band in s.bands:
            scene_band = ImageDict({band: s[band]})
            config = load_config(scene_band, band, n_bins=n_bins, species=species)
            radiative = RadiativePipeline(
                atmo_config=config["atmo_config"],
                geo_config=config["geo_config"],
                spectral_config=config["spectral_config"],
                remove_rayleigh=remove_rayleigh,
                afgl_type=afgl_type,
                cache=cache,
                dedup=dedup,
            )
            scene_band = radiative(scene_band)
            s[band] = scene_band[band]
        return s

    return _map_scenes(scene, _run)


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


_REQUIRED_VARS = ["aot", "h", "rh", "href", "vza", "vaa", "sza", "saa"]


def load_config(
    scene: ImageDict,
    band: SensorBand,
    *,
    aggregate: bool = False,
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
) -> FullConfig:
    """Return the configs of the atmosphere and geometry held by ``scene[band]``.

    Parameters
    ----------
    aggregate : bool, optional
        Reduce every field to its mean; exclusive with *n_bins*.
    species : dict[str, float] or None, optional
        Species; else those stored by :func:`load_scene`, else sulphate.

    The other parameters are those of the module docstring.

    Raises
    ------
    MissingVariableError
        If ``scene[band]`` lacks one of the fields.
    ConfigurationError
        If both *aggregate* and *n_bins* are given.
    """
    from adjeff.exceptions import ConfigurationError

    if aggregate and n_bins is not None:
        raise ConfigurationError("`aggregate` and `n_bins` are mutually exclusive.")

    ds = scene[band]
    missing = [v for v in _REQUIRED_VARS if v not in ds]
    if missing:
        raise MissingVariableError(
            f"Variables {missing!r} are missing from band {band!r}."
        )

    _species: dict[str, float] = (
        species or ds.attrs.get(_SPECIES_ATTR) or _DEFAULT_SPECIES
    )

    def _field(name: str) -> xr.DataArray:
        da: xr.DataArray = ds[name]
        if aggregate:
            return da.mean()
        if n_bins is not None and name in ("aot", "h"):
            return da.adjeff.digitize(n_bins=n_bins)  # type: ignore[no-any-return]
        return da

    return make_full_config(
        bands=scene.bands,
        aot=_field("aot"),
        h=_field("h"),
        rh=_field("rh"),
        href=_field("href"),
        vza=_field("vza"),
        vaa=_field("vaa"),
        sza=_field("sza"),
        saa=_field("saa"),
        species=_species,
    )


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
