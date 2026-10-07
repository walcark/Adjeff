"""High-level convenience API for the Adjeff library.

Every function here composes lower-level building blocks; nothing is
computed that could not be written by hand with the modules themselves.

**Loading**: :func:`load_scene`, :func:`load_maja`.

**Configuration**: :func:`make_full_config` from scalars,
:func:`load_config` from an already-loaded scene.

**Pipelines**: :func:`run_radiatives_from_scene`,
:func:`run_forward_pipeline`.

**PSF**: :func:`make_model`, :func:`fit_psf`, :func:`apply_psf`,
:func:`sample_psf_atm`, :func:`sample_psf_atm_from_scene`.

Typical usage
-------------
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
    """Instantiate one live PSF module per band on a common grid.

    Parameters
    ----------
    psf_type : type[PSFModule]
        PSF model class, e.g. ``KingPSF``.
    bands : list[SensorBand]
        Bands to build a PSF for.
    res_km : float
        Pixel size in km.
    n : int
        Grid side in pixels; must be odd and >= 3.
    init_parameters : dict
        Either ``{"sigma": 0.1}`` shared by every band, or
        ``{S2Band.B02: {"sigma": 0.1}, ...}`` per band.

    Returns
    -------
    dict[SensorBand, PSFModule]
        Ready to hand to a ``PSFConvModule`` as ``psfs=``.
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
    """Build an :class:`~adjeff.atmosphere.AtmoConfig` with sensible defaults.

    Each parameter accepts a single float, a list of floats (swept as a
    1-D DataArray), or a pre-built DataArray (e.g. for multi-dim sweeps).

    Parameters
    ----------
    aot : float or list or DataArray, optional
        Aerosol optical thickness (default 0.1).
    rh : float or list or DataArray, optional
        Relative humidity [%] (default 50.0).
    h : float or list or DataArray, optional
        Ground elevation [km] (default 0.0).
    href : float or list or DataArray, optional
        Aerosol scale height [km] (default 2.0).
    species : dict[str, float] or None, optional
        Aerosol species mix summing to 1.0 (default ``{"sulphate": 1.0}``).

    Returns
    -------
    AtmoConfig
    """
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
    """Build a :class:`~adjeff.atmosphere.GeoConfig` with sensible defaults.

    Parameters
    ----------
    sza : float or list or DataArray, optional
        Sun zenith angle [°] (default 30.0).
    vza : float or list or DataArray, optional
        Viewing zenith angle [°] (default 0.0).
    saa : float or list or DataArray, optional
        Sun azimuth angle [°] (default 120.0).
    vaa : float or list or DataArray, optional
        Viewing azimuth angle [°] (default 120.0).
    sat_height : float, optional
        Satellite altitude [km] (default 786.0).

    Returns
    -------
    GeoConfig
    """
    return GeoConfig(
        sza=_da(sza, "sza"),
        vza=_da(vza, "vza"),
        saa=_da(saa, "saa"),
        vaa=_da(vaa, "vaa"),
        sat_height=sat_height,
    )


class FullConfig(TypedDict):
    """Typed dict returned by :func:`make_full_config`.

    Keys match the keyword arguments expected by
    :class:`~adjeff.modules.samplers.RadiativePipeline` and
    :class:`~adjeff.modules.samplers.RhoToaSymSampler`, so the dict
    can be unpacked directly with ``**cfg``.
    """

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
    """Build a complete config dict from raw parameters.

    Single entry point that internally calls :func:`_make_atmo_config`,
    :func:`_make_geo_config`, and :class:`~adjeff.atmosphere.SpectralConfig`.
    The returned dict has keys ``"atmo_config"``, ``"geo_config"``,
    ``"spectral_config"`` and can be unpacked directly with ``**cfg`` into
    :class:`~adjeff.modules.samplers.RadiativePipeline` and
    :class:`~adjeff.modules.samplers.RhoToaSymSampler`.

    Parameters
    ----------
    bands : list[SensorBand]
        Sensor bands to simulate.
    aot : float or list or DataArray, optional
        Aerosol optical thickness (default 0.1).
    rh : float or list or DataArray, optional
        Relative humidity [%] (default 50.0).
    h : float or list or DataArray, optional
        Ground elevation [km] (default 0.0).
    href : float or list or DataArray, optional
        Aerosol scale height [km] (default 2.0).
    species : dict[str, float] or None, optional
        Aerosol species mix summing to 1.0 (default ``{"sulphate": 1.0}``).
    sza : float or list or DataArray, optional
        Sun zenith angle [°] (default 30.0).
    vza : float or list or DataArray, optional
        Viewing zenith angle [°] (default 0.0).
    saa : float or list or DataArray, optional
        Sun azimuth angle [°] (default 120.0).
    vaa : float or list or DataArray, optional
        Viewing azimuth angle [°] (default 120.0).
    sat_height : float, optional
        Satellite altitude [km] (default 786.0).

    Returns
    -------
    FullConfig
        A plain ``dict`` with three typed entries.
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
    """Apply *run* to *scene*, giving back whatever shape it came in.

    The dispatch happens in :func:`_run_each`, whose return type is the
    plain union: *SceneT* is constrained rather than bound, so mypy checks
    this body once per member, and a cast written where the type is
    already narrowed is redundant under one of them.
    """
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
    """Instantiate a :class:`~adjeff.modules.models.PSFConvModule` subclass.

    Creates a :class:`~adjeff.core.PSFGrid` and one live PSF module
    per band, then constructs the model.

    Parameters
    ----------
    model_cls : type[PSFConvModule]
        Concrete subclass to instantiate (e.g. ``Unif2Surface``).
    psf_type : type[PSFModule]
        PSF model class (e.g. ``KingPSF``, ``GaussPSF``).
    bands : list[SensorBand]
        Sensor bands to include.
    res_km : float
        Pixel size in km (passed to :class:`~adjeff.core.PSFGrid`).
    n : int
        Grid side in pixels — must be odd and ≥ 3.
    init_parameters : dict[str, float] or dict[SensorBand, dict[str, float]]
        Initial PSF parameters, either shared across bands (flat dict) or
        per-band (nested dict keyed by :class:`~adjeff.core.SensorBand`).
    device : str, optional
        PyTorch device (default ``"cuda"``).
    cache : CacheStore or None, optional
        Shared on-disk cache (default ``None``).

    Returns
    -------
    M
        An instance of *model_cls*.
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
    """Run the full forward pipeline: radiatives → rho_toa → rho_unif.

    Chains :class:`~adjeff.modules.samplers.RadiativePipeline`,
    :class:`~adjeff.modules.samplers.RhoToaSymSampler`, and
    :class:`~adjeff.modules.classic.Toa2Unif` in sequence.

    The config arguments match the keys of :func:`make_full_config`, so the
    dict can be unpacked directly::

        cfg = make_full_config(bands=[S2Band.B03], aot=0.1)

        # single scene
        scene = run_forward_pipeline(scene, **cfg)

        # multiple scenes — modules instantiated once, applied to each
        scenes = run_forward_pipeline([s1, s2, s3], **cfg)

    Parameters
    ----------
    scene : ImageDict or list[ImageDict]
        One scene or a list of scenes, each containing ``rho_s``.
    atmo_config : AtmoConfig
    geo_config : GeoConfig
    spectral_config : SpectralConfig
    cache : CacheStore or None, optional
        Shared on-disk cache (default ``None``).
    remove_rayleigh : bool, optional
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str, optional
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    nr : int, optional
        Radial sampling points for rho_toa (default 500).
    n_ph : int, optional
        Photon count per sensor for rho_toa (default ``1e5``).
    batch_size : int, optional
        Atmospheric states handed to Smart-G in one call inside
        :class:`~adjeff.modules.samplers.RadiativePipeline`. A cost
        decision only: it bounds GPU memory and never changes a value.
    rtls : tuple[float, float, float] or None, optional
        Ross-Li kernel weights ``(k0, k1p, k2p)`` of a non-lambertian
        surface, forwarded to
        :class:`~adjeff.modules.samplers.RadiativePipeline`.  ``None``,
        the default, keeps the Lambertian samplers.  It changes only the
        scalar terms the inversion uses, not the surface ``rho_toa`` is
        simulated over, so the two can be varied independently.
    stream_dims : dict[str, int] or None, optional
        Dimensions to stream over for memory management, e.g.
        ``{"aot": 3}``.  When a dimension exists in the scene's DataArrays,
        the pipeline processes that many values at a time through the full
        chain (radiatives → rho_toa → Toa2Unif), preventing OOM on large
        scenes.  ``None`` disables streaming.

    Returns
    -------
    ImageDict or list[ImageDict]
        Same type as *scene*, enriched with ``rho_toa``, radiative
        quantities, and ``rho_unif``.
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
    """Infer pixel size [km] from the y-coordinate spacing of *scene[band]*."""
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
    """Load a scene from any :class:`~adjeff.modules.loaders.ProductLoader`.

    Stores aerosol species in ``scene[band].attrs["adjeff:species"]`` for
    every band so that :func:`load_config` can recover them later.  Species
    are resolved in this order:

    1. *species* argument if provided,
    2. ``loader.species()`` if the loader exposes that method,
    3. ``{"sulphate": 1.0}`` as a last-resort default.

    Parameters
    ----------
    loader : ProductLoader
        Pre-instantiated loader (e.g. ``MajaLoader(...)``).
    compute_radiatives : bool, optional
        Run the radiative pipeline after loading (default ``False``).
    n_bins : int or None, optional
        Digitise ``aot`` and ``h`` to *n_bins* unique values before building
        the config, cutting the number of distinct Smart-G runs.  Ignored when
        *compute_radiatives* is ``False``.
    species : dict[str, float] or None, optional
        Override aerosol species mix.
    remove_rayleigh : bool, optional
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str, optional
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    cache : CacheStore or None, optional
        Shared on-disk cache (default ``None``).
    dedup : bool, optional
        Collapse repeated atmospheric states before calling Smart-G.
        Worth it when the parameters are spatial maps, where many pixels
        share a state; pure overhead when every state is distinct.

    Returns
    -------
    ImageDict
        Scene with ``scene[band].attrs["adjeff:species"]`` set for every band.
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
    """Load a MAJA L2A product via :func:`load_scene`.

    Convenience wrapper that instantiates
    :class:`~adjeff.modules.loaders.MajaLoader` and delegates to
    :func:`load_scene`, which persists CAMS aerosol species in
    ``scene[band].attrs["adjeff:species"]``.

    Parameters
    ----------
    product_path : Path
        Folder containing the MAJA product.
    bands : list[SensorBand]
        Bands to load.
    res : float or list[float]
        Target spatial resolution in km (e.g. ``0.12`` for 120 m).
    mnt_path : Path or None, optional
        Folder containing the DEM at 20 m resolution.  Must be provided;
        ``None`` raises :class:`~adjeff.exceptions.ConfigurationError`.
    href : float, optional
        Aerosol scale height [km] (default ``2.0``).
    as_map : bool, optional
        When ``True``, load 2-D atmospheric parameters as full spatial maps
        instead of spatially-averaged scalars (default ``False``).
    cache : CacheStore or None, optional
        Shared on-disk cache (default ``None``).
    compute_radiatives : bool, optional
        When ``True``, run the radiative pipeline after loading (default
        ``False``).
    n_bins : int or None, optional
        Digitise ``aot`` and ``h`` to *n_bins* unique values before building
        the config, cutting the number of distinct Smart-G runs.  Ignored when
        *compute_radiatives* is ``False``.
    remove_rayleigh : bool, optional
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str, optional
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    dedup : bool, optional
        Collapse repeated atmospheric states before calling Smart-G.
        Worth it when the parameters are spatial maps, where many pixels
        share a state; pure overhead when every state is distinct.

    Returns
    -------
    ImageDict
        Scene with ``rho_s``, atmospheric/geometric variables, and
        ``scene[band].attrs["adjeff:species"]`` for every band.
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
    """Run the radiative pipeline using configs embedded in *scene*.

    Unlike :func:`run_forward_pipeline` which takes explicit config objects,
    this function reads ``aot``, ``h``, ``rh``, ``href``, ``vza``, ``vaa``,
    ``sza``, ``saa`` directly from the scene (as produced by a
    :class:`~adjeff.modules.loaders.ProductLoader`).

    Because viewing geometry (``vza``, ``vaa``) varies across S2 bands, a
    separate :class:`~adjeff.modules.samplers.RadiativePipeline` is built
    and run for each band.  Results are merged back into a single scene.

    Parameters
    ----------
    scene : ImageDict or list[ImageDict]
        Scene(s) produced by a ProductLoader (must contain the atmospheric
        and geometric variables listed above).
    n_bins : int or None, optional
        Digitise ``aot`` and ``h`` to *n_bins* unique values before building
        the config, cutting the number of distinct Smart-G runs.
    species : dict[str, float] or None, optional
        Aerosol species mix summing to 1.0.  Defaults to
        ``{"sulphate": 1.0}`` when ``None``.
    remove_rayleigh : bool, optional
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str, optional
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    cache : CacheStore or None, optional
        Shared on-disk cache (default ``None``).
    dedup : bool, optional
        Collapse repeated atmospheric states before calling Smart-G.
        Worth it when the parameters are spatial maps, where many pixels
        share a state; pure overhead when every state is distinct.

    Returns
    -------
    ImageDict or list[ImageDict]
        Same type as *scene*, enriched with the six radiative quantities
        (``tdir_down``, ``tdif_down``, ``tdir_up``, ``tdif_up``,
        ``rho_atm``, ``sph_alb``).
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
    """Sample the atmospheric PSF and return a frozen PSF tree.

    Internally builds a constant input scene to carry the spatial grid
    (only ``res`` and ``n`` matter to the sampler — the reflectance values
    are irrelevant), runs
    :class:`~adjeff.reference.WuPsfSampler`, then wraps the
    resulting ``psf_atm`` DataArrays into a PSF tree.

    Requires a CUDA GPU (delegates to Smart-G).

    Parameters
    ----------
    bands : list[SensorBand]
        Sensor bands to simulate.
    res_km : float
        Pixel size [km] — defines the Smart-G Entity sampling grid.
    n : int
        Grid side in pixels (must be odd and ≥ 3).
    atmo_config : AtmoConfig
        Atmospheric parameters (may contain swept dimensions).
    geo_config : GeoConfig
        Geometric parameters (sza, vza, saa, vaa must be scalar per call).
    remove_rayleigh : bool, optional
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str, optional
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    n_ph : int, optional
        Photon count per Smart-G run (default ``1e6``).
    cache : CacheStore or None, optional
        Shared on-disk cache (default ``None``).

    Returns
    -------
    xr.DataTree
        One group per band, each holding a ``kernel``.
        Extra atmospheric dimensions (``aot``, ``rh``, ...) are preserved.
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
    """Build a :class:`FullConfig` from the fields stored in a scene.

    Reads the atmospheric and geometric parameters from ``scene[band]``,
    which avoids the coordinate-alignment pitfalls of building a config
    independently from the scene it describes.  On top of that:

    - **Species recovery** from ``scene[band].attrs["adjeff:species"]``
      (written by :func:`load_scene`) when *species* is ``None``.
    - **Spatial aggregation** (``aggregate=True``) to reduce all fields to
      scalars via ``.mean()``, useful when a single representative
      atmospheric state is needed.
    - **Digitisation** (``n_bins``) of ``aot`` and ``h``, which cuts the
      number of distinct atmospheric states to simulate.

    Parameters
    ----------
    scene : ImageDict
        Scene produced by :func:`load_scene` or another loader.
    band : SensorBand
        Band from which to read the parameters.
    aggregate : bool, optional
        Reduce every field to its spatial mean.  Incompatible with *n_bins*.
    n_bins : int or None, optional
        Digitise ``aot`` and ``h`` to *n_bins* unique values before building
        the config, cutting the number of distinct Smart-G runs.  Incompatible
        with *aggregate*.
    species : dict[str, float] or None, optional
        Aerosol species mix.  Resolution order: argument → attrs → default.

    Returns
    -------
    FullConfig

    Raises
    ------
    MissingVariableError
        If any required variable is absent from ``scene[band]``.
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
    """Fit a PSF model for one or more bands in one call.

    Wraps the full pipeline: build disk training scenes → forward pipeline
    → instantiate model → run optimiser stages → return frozen
    PSF tree.

    Configuration is derived from ``bands[0]`` via :func:`load_config`
    with ``aggregate=True``.

    Parameters
    ----------
    scene : ImageDict
        Reference scene with atmospheric and geometric fields.
    bands : list[SensorBand]
        Bands for which to fit the PSF.
    psf_type : type[PSFModule]
        Analytical PSF class (e.g. :class:`~adjeff.core.KingPSF`).
    init_parameters : dict[str, float]
        Initial parameter values shared across bands.
    model_cls : type[PSFConvModule], optional
        Inverse model to optimise
        (default :class:`~adjeff.modules.models.Unif2Surface`).
    target_var : str or None, optional
        Optimisation target variable.  ``None`` → ``model_cls.output_vars[0]``.
    train_radii : list[float] or None, optional
        Disk radii [km] for training scenes (default ``[1, 5, 50]`` km).
    stages : list[AdamConfig | LBFGSConfig] or None, optional
        Optimiser stages.  ``None`` → Adam (20 steps) + L-BFGS (30 steps).
    res_km : float or None, optional
        PSF grid pixel size [km].  ``None`` → inferred from *scene[bands[0]]*.
    n_train : int or None, optional
        PSF grid side in pixels for training (default 1999, must be odd).
    cache : CacheStore or None, optional
        Shared on-disk cache (default ``None``).
    device : str, optional
        PyTorch device (default ``"cuda"``).

    Returns
    -------
    xr.DataTree
        Frozen PSF tree with one optimised kernel per band.
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
    """Apply a frozen PSF tree to a scene to predict the output variable.

    Two modes:

    - **Direct** (``n=None``): wraps *tree* in a model and applies it
      as-is.  The kernel size equals the one used during training.
    - **Rebuild** (``n`` provided): reconstructs the PSF on an *n*×*n* grid
      from the stored parameters, then applies it.  Requires *psf_type*.

    Parameters
    ----------
    scene : ImageDict
        Scene containing the variables required by *model_cls*.
    tree : xr.DataTree
        Frozen PSF tree, typically returned by :func:`fit_psf`.
    band : SensorBand
        Band to apply.
    model_cls : type[PSFConvModule], optional
        Model class (default :class:`~adjeff.modules.models.Unif2Surface`).
    psf_type : type[PSFModule] or None, optional
        PSF class for rebuild mode.  Required when *n* is provided.
    n : int or None, optional
        Rebuild the PSF on an *n*×*n* grid (must be odd).
    res_km : float or None, optional
        Pixel size [km] for the rebuild grid.  ``None`` → inferred from
        *scene[band]*.
    device : str, optional
        PyTorch device (default ``"cuda"``).

    Returns
    -------
    ImageDict
        Copy of *scene* enriched with the model's output variable.

    Raises
    ------
    MissingVariableError
        If *n* is provided without *psf_type*.
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
    """Sample the atmospheric PSF for a given scene.

    Derives configuration from *scene* via :func:`load_config` with
    ``aggregate=True`` (the PSF sampler requires scalar geometric parameters).

    Parameters
    ----------
    scene : ImageDict
        Reference scene with atmospheric and geometric fields for *band*.
    band : SensorBand
        Band of interest.
    n : int, optional
        PSF grid side in pixels (default 1999, must be odd).
    remove_rayleigh : bool, optional
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str, optional
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    n_ph : int, optional
        Photon count per Smart-G run (default ``1e6``).
    res_km : float or None, optional
        Pixel size [km].  ``None`` → inferred from *scene[band]*.
    cache : CacheStore or None, optional
        Shared on-disk cache (default ``None``).

    Returns
    -------
    xr.DataTree
        Frozen PSF tree with one ``kernel`` for *band*.
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
