"""High-level convenience API for the Adjeff library.

New functions added from api_bis:

- :func:`load_scene`        — generic loader with species persistence.
- :func:`load_maja`         — load_scene pre-wired for MajaLoader.
- :func:`load_config`       — FullConfig from a scene with aggregation.
- :func:`fit_psf`           — end-to-end PSF fitting in one call.
- :func:`apply_psf`         — apply a frozen PSF tree to a scene.
- :func:`sample_psf_atm_from_scene` — atmospheric PSF from a scene.


Typical usage
-------------
>>> cfg = make_full_config(
...     atmo=make_atmo_config(aot=0.1, rh=50.0),
...     geo=make_geo_config(sza=30.0, vza=0.0),
...     bands=[S2Band.B03],
... )
>>> model = make_model(
...     Unif2Surface,
...     KingPSF,
...     [S2Band.B03],
...     res_km=0.12,
...     n=1999,
...     init_parameters={"sigma": 0.1, "gamma": 1.0},
... )
>>> scene = run_forward_pipeline(rho_s_scene, **cfg)
>>> psf_dict = optimize_adam_lbfgs(model, train_images, Loss(Metric.RMSE_RAD))
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, TypeVar, cast, overload

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
    AdamConfig,
    AdamStage,
    LBFGSConfig,
    LBFGSStage,
    Loss,
    Metric,
    OptimizerPipeline,
    TrainingImages,
)
from adjeff.optim._combo_stage import _ComboStage  # private module
from adjeff.reference import WuPsfSampler
from adjeff.utils import CacheStore

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------


def make_atmo_config(
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
    aot : float or list or DataArray
        Aerosol optical thickness (default 0.1).
    rh : float or list or DataArray
        Relative humidity [%] (default 50.0).
    h : float or list or DataArray
        Ground elevation [km] (default 0.0).
    href : float or list or DataArray
        Aerosol scale height [km] (default 2.0).
    species : dict[str, float] or None
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


def make_geo_config(
    sza: _Scalar = 30.0,
    vza: _Scalar = 0.0,
    saa: _Scalar = 120.0,
    vaa: _Scalar = 120.0,
    sat_height: float = 786.0,
) -> GeoConfig:
    """Build a :class:`~adjeff.atmosphere.GeoConfig` with sensible defaults.

    Parameters
    ----------
    sza : float or list or DataArray
        Sun zenith angle [°] (default 30.0).
    vza : float or list or DataArray
        Viewing zenith angle [°] (default 0.0).
    saa : float or list or DataArray
        Sun azimuth angle [°] (default 120.0).
    vaa : float or list or DataArray
        Viewing azimuth angle [°] (default 120.0).
    sat_height : float
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


def config_from_scene(
    scene: ImageDict,
    band: SensorBand,
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
) -> FullConfig:
    """Build a :class:`FullConfig` from parameters stored in an ImageDict.

    Reads atmospheric and geometric parameters directly from
    ``scene[band]``, avoiding manual extraction and the coordinate-
    alignment pitfalls that arise when building configs independently
    from the scene.

    Parameters
    ----------
    scene : ImageDict
        Scene produced by a :class:`~adjeff.modules.loaders.ProductLoader`
        (must contain ``aot``, ``h``, ``rh``, ``href``, ``vza``, ``vaa``,
        ``sza``, ``saa`` in the Dataset for *band*).
    band : SensorBand
        Band from which to read the parameters.
    n_bins : int or None, optional
        If provided, ``aot`` and ``h`` are digitized to *n_bins* unique
        values before building the config, reducing the number of unique
        atmospheric configurations to simulate.
    species : dict[str, float] or None, optional
        Aerosol species mix summing to 1.0.  Defaults to
        ``{"sulphate": 1.0}`` when ``None``.

    Returns
    -------
    FullConfig
        A plain dict with keys ``"atmo_config"``, ``"geo_config"``,
        ``"spectral_config"``.

    Raises
    ------
    MissingVariableError
        If any of the required variables are absent from ``scene[band]``.
    """
    _REQUIRED = ["aot", "h", "rh", "href", "vza", "vaa", "sza", "saa"]
    ds = scene[band]
    missing = [v for v in _REQUIRED if v not in ds]
    if missing:
        raise MissingVariableError(
            f"Variables {missing!r} are missing from band {band!r}. "
            "Load the scene with a ProductLoader first."
        )

    aot: xr.DataArray = ds["aot"]
    h: xr.DataArray = ds["h"]
    if n_bins is not None:
        aot = aot.adjeff.digitize(n_bins=n_bins)
        h = h.adjeff.digitize(n_bins=n_bins)

    return make_full_config(
        bands=scene.bands,
        aot=aot,
        h=h,
        rh=ds["rh"],
        href=ds["href"],
        vza=ds["vza"],
        vaa=ds["vaa"],
        sza=ds["sza"],
        saa=ds["saa"],
        species=species,
    )


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

    Single entry point that internally calls :func:`make_atmo_config`,
    :func:`make_geo_config`, and :class:`~adjeff.atmosphere.SpectralConfig`.
    The returned dict has keys ``"atmo_config"``, ``"geo_config"``,
    ``"spectral_config"`` and can be unpacked directly with ``**cfg`` into
    :class:`~adjeff.modules.samplers.RadiativePipeline` and
    :class:`~adjeff.modules.samplers.RhoToaSymSampler`.

    Parameters
    ----------
    bands : list[SensorBand]
        Sensor bands to simulate.
    aot : float or list or DataArray
        Aerosol optical thickness (default 0.1).
    rh : float or list or DataArray
        Relative humidity [%] (default 50.0).
    h : float or list or DataArray
        Ground elevation [km] (default 0.0).
    href : float or list or DataArray
        Aerosol scale height [km] (default 2.0).
    species : dict[str, float] or None
        Aerosol species mix summing to 1.0 (default ``{"sulphate": 1.0}``).
    sza : float or list or DataArray
        Sun zenith angle [°] (default 30.0).
    vza : float or list or DataArray
        Viewing zenith angle [°] (default 0.0).
    saa : float or list or DataArray
        Sun azimuth angle [°] (default 120.0).
    vaa : float or list or DataArray
        Viewing azimuth angle [°] (default 120.0).
    sat_height : float
        Satellite altitude [km] (default 786.0).

    Returns
    -------
    FullConfig
        A plain ``dict`` with three typed entries.
    """
    return FullConfig(
        atmo_config=make_atmo_config(
            aot=aot, rh=rh, h=h, href=href, species=species
        ),
        geo_config=make_geo_config(
            sza=sza, vza=vza, saa=saa, vaa=vaa, sat_height=sat_height
        ),
        spectral_config=SpectralConfig.from_bands(bands),
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

M = TypeVar("M", bound=PSFConvModule)


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
    device : str
        PyTorch device (default ``"cuda"``).
    cache : CacheStore or None
        Optional cache backend (default ``None``).

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


# ---------------------------------------------------------------------------
# Forward pipeline
# ---------------------------------------------------------------------------


@overload
def run_forward_pipeline(
    scene: ImageDict,
    atmo_config: AtmoConfig,
    geo_config: GeoConfig,
    spectral_config: SpectralConfig,
    cache: CacheStore | None = ...,
    remove_rayleigh: bool = ...,
    afgl_type: str = ...,
    nr: int = ...,
    n_ph: int = ...,
    radiative_chunks: dict[str, int] | None = ...,
    stream_dims: dict[str, int] | None = ...,
) -> ImageDict: ...


@overload
def run_forward_pipeline(
    scene: list[ImageDict],
    atmo_config: AtmoConfig,
    geo_config: GeoConfig,
    spectral_config: SpectralConfig,
    cache: CacheStore | None = ...,
    remove_rayleigh: bool = ...,
    afgl_type: str = ...,
    nr: int = ...,
    n_ph: int = ...,
    radiative_chunks: dict[str, int] | None = ...,
    stream_dims: dict[str, int] | None = ...,
) -> list[ImageDict]: ...


def run_forward_pipeline(
    scene: ImageDict | list[ImageDict],
    atmo_config: AtmoConfig,
    geo_config: GeoConfig,
    spectral_config: SpectralConfig,
    cache: CacheStore | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    nr: int = 500,
    n_ph: int = int(1e5),
    radiative_chunks: dict[str, int] | None = None,
    stream_dims: dict[str, int] | None = None,
) -> ImageDict | list[ImageDict]:
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
    cache : CacheStore or None
        Shared cache forwarded to all modules.
    remove_rayleigh : bool
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    nr : int
        Radial sampling points for rho_toa (default 500).
    n_ph : int
        Photon count per sensor for rho_toa (default ``1e5``).
    radiative_chunks : dict[str, int] or None
        Chunk sizes for Smart-G calls inside
        :class:`~adjeff.modules.RadiativePipeline`,
        e.g. ``{"wl": 4}``. ``None`` disables chunking.
    stream_dims : dict[str, int] or None
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
        sweep_chunks=radiative_chunks,
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

    if isinstance(scene, list):
        return [pipeline(s) for s in scene]
    return pipeline(scene)


# ---------------------------------------------------------------------------
# Species / stage constants (used by new API functions)
# ---------------------------------------------------------------------------

_SPECIES_ATTR = "adjeff:species"
_DEFAULT_SPECIES: dict[str, float] = {"sulphate": 1.0}
_DEFAULT_LOSS = Loss(Metric.RMSE_RAD)
_DEFAULT_STAGES: list[AdamConfig | LBFGSConfig] = [
    AdamConfig(
        min_steps=5,
        max_steps=20,
        loss_relative_tolerance=1e-4,
        loss=_DEFAULT_LOSS,
        lr=1e-2,
    ),
    LBFGSConfig(
        min_steps=5,
        max_steps=30,
        loss_relative_tolerance=1e-6,
        loss=_DEFAULT_LOSS,
    ),
]


# ---------------------------------------------------------------------------
# Internal helpers (new API)
# ---------------------------------------------------------------------------


def _res_from_scene(scene: ImageDict, band: SensorBand) -> float:
    """Infer pixel size [km] from the y-coordinate spacing of *scene[band]*."""
    y: xr.DataArray = scene[band].coords["y"]
    return float(abs(float(y[1]) - float(y[0])))


def _configs_to_stages(
    stages: list[AdamConfig | LBFGSConfig],
) -> list[_ComboStage]:
    """Convert config objects to the corresponding Stage wrappers."""
    out: list[_ComboStage] = []
    for cfg in stages:
        if isinstance(cfg, AdamConfig):
            out.append(AdamStage(cfg))
        else:
            out.append(LBFGSStage(cfg))
    return out


def _to_scalar(v: xr.DataArray | float) -> float:
    """Extract a Python float from a scalar, 0-d, or 1-element DataArray."""
    if isinstance(v, xr.DataArray):
        return float(v.values.flat[0])
    return float(v)


# ---------------------------------------------------------------------------
# load_scene — generic loader with species persistence
# ---------------------------------------------------------------------------


def load_scene(
    loader: ProductLoader,
    *,
    compute_radiatives: bool = False,
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    cache: CacheStore | None = None,
    deduplicate_dims: list[str] | None = None,
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
        Bins for ``aot``/``h`` digitisation when *compute_radiatives* is
        ``True``.
    species : dict[str, float] or None, optional
        Override aerosol species mix.
    remove_rayleigh : bool, optional
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str, optional
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    cache : CacheStore or None, optional
        Shared on-disk cache (default ``None``).
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before Smart-G calls.

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
            deduplicate_dims=deduplicate_dims,
        )

    return scene


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


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
    deduplicate_dims: list[str] | None = None,
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
    mnt_path : Path or None
        Folder containing the DEM at 20 m resolution.  Must be provided;
        ``None`` raises :class:`~adjeff.exceptions.ConfigurationError`.
    href : float
        Aerosol scale height [km] (default ``2.0``).
    as_map : bool
        When ``True``, load 2-D atmospheric parameters as full spatial maps
        instead of spatially-averaged scalars (default ``False``).
    cache : CacheStore or None
        Optional on-disk cache shared between the loader and the radiative
        pipeline (default ``None``).
    compute_radiatives : bool
        When ``True``, run the radiative pipeline after loading (default
        ``False``).
    n_bins : int or None
        Number of bins used to digitise ``aot`` and ``h``, reducing the
        number of Smart-G runs.  Ignored when *compute_radiatives* is
        ``False``.
    remove_rayleigh : bool
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before running Smart-G.  Pass
        ``["x", "y"]`` when *as_map* is ``True`` (default ``None``).

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
        deduplicate_dims=deduplicate_dims,
    )


# ---------------------------------------------------------------------------
# Scene-based radiative pipeline
# ---------------------------------------------------------------------------


@overload
def run_radiatives_from_scene(
    scene: ImageDict,
    n_bins: int | None = ...,
    species: dict[str, float] | None = ...,
    remove_rayleigh: bool = ...,
    afgl_type: str = ...,
    cache: CacheStore | None = ...,
    deduplicate_dims: list[str] | None = ...,
) -> ImageDict: ...


@overload
def run_radiatives_from_scene(
    scene: list[ImageDict],
    n_bins: int | None = ...,
    species: dict[str, float] | None = ...,
    remove_rayleigh: bool = ...,
    afgl_type: str = ...,
    cache: CacheStore | None = ...,
    deduplicate_dims: list[str] | None = ...,
) -> list[ImageDict]: ...


def run_radiatives_from_scene(
    scene: ImageDict | list[ImageDict],
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
    remove_rayleigh: bool = False,
    afgl_type: str = "afgl_exp_h8km",
    cache: CacheStore | None = None,
    deduplicate_dims: list[str] | None = None,
) -> ImageDict | list[ImageDict]:
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
        If provided, ``aot`` and ``h`` are digitised to *n_bins* unique
        values before building the config, reducing the number of distinct
        Smart-G runs.
    species : dict[str, float] or None, optional
        Aerosol species mix summing to 1.0.  Defaults to
        ``{"sulphate": 1.0}`` when ``None``.
    remove_rayleigh : bool
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    cache : CacheStore or None
        Shared cache forwarded to all pipeline instances.
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before running Smart-G, reducing
        redundant simulations when ``aot`` and ``h`` are 2-D maps.  Pass
        ``["x", "y"]`` when the scene was loaded with ``as_map=True``
        (default ``None``).

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
            config = config_from_scene(
                scene=scene_band,
                band=band,
                n_bins=n_bins,
                species=species,
            )
            radiative = RadiativePipeline(
                atmo_config=config["atmo_config"],
                geo_config=config["geo_config"],
                spectral_config=config["spectral_config"],
                remove_rayleigh=remove_rayleigh,
                afgl_type=afgl_type,
                cache=cache,
                deduplicate_dims=deduplicate_dims,
            )
            scene_band = radiative(scene_band)
            s[band] = scene_band[band]
        return s

    if isinstance(scene, list):
        return [_run(s) for s in scene]
    return _run(scene)


# ---------------------------------------------------------------------------
# PSF sampling
# ---------------------------------------------------------------------------


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
    remove_rayleigh : bool
        Suppress Rayleigh scattering (default ``False``).
    afgl_type : str
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).
    n_ph : int
        Photon count per Smart-G run (default ``1e6``).
    cache : CacheStore or None
        Optional result cache.

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


# ---------------------------------------------------------------------------
# Optimizer shortcut
# ---------------------------------------------------------------------------


def optimize_adam_lbfgs(
    model: PSFConvModule,
    train_images: TrainingImages,
    loss: Loss,
    adam_config: AdamConfig | None = None,
    lbfgs_config: LBFGSConfig | None = None,
    device: str = "cuda",
    zarr_path: str | Path | None = None,
) -> xr.DataTree:
    """Optimize a model's PSF with an Adam warm-up followed by L-BFGS.

    Parameters
    ----------
    model : PSFConvModule
        Trainable model, holding live PSF modules.
    train_images : TrainingImages
        Collection of reference scenes.
    loss : Loss
        Loss function instance (e.g. ``Loss(Metric.RMSE_RAD)``).
        Used as default loss in *adam_config* and *lbfgs_config* when those
        are ``None``.
    adam_config : AdamConfig or None, optional
        Adam stage configuration.  When ``None``, defaults to
        ``AdamConfig(min_steps=5, max_steps=20,
        loss_relative_tolerance=1e-4, loss=loss, lr=1e-2)``.
    lbfgs_config : LBFGSConfig or None, optional
        L-BFGS stage configuration.  When ``None``, defaults to
        ``LBFGSConfig(min_steps=5, max_steps=30,
        loss_relative_tolerance=1e-6, loss=loss)``.
    device : str
        PyTorch device (default ``"cuda"``).
    zarr_path : str or Path or None, optional
        When provided, each band's stacked kernel is written to zarr as
        it is reconstructed and immediately freed from RAM.  The returned
        tree is backed by zarr on disk (lazy, minimal RAM footprint).
        When ``None`` (default), kernels are kept in memory.

    Returns
    -------
    xr.DataTree
        Frozen PSF tree with optimised kernels stacked over all atmospheric
        combos found in *train_images*.
    """
    if adam_config is None:
        adam_config = AdamConfig(
            min_steps=5,
            max_steps=20,
            loss_relative_tolerance=1e-4,
            loss=loss,
            lr=1e-2,
        )
    if lbfgs_config is None:
        lbfgs_config = LBFGSConfig(
            min_steps=5,
            max_steps=30,
            loss_relative_tolerance=1e-6,
            loss=loss,
        )
    optimizer = OptimizerPipeline(
        stages=[AdamStage(adam_config), LBFGSStage(lbfgs_config)],
        train_images=train_images,
        device=device,
    )
    return optimizer.run(model, zarr_path=zarr_path)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

_REQUIRED_VARS = ["aot", "h", "rh", "href", "vza", "vaa", "sza", "saa"]


def load_config(
    scene: ImageDict,
    band: SensorBand,
    *,
    aggregate: bool = False,
    n_bins: int | None = None,
    species: dict[str, float] | None = None,
) -> FullConfig:
    """Build a :class:`FullConfig` from a scene with optional aggregation.

    Extends :func:`config_from_scene` with:

    - **Species recovery** from ``scene[band].attrs["adjeff:species"]``
      (written by :func:`load_scene`) when *species* is ``None``.
    - **Spatial aggregation** (``aggregate=True``) to reduce all fields to
      scalars via ``.mean()``, useful when a single representative
      atmospheric state is needed.

    Parameters
    ----------
    scene : ImageDict
        Scene produced by :func:`load_scene` or another loader.
    band : SensorBand
        Band from which to read the parameters.
    aggregate : bool, optional
        Reduce every field to its spatial mean.  Incompatible with *n_bins*.
    n_bins : int or None, optional
        Digitise ``aot`` and ``h`` to *n_bins* unique values.  Incompatible
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
        raise ConfigurationError(
            "`aggregate` and `n_bins` are mutually exclusive."
        )

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


# ---------------------------------------------------------------------------
# fit_psf
# ---------------------------------------------------------------------------


def fit_psf(
    scene: ImageDict,
    bands: list[SensorBand],
    psf_type: type[PSFModule],
    init_parameters: dict[str, float],
    *,
    model_cls: type[PSFConvModule] = Unif2Surface,
    target_var: str | None = None,
    train_radii: list[float] | None = None,
    stages: list[AdamConfig | LBFGSConfig] | None = None,
    res_km: float | None = None,
    n_train: int = 1999,
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
    n_train : int, optional
        PSF grid side in pixels for training (default 1999, must be odd).
    cache : CacheStore or None, optional
        Shared cache forwarded to the forward pipeline.
    device : str, optional
        PyTorch device (default ``"cuda"``).

    Returns
    -------
    xr.DataTree
        Frozen PSF tree with one optimised kernel per band.
    """
    _target_var: str = target_var or model_cls.output_vars[0]
    _res_km: float = res_km or _res_from_scene(scene, bands[0])
    _stages = _DEFAULT_STAGES if stages is None else stages
    _radii = train_radii or [1.0, 5.0, 50.0]

    cfg = load_config(scene, bands[0], aggregate=True)

    disk_scenes = [
        disk_image_dict(
            radius=r,
            res_km=_res_km,
            rho_min=0.0,
            rho_max=0.5,
            bands=bands,
            var=_target_var,
            n=n_train,
        )
        for r in _radii
    ]
    disk_scenes = run_forward_pipeline(disk_scenes, **cfg, cache=cache)
    train_images = TrainingImages(
        images=disk_scenes,
        weights=[1.0] * len(disk_scenes),
    )

    model = model_cls(
        psfs=_build_psfs(psf_type, bands, _res_km, n_train, init_parameters),
        device=device,
        cache=cache,
    )

    optimizer = OptimizerPipeline(
        stages=_configs_to_stages(_stages),
        train_images=train_images,
        device=device,
    )
    return optimizer.run(model)


# ---------------------------------------------------------------------------
# apply_psf
# ---------------------------------------------------------------------------


def apply_psf(
    scene: ImageDict,
    psf_dict: xr.DataTree,
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

    - **Direct** (``n=None``): wraps *psf_dict* in a model and applies it
      as-is.  The kernel size equals the one used during training.
    - **Rebuild** (``n`` provided): reconstructs the PSF on an *n*×*n* grid
      from the stored parameters, then applies it.  Requires *psf_type*.

    Parameters
    ----------
    scene : ImageDict
        Scene containing the variables required by *model_cls*.
    psf_dict : xr.DataTree
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
        raise ConfigurationError(
            "`psf_type` is required when `n` is provided."
        )

    if n is not None and psf_type is not None:
        params: dict[str, float] = {
            name: _to_scalar(value)
            for name, value in psf_params(psf_dict, band).items()
        }
        _res_km: float = res_km or _res_from_scene(scene, band)
        model = model_cls(
            psfs=_build_psfs(psf_type, [band], _res_km, n, params),
            device=device,
        )
    else:
        model = model_cls(kernels=psf_dict, device=device)

    model.eval()
    return model(scene)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# sample_psf_atm_from_scene
# ---------------------------------------------------------------------------


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
        Optional result cache.

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
