"""High-level convenience API for the Adjeff library.

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
from typing import TypedDict, TypeVar, overload

import numpy as np
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import (
    ImageDict,  # noqa: E402
    PSFDict,
    PSFGrid,
    SensorBand,
    gaussian_image_dict,
    init_psf_dict,
)
from adjeff.core._psf import PSFModule
from adjeff.exceptions import MissingVariableError
from adjeff.modules.classic.toa_to_unif import Toa2Unif
from adjeff.modules.loaders.maja_loader import MajaLoader
from adjeff.modules.models.psf_conv_module import PSFConvModule
from adjeff.modules.samplers.psf_atm import PsfAtmSampler
from adjeff.modules.samplers.radiatives import RadiativePipeline
from adjeff.modules.samplers.rho_toa_sym import RhoToaSymSampler
from adjeff.optim import Loss, OptimizerPipeline, TrainingImages
from adjeff.optim.adam_optimizer import AdamConfig, AdamStage
from adjeff.optim.lbfgs_optimizer import LBFGSConfig, LBFGSStage
from adjeff.utils import CacheStore

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

_Scalar = float | list[float] | xr.DataArray


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

    Creates a :class:`~adjeff.core.PSFGrid` and a trainable
    :class:`~adjeff.core.PSFDict` for each band, then constructs the model.

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
    grids: dict[SensorBand, PSFGrid] = {
        band: PSFGrid(res=res_km, n=n) for band in bands
    }
    psf_dict: PSFDict = init_psf_dict(
        grids=grids,
        model=psf_type,
        init_parameters=init_parameters,
    )
    return model_cls(psf_dict=psf_dict, device=device, cache=cache)


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
        Chunk sizes forwarded to :class:`~adjeff.modules.RadiativePipeline`,
        e.g. ``{"wl": 4, "aot": 3}``. ``None`` disables chunking.

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
        chunks=radiative_chunks,
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
    toa2unif = Toa2Unif()

    def _run(s: ImageDict) -> ImageDict:
        return toa2unif(rho_toa(radiative(s)))

    if isinstance(scene, list):
        return [_run(s) for s in scene]
    return _run(scene)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


# TODO: make the function dependant on the loader if another loader is
# TODO: introduced in the future.
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
    """Load a MAJA L2A product into an :class:`~adjeff.core.ImageDict`.

    Wraps :class:`~adjeff.modules.loaders.MajaLoader` and optionally runs
    :func:`run_radiatives_from_scene` in a single call.

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
        When ``True``, run the radiative pipeline after loading, enriching
        the scene with ``tdir_down``, ``tdif_down``, ``tdir_up``,
        ``tdif_up``, ``rho_atm``, and ``sph_alb`` (default ``False``).
    n_bins : int or None
        Number of bins used to digitise ``aot`` and ``h``, reducing the
        number of Smart-G runs.  Ignored when *compute_radiatives* is
        ``False``.
    remove_rayleigh : bool
        Suppress Rayleigh scattering in the radiative pipeline
        (default ``False``).  Ignored when *compute_radiatives* is ``False``.
    afgl_type : str
        AFGL atmosphere profile (default ``"afgl_exp_h8km"``).  Ignored
        when *compute_radiatives* is ``False``.
    deduplicate_dims : list[str] or None, optional
        Spatial dimensions to deduplicate before running Smart-G.  Pass
        ``["x", "y"]`` when *as_map* is ``True`` to avoid running one
        simulation per pixel (default ``None``).

    Returns
    -------
    ImageDict
        Scene with ``rho_s`` and atmospheric/geometric variables, plus
        radiative quantities when *compute_radiatives* is ``True``.
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
    scene = loader.forward()
    if compute_radiatives:
        scene = run_radiatives_from_scene(
            scene,
            n_bins=n_bins,
            species=loader.species(),
            remove_rayleigh=remove_rayleigh,
            afgl_type=afgl_type,
            cache=cache,
            deduplicate_dims=deduplicate_dims,
        )
    return scene


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
) -> PSFDict:
    """Sample the atmospheric PSF and return a frozen :class:`PSFDict`.

    Internally builds a constant input scene to carry the spatial grid
    (only ``res`` and ``n`` matter to the sampler — the reflectance values
    are irrelevant), runs
    :class:`~adjeff.modules.samplers.PsfAtmSampler`, then wraps the
    resulting ``psf_atm`` DataArrays into a :class:`~adjeff.core.PSFDict`.

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
    PSFDict
        Frozen PSFDict with one ``kernel`` DataArray per band.
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
    sampler = PsfAtmSampler(
        atmo_config=atmo_config,
        geo_config=geo_config,
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        n_ph=n_ph,
        cache=cache,
    )
    out = sampler(scene)
    return PSFDict.from_kernels(
        {band: out[band]["psf_atm"] for band in out.bands}
    )


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
) -> PSFDict:
    """Optimize a model's PSF with an Adam warm-up followed by L-BFGS.

    Parameters
    ----------
    model : PSFConvModule
        Trainable model (must hold a trainable :class:`~adjeff.core.PSFDict`).
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

    Returns
    -------
    PSFDict
        Frozen PSFDict with optimised kernels stacked over all atmospheric
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
    return optimizer.run(model)
