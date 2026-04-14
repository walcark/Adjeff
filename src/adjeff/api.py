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

from typing import TypedDict, TypeVar, overload

import numpy as np
import xarray as xr

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.core import (
    ImageDict,  # noqa: E402
    PSFDict,
    PSFGrid,
    SensorBand,
    init_psf_dict,
)
from adjeff.core._psf import PSFModule
from adjeff.exceptions import MissingVariableError
from adjeff.modules.classic.toa_to_unif import Toa2Unif
from adjeff.modules.models.psf_conv_module import PSFConvModule
from adjeff.modules.samplers.radiatives import RadiativePipeline
from adjeff.modules.samplers.rho_toa import SmartgSampler_Rho_toa_sym
from adjeff.optim import Loss, OptimizerPipeline, TrainingImages
from adjeff.optim.adam_optimizer import AdamConfig, _AdamStage
from adjeff.optim.lbfgs_optimizer import LBFGSConfig, _LBFGSStage
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
    :class:`~adjeff.modules.samplers.SmartgSampler_Rho_toa_sym`, so the dict
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
    :class:`~adjeff.modules.samplers.SmartgSampler_Rho_toa_sym`.

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
) -> ImageDict | list[ImageDict]:
    """Run the full forward pipeline: radiatives → rho_toa → rho_unif.

    Chains :class:`~adjeff.modules.samplers.RadiativePipeline`,
    :class:`~adjeff.modules.samplers.SmartgSampler_Rho_toa_sym`, and
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

    Returns
    -------
    ImageDict or list[ImageDict]
        Same type as *scene*, enriched with ``rho_toa``, radiative
        quantities, and ``rho_unif``.
    """
    common = dict(
        atmo_config=atmo_config,
        geo_config=geo_config,
        spectral_config=spectral_config,
        remove_rayleigh=remove_rayleigh,
        afgl_type=afgl_type,
        cache=cache,
    )
    radiative = RadiativePipeline(**common)  # type: ignore[arg-type]
    rho_toa = SmartgSampler_Rho_toa_sym(**common, nr=nr, n_ph=n_ph)  # type: ignore[arg-type]
    toa2unif = Toa2Unif()

    def _run(s: ImageDict) -> ImageDict:
        return toa2unif(rho_toa(radiative(s)))

    if isinstance(scene, list):
        return [_run(s) for s in scene]
    return _run(scene)


# ---------------------------------------------------------------------------
# Optimizer shortcut
# ---------------------------------------------------------------------------


def optimize_adam_lbfgs(
    model: PSFConvModule,
    train_images: TrainingImages,
    loss: Loss,
    adam_min_steps: int = 5,
    adam_max_steps: int = 20,
    adam_lr: float = 1e-2,
    adam_tolerance: float = 1e-4,
    lbfgs_min_steps: int = 5,
    lbfgs_max_steps: int = 30,
    lbfgs_tolerance: float = 1e-6,
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
    adam_min_steps : int
        Minimum Adam steps before early stopping (default 5).
    adam_max_steps : int
        Maximum Adam steps (default 20).
    adam_lr : float
        Adam learning rate (default ``1e-2``).
    adam_tolerance : float
        Relative loss tolerance for Adam early stopping (default ``1e-4``).
    lbfgs_min_steps : int
        Minimum L-BFGS steps before early stopping (default 5).
    lbfgs_max_steps : int
        Maximum L-BFGS steps (default 30).
    lbfgs_tolerance : float
        Relative loss tolerance for L-BFGS early stopping (default ``1e-6``).
    device : str
        PyTorch device (default ``"cuda"``).

    Returns
    -------
    PSFDict
        Frozen PSFDict with optimised kernels stacked over all atmospheric
        combos found in *train_images*.
    """
    optimizer = OptimizerPipeline(
        stages=[
            _AdamStage(
                AdamConfig(
                    min_steps=adam_min_steps,
                    max_steps=adam_max_steps,
                    loss_relative_tolerance=adam_tolerance,
                    loss=loss,
                    lr=adam_lr,
                )
            ),
            _LBFGSStage(
                LBFGSConfig(
                    min_steps=lbfgs_min_steps,
                    max_steps=lbfgs_max_steps,
                    loss_relative_tolerance=lbfgs_tolerance,
                    loss=loss,
                )
            ),
        ],
        train_images=train_images,
        device=device,
    )
    return optimizer.run(model)
