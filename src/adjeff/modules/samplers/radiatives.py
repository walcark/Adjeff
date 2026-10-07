"""Pipeline sampling the six 5S terms.

Classes
-------
    RadiativePipeline
        The six samplers in a row, with the RTLS variants on ``rtls``.
"""

from typing import Any

from adjeff.atmosphere import AtmoConfig, GeoConfig, SpectralConfig
from adjeff.utils import CacheStore

from ..pipeline import Pipeline
from ..scene_module import SceneModule
from .rho_atm import RhoAtmSampler
from .sph_alb import SphAlbSampler
from .sph_alb_brdf import SphAlbBrdfSampler
from .tdif_down import TdifDownSampler
from .tdif_up import TdifUpSampler
from .tdif_up_brdf import TdifUpBrdfSampler
from .tdir_down import TdirDownSampler
from .tdir_up import TdirUpSampler


class RadiativePipeline(Pipeline):
    """The six 5S samplers in a row, on a CUDA GPU.

    Order: ``tdir_down, tdir_up, tdif_down, rho_atm, sph_alb, tdif_up``.

    Parameters
    ----------
    atmo_config, geo_config, spectral_config : configs
        Atmosphere, geometry and bands.
    remove_rayleigh : bool
        Suppress Rayleigh scattering.
    afgl_type : str, optional
        AFGL atmosphere profile.
    n_ph_sph_alb, n_ph_rho_atm, n_ph_tdif_up, n_ph_tdif_down : int, optional
        Photons per call of each Monte-Carlo sampler.
    cache, batch_size, dedup : optional
        Passed to every sampler.
    rename : dict[str, str] or None, optional
        Variables written, split among the samplers by role; two renames
        compare two surface models in one scene.
    rtls : tuple[float, float, float] or None, optional
        RTLS weights ``(k0, k1p, k2p)``.  Given, ``sph_alb`` and
        ``tdif_up`` come from their BRDF variants, which read the
        downward terms sampled before them; the other four do not
        depend on the surface.
    """

    def __init__(
        self,
        atmo_config: AtmoConfig,
        geo_config: GeoConfig,
        spectral_config: SpectralConfig,
        remove_rayleigh: bool,
        afgl_type: str = "afgl_exp_h8km",
        n_ph_sph_alb: int = int(2e7),
        n_ph_rho_atm: int = int(2e7),
        n_ph_tdif_up: int = int(3e7),
        n_ph_tdif_down: int = int(3e7),
        cache: CacheStore | None = None,
        batch_size: int = 64,
        dedup: bool = False,
        rtls: tuple[float, float, float] | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        common: dict[str, Any] = dict(
            atmo_config=atmo_config,
            geo_config=geo_config,
            spectral_config=spectral_config,
            remove_rayleigh=remove_rayleigh,
            afgl_type=afgl_type,
            cache=cache,
            batch_size=batch_size,
            dedup=dedup,
        )

        def only(cls: type[SceneModule]) -> dict[str, str]:
            """Return the part of *rename* naming a role *cls* declares.

            Each module knows a different set of roles, and one is
            rejected outright by a name it does not have.  A pipeline
            rename is written for the chain, not for one module, so it
            is split rather than forwarded whole.
            """
            roles = {
                *cls._required_vars,
                *cls._output_vars,
                *cls._optional_vars,
            }
            return {k: v for k, v in (rename or {}).items() if k in roles}

        # The two surface-dependent terms are replaced, not added: they
        # write the same slots, and they are the two most expensive
        # samplers of the six.  Their BRDF variants read the downward
        # terms produced above them, hence the position.
        if rtls is None:
            surface_modules: list[Any] = [
                SphAlbSampler(
                    atmo_config=atmo_config,
                    spectral_config=spectral_config,
                    remove_rayleigh=remove_rayleigh,
                    afgl_type=afgl_type,
                    n_ph=n_ph_sph_alb,
                    cache=cache,
                    batch_size=batch_size,
                    dedup=dedup,
                    rename=only(SphAlbSampler),
                ),
                TdifUpSampler(**common, n_ph=n_ph_tdif_up, rename=only(TdifUpSampler)),
            ]
        else:
            k0, k1p, k2p = rtls
            surface_modules = [
                SphAlbBrdfSampler(**common, k0=k0, k1p=k1p, k2p=k2p, n_ph=n_ph_sph_alb),
                TdifUpBrdfSampler(
                    **common,
                    k0=k0,
                    k1p=k1p,
                    k2p=k2p,
                    n_ph=n_ph_tdif_up,
                    n_ph_tdif_down=n_ph_tdif_down,
                    rename=only(TdifUpBrdfSampler),
                ),
            ]
        super().__init__(
            [
                TdirDownSampler(**common, rename=only(TdirDownSampler)),
                TdirUpSampler(**common, rename=only(TdirUpSampler)),
                TdifDownSampler(
                    **common, n_ph=n_ph_tdif_down, rename=only(TdifDownSampler)
                ),
                RhoAtmSampler(**common, n_ph=n_ph_rho_atm, rename=only(RhoAtmSampler)),
                *surface_modules,
            ]
        )
