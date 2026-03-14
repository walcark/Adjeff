"""Methods to instanciate Smart-G surface objects from a ground image.

Ground images in adjeff are either artibrary (real image, complex scene)
or analytical (gaussian, disk) shapes. The following methods instanciate
both the Smart-G `Environment` and `Surface` from this knowledge.
"""

from smartg.smartg import (  # type: ignore[import-untyped]
    Entity,
    Environment,
    LambSurface,
)


class SurfaceFactory:
    """Class that computes Smart-G surface-related objects."""

    def __init__(self) -> None:
        pass

    @property
    def entity(self) -> Entity:
        """Return an entity object based on input image coordinates."""
        return Entity()

    @property
    def surface(self) -> LambSurface:
        """Return an Lambertian Surface object based on the input image."""
        return LambSurface()

    @property
    def environment(self) -> Environment:
        """Return an Environment object based on the input image."""
        return Environment()

    # TODO: chaque méthode appelle son pendant _entity, _surface, _environment
    # TODO: qui ne travaille que sur le dico {coords, rho_min, rho_max, sigma}
    # TODO: et qui ont toutes un cache.
