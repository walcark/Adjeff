"""Tests for adjeff.analysis: encircled energy, FWHM and MTF.

The encircled energy existed twice in the package, once in the radial
accessor and once rewritten inside the landscape scan for speed, and
neither was tested against a profile whose answer is known in closed
form.  A Gaussian is that profile.
"""

import math

import numpy as np
import pytest
import xarray as xr

from adjeff.analysis import (
    encircled_energy,
    encircled_radii,
    encircled_radius,
    fwhm,
    mtf,
)
from adjeff.core import GaussPSF, KingPSF, PSFGrid, S2Band

BAND = S2Band.B03
RES_KM = 0.05
N = 201


def gaussian(sigma: float) -> xr.DataArray:
    """Return a normalised Gaussian kernel of width *sigma* [km]."""
    return GaussPSF(PSFGrid(RES_KM, N), BAND, sigma=sigma).to_dataarray()


# --- encircled energy ---


def test_encircled_energy_rises_from_zero_to_one():
    """A cumulated fraction is monotone and ends at one."""
    cdf = encircled_energy(gaussian(0.3))

    assert cdf.dims == ("r",)
    assert float(cdf[0]) >= 0.0
    assert float(cdf[-1]) == pytest.approx(1.0, abs=1e-6)
    assert bool((np.diff(cdf.values) >= -1e-7).all())


def test_encircled_radius_matches_the_closed_form_for_a_gaussian():
    """A 2-D Gaussian encircles a fraction f at r = sigma * sqrt(-2 ln(1-f)).

    This is the check neither of the two previous implementations had:
    a profile whose answer is known without running the code.
    """
    sigma = 0.4
    kernel = gaussian(sigma)

    for fraction in (0.5, 0.9):
        expected = sigma * math.sqrt(-2.0 * math.log(1.0 - fraction))
        assert encircled_radius(kernel, fraction) == pytest.approx(expected, rel=0.05)


def test_encircled_radius_grows_with_the_fraction():
    """More energy, larger circle."""
    kernel = gaussian(0.3)

    assert encircled_radius(kernel, 0.5) < encircled_radius(kernel, 0.9)


def test_encircled_radius_rejects_a_non_square_kernel():
    """A kernel is square; anything else is a caller mistake."""
    strip = xr.DataArray(
        np.ones((3, 5)),
        dims=["y_psf", "x_psf"],
        coords={"y_psf": np.arange(3.0), "x_psf": np.arange(5.0)},
    )

    with pytest.raises(ValueError, match="square"):
        encircled_radius(strip)


def test_encircled_radii_batches_over_one_grid():
    """The landscape scan asks for many kernels on a shared grid."""
    import torch

    kernels = [
        torch.from_numpy(gaussian(s).values.astype(np.float32)) for s in (0.2, 0.4, 0.8)
    ]

    radii = encircled_radii(kernels, n=N, res=RES_KM, fractions=[0.5])

    assert set(radii) == {"EE50%"}
    assert (np.diff(radii["EE50%"]) > 0).all()


def test_encircled_radii_of_nothing_is_empty():
    """An empty scan returns empty arrays, one per fraction."""
    radii = encircled_radii([], n=0, res=1.0, fractions=[0.5, 0.9])

    assert radii["EE50%"].shape == (0,)
    assert radii["EE90%"].shape == (0,)


# --- FWHM ---


def test_fwhm_matches_the_closed_form_for_a_gaussian():
    """A Gaussian of width sigma has FWHM = 2 sqrt(2 ln 2) sigma."""
    sigma = 0.4

    expected = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma

    assert fwhm(gaussian(sigma)) == pytest.approx(expected, rel=0.05)


def test_fwhm_grows_with_the_kernel_width():
    """Wider kernel, wider half maximum."""
    assert fwhm(gaussian(0.2)) < fwhm(gaussian(0.6))


# --- MTF ---


def test_mtf_starts_at_one_and_decreases():
    """Contrast is preserved at zero frequency and lost above it."""
    curve = mtf(gaussian(0.3))

    assert curve.dims == ("f",)
    assert float(curve[0]) == pytest.approx(1.0, rel=1e-3)
    finite = curve.values[np.isfinite(curve.values)]
    assert finite[-1] < finite[0]


def test_mtf_of_a_wider_psf_falls_faster():
    """A wider kernel destroys contrast at lower frequencies.

    This is the whole point of reporting an MTF: it turns a spatial
    width into the frequency band an instrument still transmits.
    """
    narrow = mtf(gaussian(0.2))
    wide = mtf(gaussian(0.6))

    mid = len(narrow.values) // 8
    assert wide.values[mid] < narrow.values[mid]


def test_mtf_stops_at_nyquist():
    """Beyond half a cycle per pixel there is nothing left to measure."""
    curve = mtf(gaussian(0.3))

    assert float(curve.coords["f"].max()) <= 0.5 / RES_KM


def test_king_tail_shows_up_in_the_outer_radius():
    """A shallower power law puts more energy far from the centre."""
    grid = PSFGrid(RES_KM, N)
    shallow = KingPSF(grid, BAND, sigma=0.3, gamma=1.05).to_dataarray()
    steep = KingPSF(grid, BAND, sigma=0.3, gamma=4.0).to_dataarray()

    assert encircled_radius(shallow, 0.9) > encircled_radius(steep, 0.9)


def _king_radius(sigma: float, gamma: float, fraction: float) -> float:
    """Return the radius enclosing *fraction* of a King profile, in closed form.

    Integrating ``(1 + r**2 / a)**-gamma`` over the disc, with
    ``a = 2 * gamma * sigma**2``, gives ``E(r) = pi * a * (1 - (1 +
    r**2 / a)**(1 - gamma)) / (gamma - 1)``.  The kernel is measured on a
    finite grid, so the fraction is taken of the energy inside its
    half-diagonal, not of the whole plane.
    """
    a = 2.0 * gamma * sigma**2

    def energy(r: float) -> float:
        return float(1.0 - (1.0 + r**2 / a) ** (1.0 - gamma))

    r_max = (N // 2) * RES_KM * math.sqrt(2.0)
    target = fraction * energy(r_max)
    return float(math.sqrt(a * ((1.0 - target) ** (1.0 / (1.0 - gamma)) - 1.0)))


@pytest.mark.parametrize("gamma", [1.2, 2.0, 4.0])
@pytest.mark.parametrize("fraction", [0.10, 0.50, 0.90, 0.99])
def test_encircled_radius_matches_the_closed_form_for_a_king(gamma, fraction):
    """The measured radius must land on the one the King integral gives.

    Reading the cumulated energy at the bin centre rather than at the
    outer edge it belongs to, and snapping to that centre instead of
    interpolating, used to cost up to seven percent here.
    """
    sigma = 0.3
    kernel = KingPSF(PSFGrid(RES_KM, N), BAND, sigma=sigma, gamma=gamma)
    measured = encircled_radius(kernel.to_dataarray(), fraction)

    expected = _king_radius(sigma, gamma, fraction)
    assert measured == pytest.approx(expected, rel=0.01)


def test_encircled_energy_starts_at_zero_and_is_read_at_the_edges():
    """No energy is enclosed by a radius of zero, and the curve closes."""
    curve = encircled_energy(gaussian(0.3))

    assert float(curve.coords["r"][0]) == 0.0
    assert float(curve[0]) == 0.0
    assert float(curve[-1]) == pytest.approx(1.0)
    assert curve.sizes["r"] == curve.coords["r"].size


# --- error metrics on DataArrays ---


def _field(values, n=9):
    """Return a square field with the coordinates the metrics need."""
    coord = np.linspace(-1.0, 1.0, n)
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=["y", "x"],
        coords={"y": coord, "x": coord},
    )


def test_rmse_of_a_constant_offset_is_that_offset():
    """Scaled on the reference, an offset of 0.1 over a peak of 1 reads 0.1."""
    from adjeff.analysis import rmse

    truth = _field(np.ones((9, 9)))

    assert rmse(truth + 0.1, truth) == pytest.approx(0.1, rel=1e-4)


def test_rmse_rejects_a_shape_mismatch():
    """A leftover singleton dimension must raise, not broadcast.

    A silent broadcast returns a number that means nothing, which is
    exactly what the article's own helper had to guard against.
    """
    from adjeff.analysis import rmse

    truth = _field(np.ones((9, 9)))

    with pytest.raises(ValueError, match="shape mismatch"):
        rmse(truth.expand_dims("aot"), truth)


def test_rmse_over_a_disc_ignores_what_lies_outside():
    """A radius keeps a disc, and nothing beyond it counts."""
    from adjeff.analysis import rmse

    truth = _field(np.ones((9, 9)))
    spoiled = truth.copy()
    spoiled.values[0, 0] = 10.0

    inside = rmse(spoiled, truth, mask=0.5)
    everywhere = rmse(spoiled, truth)

    assert inside == pytest.approx(0.0, abs=1e-6)
    assert everywhere > 0.05


def test_radial_weighting_favours_the_centre():
    """Equal weight per radius means the few central pixels count more."""
    from adjeff.analysis import rmse

    truth = _field(np.ones((9, 9)))
    centre_off = truth.copy()
    centre_off.values[4, 4] = 1.5

    assert rmse(centre_off, truth, radial=True) > rmse(centre_off, truth)


def test_bias_keeps_the_sign_that_rmse_loses():
    """A systematic offset reads negative when the estimate is low."""
    from adjeff.analysis import bias, rmse

    truth = _field(np.ones((9, 9)))

    assert bias(truth - 0.2, truth) == pytest.approx(-0.2, rel=1e-6)
    assert rmse(truth - 0.2, truth) > 0.0


def test_mae_is_smaller_than_rmse_on_a_spiky_error():
    """The square is what makes a few large errors dominate."""
    from adjeff.analysis import mae, rmse

    truth = _field(np.zeros((9, 9)))
    spiky = truth.copy()
    spiky.values[4, 4] = 1.0

    assert mae(spiky, truth) < rmse(spiky, truth)
