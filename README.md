# Adjeff

<p align="center">
  <img src="https://github.com/walcark/Adjeff/actions/workflows/ci.yml/badge.svg">
  <a href="https://pixi.sh"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://pypi.org/project/adjeff/"><img src="https://img.shields.io/pypi/v/Adjeff.svg"></a>
  <img src="https://img.shields.io/github/license/walcark/Adjeff">
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
</p>

Adjeff simulates and corrects the **adjacency effect** in Sentinel-2
imagery: the light that the atmosphere scatters from neighbouring pixels
into the one observed.

- **Forward simulation**: top-of-atmosphere reflectance of any surface,
  by GPU Monte Carlo radiative transfer ([Smart-G](https://github.com/hygeos/smartg)).
- **PSF learning**: the atmospheric point spread function $P$, fitted
  per band and atmospheric state as an analytical kernel.
- **Correction**: surface reflectance recovered from an observed scene,
  with the learned PSF.

$$\rho_{toa} = \rho_{atm} + T^\downarrow \frac{T^\uparrow_{dir}\,\rho_s + T^\uparrow_{dif}\,(\rho_s \ast P)}{1 - s\,(\rho_s \ast P)}$$

## Installation

Adjeff is managed with [pixi](https://pixi.sh): Smart-G and part of its
dependencies come from conda-forge.

```bash
git clone https://github.com/walcark/Adjeff.git && cd Adjeff
pixi install -e gpu          # or -e cpu, without Smart-G simulations
export SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata
```

Simulations need a CUDA 12.6 GPU and the Smart-G auxiliary data.
Analysis, PSF models and the 5S formulas run on CPU.

## Example

```python
import adjeff
from adjeff.core import KingPSF, S2Band, disk_image_dict
from adjeff.modules.models import Unif2Surface
from adjeff.optim import TrainingImages

bands = [S2Band.B03]
cfg = adjeff.make_full_config(bands, aot=0.2, rh=50.0, h=0.0, sza=30.0, vza=5.0)

# Bright disks of 1, 5 and 50 km on a dark ground, 120 m pixels.
disks = [
    disk_image_dict(radius=r, res_km=0.12, rho_max=0.5, bands=bands, n=1999)
    for r in (1.0, 5.0, 50.0)
]

# Simulate their radiative terms, rho_toa and rho_unif (GPU).
disks = adjeff.run_forward_pipeline(disks, **cfg)

# Fit the King PSF that recovers rho_s from rho_unif.
model = adjeff.make_model(
    Unif2Surface, KingPSF, bands, res_km=0.12, n=1999,
    init_parameters={"sigma": 0.5, "gamma": 2.0},
)
tree = adjeff.fit(model, TrainingImages(disks))

# Correct a scene carrying rho_unif and the radiative terms.
corrected = adjeff.apply_psf(disks[0], tree, S2Band.B03)
```

`adjeff.fit_psf(scene, bands, KingPSF, ...)` does the same in one call,
at the mean atmosphere of a loaded product.

## Documentation

| | |
|---|---|
| [**User guide**](https://github.com/walcark/Adjeff/blob/main/docs/guide.md) | Concepts, configuration, samplers, PSF, fitting, cache, logging |
| [**Notebooks**](https://github.com/walcark/Adjeff/tree/main/notebooks) | Six worked examples, from a synthetic scene to a learned PSF |
| [**API reference**](https://github.com/walcark/Adjeff/tree/main/docs/source/api) | Generated from the docstrings (`pixi run build-doc`) |
| [**Changelog**](https://github.com/walcark/Adjeff/blob/main/CHANGELOG.md) | What changed, and why |
| [**Roadmap**](https://github.com/walcark/Adjeff/blob/main/docs/roadmap.md) | Open questions and planned work |

## Development

```bash
pixi install -e dev
pixi run -e dev all                    # format, lint, mypy --strict, tests
pixi run -e dev-gpu test-integration   # Smart-G tests, on GPU
```

## License

Apache-2.0.
