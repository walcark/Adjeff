# Plan d'amélioration Sphinx

Objectif : passer d'une doc plate (un seul toctree) à une navigation
tabulée style pyGeodes, avec intégration des notebooks comme section
"Examples".

---

## 1. État actuel

```
docs/source/
├── index.rst          ← page unique avec automodule + toctree flat
├── atmosphere.rst
├── core.rst
├── modules.rst
├── utils.rst
└── _static/custom.css
```

Thème : `pydata_sphinx_theme`.  
Extensions actives : `autodoc`, `napoleon`, `viewcode`, `intersphinx`,
`autosummary`.

Notebooks dans `notebooks/` (6 au total) :
- `01-create-and-display-image.ipynb`
- `02-atmospheric-configuration.ipynb`
- `03-compute-radiative-quantities.ipynb`
- `04-simulate-rho-toa.ipynb`
- `05-compute-2d-radiative-from-maja-output.ipynb`
- `06-learn-psf.ipynb`

---

## 2. Cible : navigation tabulée

```
[ User Guide ]  [ Examples ]  [ API Reference ]  [ Development ]
```

Chaque onglet = un sous-dossier dans `docs/source/` avec son propre
`index.rst`. pydata-sphinx-theme crée automatiquement les onglets à
partir des entrées de premier niveau du toctree racine.

---

## 3. Nouvelle structure de fichiers

```
docs/source/
├── index.rst                  ← toctree racine (4 entrées = 4 onglets)
│
├── user_guide/
│   └── index.rst              ← concepts clés, installation, démarrage rapide
│       ├── installation.rst
│       ├── quickstart.rst
│       ├── pipeline.rst       ← ImageDict → SceneModule → Pipeline
│       ├── atmosphere.rst     ← AtmoConfig, GeoConfig, SpectralConfig
│       └── psf.rst            ← PSFGrid, PSFModule, PSFDict
│
├── examples/
│   └── index.rst              ← toctree vers les notebooks
│       ├── 01-create-and-display-image.ipynb  (lien symlink ou nbsphinx-link)
│       ├── 02-atmospheric-configuration.ipynb
│       ├── 03-compute-radiative-quantities.ipynb
│       ├── 04-simulate-rho-toa.ipynb
│       ├── 05-compute-2d-radiative-from-maja-output.ipynb
│       └── 06-learn-psf.ipynb
│
├── api/
│   ├── index.rst              ← toctree vers les pages API
│   ├── core.rst               ← (déplacer depuis source/core.rst)
│   ├── atmosphere.rst
│   ├── modules.rst
│   ├── utils.rst
│   ├── sweep.rst              ← nouveau (SweepBundle, UniqueIndex)
│   ├── optim.rst              ← nouveau (Loss, LBFGSOptimizer, etc.)
│   └── exceptions.rst         ← nouveau (hiérarchie d'exceptions)
│
├── development/
│   └── index.rst
│       ├── contributing.rst   ← conventions, workflow, CLAUDE.md
│       └── changelog.rst
│
└── _static/custom.css
```

---

## 4. Installation des extensions nécessaires

### 4.1 Notebooks : `nbsphinx`

```toml
# pyproject.toml — mettre à jour install-sphinx
install-sphinx = { cmd = "python -m pip install --upgrade sphinx pydata-sphinx-theme nbsphinx nbsphinx-link" }
```

`nbsphinx` convertit les `.ipynb` en pages HTML.  
`nbsphinx-link` permet de référencer des notebooks **hors** du dossier
`docs/source/` (cas des notebooks dans `notebooks/`).

Alternativement, créer des **symlinks** dans `docs/source/examples/` :
```bash
cd docs/source/examples
ln -s ../../../notebooks/01-create-and-display-image.ipynb .
# etc.
```

### 4.2 conf.py — extensions à ajouter

```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "nbsphinx",          # ← nouveau
]

# Noyau Jupyter à utiliser pour exécuter les cellules (si nécessaire)
nbsphinx_kernel_name = "python3"
# Ne pas ré-exécuter les notebooks à chaque build (ils sont pré-exécutés)
nbsphinx_execute = "never"
```

### 4.3 conf.py — options pydata pour les onglets

```python
html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "navbar_center": ["navbar-nav"],   # ← affiche les onglets au centre
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/walcark/adjeff",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
}
```

---

## 5. index.rst racine (toctree = 4 onglets)

```rst
Adjeff
======

Adjeff is a Python library for modelling adjacency effects in satellite
imagery (Sentinel-2), built around Smart-G radiative transfer and a
PyTorch/xarray pipeline.

.. toctree::
   :hidden:
   :maxdepth: 2

   user_guide/index
   examples/index
   api/index
   development/index
```

Le `:hidden:` supprime le toctree du corps de la page — les onglets
apparaissent dans la barre de navigation pydata.

---

## 6. examples/index.rst

```rst
Examples
========

Les notebooks suivants illustrent les cas d'usage typiques d'Adjeff.
Ils peuvent être téléchargés et exécutés localement (GPU requis pour
les notebooks 03 à 06).

.. toctree::
   :maxdepth: 1

   01-create-and-display-image
   02-atmospheric-configuration
   03-compute-radiative-quantities
   04-simulate-rho-toa
   05-compute-2d-radiative-from-maja-output
   06-learn-psf
```

Si les notebooks sont des symlinks dans `examples/`, nbsphinx les
trouve directement. Sinon, utiliser `nbsphinx-link` :

```rst
.. nbgallery::                   ← optionnel, affichage galerie
   notebooks/01-create-...
```

---

## 7. api/index.rst (toctree API)

Déplacer les `.rst` existants dans `api/` et ajouter les manquants :

```rst
API Reference
=============

.. toctree::
   :maxdepth: 2

   core
   atmosphere
   modules
   sweep          ← nouveau (adjeff.sweep)
   optim          ← nouveau (adjeff.optim)
   utils
   exceptions     ← nouveau (adjeff.exceptions)
   accessor       ← nouveau (adjeff.accessor)
```

### Fichiers à créer

**`api/sweep.rst`**
```rst
Sweep
=====

.. automodule:: adjeff.sweep

.. automodule:: adjeff.sweep.bundle
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: adjeff.sweep._dedup
   :members:
   :undoc-members:
   :show-inheritance:
```

**`api/optim.rst`**
```rst
Optimisation
============

.. automodule:: adjeff.optim

.. automodule:: adjeff.optim.loss
   :members: Loss
   :show-inheritance:

.. automodule:: adjeff.optim.metrics
   :members: Metric
   :show-inheritance:

.. automodule:: adjeff.optim.training_set
   :members: TrainingImages, TrainingSet, TrainingSample
   :show-inheritance:

.. automodule:: adjeff.optim.optimizer
   :members: OptimizerPipeline, SingleStageOptimizer
   :show-inheritance:

.. automodule:: adjeff.optim.lbfgs_optimizer
   :members: LBFGSOptimizer, LBFGSConfig, LBFGSStage
   :show-inheritance:

.. automodule:: adjeff.optim.adam_optimizer
   :members: AdamOptimizer, AdamConfig, AdamStage
   :show-inheritance:
```

**`api/exceptions.rst`**
```rst
Exceptions
==========

.. automodule:: adjeff.exceptions
   :members:
   :show-inheritance:
```

**`api/accessor.rst`**
```rst
Accessor
========

.. automodule:: adjeff.accessor
   :members:
   :undoc-members:
   :show-inheritance:
```

### Corrections dans utils.rst

Supprimer la section `ConfigBundle` (module `adjeff.utils.config_bundle`
qui n'existe plus — renommé en `sweep`).

---

## 8. user_guide/index.rst (squelette)

```rst
User Guide
==========

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   pipeline
   atmosphere
   psf
```

Contenu suggéré :

- **installation.rst** — `pixi run -e dev test`, dépendances GPU
- **quickstart.rst** — exemple minimal end-to-end (ImageDict →
  RadiativePipeline → Toa2Unif → LBFGSOptimizer)
- **pipeline.rst** — expliquer SceneModule, Pipeline, required_vars /
  output_vars
- **atmosphere.rst** — AtmoConfig, GeoConfig, SpectralConfig, sweeps
- **psf.rst** — PSFGrid, modèles analytiques, PSFDict, optimisation

---

## 9. Ordre de travail suggéré

1. Créer la structure de dossiers (`user_guide/`, `examples/`, `api/`,
   `development/`)
2. Déplacer les `.rst` existants dans `api/`
3. Créer les nouveaux `.rst` manquants (`sweep`, `optim`, `exceptions`,
   `accessor`)
4. Mettre à jour `index.rst` racine (toctree hidden)
5. Installer `nbsphinx` + créer symlinks des notebooks
6. Créer `examples/index.rst`
7. Vérifier que `pixi run -e dev-gpu build-doc` passe sans erreur
8. Rédiger `user_guide/` au fil des sessions
