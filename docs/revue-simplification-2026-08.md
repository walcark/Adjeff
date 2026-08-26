# Revue de simplification, adjeff 0.8.0

Date : 2026-08-25. Branche : `fix/rho-atm-duplicate`.

Revue de la codebase (10 552 lignes dans `src/adjeff`) sur trois axes
demandés : délégation à des bibliothèques existantes, simplification de
l'expérience utilisateur, et opportunité d'un sous-module d'analyse.

Le code utilisateur de référence est celui des figures de l'article, dans
`../adjeff-article-rse/figures` et `../adjeff-article-rse/src/adjeff_article_1`.

**Synthèse chiffrée** : environ 1 200 à 1 500 lignes sur 10 552 (12 à 14 %)
sont remplaçables par des dépendances déjà présentes ou par de la
déduplication, sans perte de fonctionnalité.

### Où sont réellement les lignes

Classification ligne par ligne via `ast` pour les docstrings et `tokenize`
pour les commentaires.

| Catégorie | Lignes | Part |
| --- | --- | --- |
| Code | 5 178 | 49,1 % |
| Docstrings | 3 922 | 37,2 % |
| Lignes vides | 1 179 | 11,2 % |
| Commentaires | 273 | 2,6 % |

387 symboles documentables, soit **10,1 lignes de doc par symbole**. Le coût
de la documentation est linéaire au nombre de symboles publics, pas à la
complexité du code. Le seul levier est d'en publier moins, pas d'écrire plus
court.

| Package | code | doc | doc/code | Lecture |
| --- | --- | --- | --- | --- |
| `modules` | 1 971 | 1 400 | 0,71 | dont `_smartg.py` : 592 code |
| `(racine)` = `api.py` + `accessor.py` | 761 | 710 | **0,93** | façade, re-documente l'existant |
| `optim` | 755 | 354 | 0,47 | le mieux dosé |
| `core` | 678 | 476 | 0,70 | |
| `utils` | 623 | 598 | **0,96** | sur-documenté pour de l'interne |
| `atmosphere` | 318 | 296 | **0,93** | |
| `reference` | 72 | 88 | **1,22** | justifié, la prose *est* la valeur |

**Conséquence** : diviser le total par 2 exigerait de supprimer toute la
documentation. Le plancher atteignable par le présent plan est de l'ordre de
`code 5 178 → ~4 300` et `total 10 552 → ~9 000`, soit environ −15 %.
Les verrous non compressibles sont l'adaptation à Smart-G (592 lignes de
code dans `_smartg.py`, prix d'une API externe verbeuse et à forme
instable), la couche cache et provenance (~450 lignes, ce qui rend les runs
reproductibles) et les 5 familles de PSF analytiques (182 lignes, le sujet
scientifique).

---

## 1. Délégation à des bibliothèques existantes

Objectif double : moins de lignes, et une vérité externe mieux validée.

### 1.1 Gains directs

| Zone | LOC | Remplacer par | Gain | Risque |
| --- | --- | --- | --- | --- |
| `utils/torchutils.py` : `Transform`, `IdentityTransform`, `ExpTransform`, `SigmoidTransform`, `ConstrainedParameter` | ~170 | `torch.distributions.transforms` (mêmes noms, même contrat `forward` / `.inv`) plus `torch.nn.utils.parametrize` | −140 | faible |
| `utils/radial.py`, bloc `radial()` de `accessor.py`, rebinning dupliqué dans `optim/landscape.py` | ~380 | `scipy.stats.binned_statistic` (déjà dépendance). Validation croisée avec `photutils.profiles.RadialProfile` / `CurveOfGrowth` | −180, et une seule implémentation du binning radial | faible |
| `optim/_combo_stage.py`, `adam_optimizer.py`, `lbfgs_optimizer.py`, `_config.py` | 403 | `scipy.optimize.least_squares(method="trf", bounds=...)` pour les PSF à 1 à 3 paramètres | −350, bornes natives (supprime aussi `ConstrainedParameter`), et fin du contournement `IndexError` du line-search strong-Wolfe | moyen |
| `core/bands.py` : table `wl_nm` codée en dur | 53 | `sensorsio` (déjà dépendance) ou les SRF officielles S2A / S2B | vérité externe : les longueurs d'onde centrales diffèrent entre S2A et S2B, une valeur unique est une approximation non tracée | faible |
| `utils/logger.py` : `MultilineConsoleRenderer` | 75 | `structlog.dev.ConsoleRenderer(pad_event=..., sort_keys=...)`. **Code mort** : jamais configuré, voir section 7 | −75 | faible |
| `optim/metrics.py` : `mae`, `mse`, `rmse` | ~40 | `torch.nn.functional.{l1,mse}_loss`, les variantes `_RAD` restent maison | −20 | faible |

Vérifications faites dans l'environnement `dev` : `torch.distributions.transforms`
expose bien `ExpTransform`, `SigmoidTransform`, `AffineTransform`,
`ComposeTransform` ; `scipy.stats.binned_statistic`, `scipy.optimize.least_squares`
et `torch.nn.utils.parametrize` sont disponibles.

### 1.2 Point à trancher

`utils/xrutils.ParamBatch` (171 lignes, dont ~110 pour la classe) recouvre
en partie le `batch()` et le `dedup` de `xsweep`, qui empile et déduplique
déjà. `ParamBatch` re-empile ensuite à l'intérieur de la fonction physique
(`_smartg._make_atmosphere`). À mesurer avant de trancher : y a-t-il un
empilement redondant, ou les deux couches répondent-elles à des besoins
distincts ?

### 1.3 À garder tel quel

| Zone | Raison |
| --- | --- |
| `utils/convolve.py` | `torch.fft` **est** déjà la vérité externe. Aucun équivalent qui soit à la fois GPU et compatible autograd. La convolution linéaire par FFT avec extension anti-repliement est correcte. |
| `utils/cache_store.py` | `joblib.Memory` ou `diskcache` ne rendraient pas les vues zarr paresseuses, qui sont précisément l'intérêt du cache ici. |

### 1.4 Attention sur `scipy.optimize.least_squares`

C'est le plus gros gain de la liste, et le seul à risque moyen. Arbitrage :

- **Pour** : les PSF analytiques ont 1 à 3 paramètres, `trf` gère les bornes
  nativement (donc plus besoin des transformations Exp / Sigmoid ni de
  `ConstrainedParameter`), la convergence sur un problème de moindres carrés
  à si peu de paramètres est mieux établie que celle d'un enchaînement
  Adam puis L-BFGS réglé à la main.
- **Contre** : jacobienne numérique, soit 3 passes avant supplémentaires par
  itération. Sur une convolution FFT 1999 × 1999 en GPU, c'est probablement
  négligeable, mais cela reste à mesurer.
- **Conséquence** : le soin apporté dans `convolve.py` à faire passer le
  gradient à travers le noyau (`F.pad` plutôt qu'une affectation en place)
  devient inutile pour ce chemin, sans devenir gênant.

**Méthode recommandée** : prototyper sur une seule figure (par exemple
`figure5.py`, King PSF) et comparer les paramètres ajustés et le temps de
calcul avant de généraliser.

---

## 2. Déduplication interne (pas de bibliothèque, même objectif de LOC)

### 2.1 Code dupliqué

| Zone | LOC | Après | Nature de la redite |
| --- | --- | --- | --- |
| `modules/samplers/*.py` (6 fichiers : `tdir_down` 93, `tdir_up` 92, `tdif_down` 90, `tdif_up` 89, `sph_alb` 84, `rho_atm` 150) | 598 | ~300 | `__init__`, `_get_configs` et `_statics` identiques à 90 %. Déclarables en `ClassVar` (liste des noms de statiques, liste des configs) avec un `__init__` unique dans `SweepSampler`. |
| `core/image_generator.py` | 356 | ~130 | `gaussian_image_dict`, `disk_image_dict` et `random_image_dict` ont le même corps : résolution de `n`, boucle sur les bandes, `square_grid`, `attrs`, `Dataset`. Seule la fonction `_*_data` et le bloc `attrs` changent. Un `_make_image_dict(data_fn, attrs_fn, ...)` suffit. |

### 2.2 Documentation recopiée dans `api.py`

`api.py` compte **534 lignes de docstring pour 510 lignes de code**, réparties
en 14 sections `Parameters` et 14 sections `Returns`. Presque aucun de ces
paramètres n'appartient à `api.py` : ils sont transmis tels quels aux modules
sous-jacents, qui les documentent déjà.

**Cas le plus net, à l'intérieur du même fichier.** `api.py:169-179`, dans
`_make_atmo_config` :

```
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
```

`api.py:271-281`, dans `make_full_config`, identique au caractère près. Et une
troisième fois dans `atmosphere/atmo_config.py:20-29`, la classe `AtmoConfig`
qui reçoit finalement ces valeurs. **Trois copies de la même phrase pour un
même paramètre.**

Ampleur mesurée :

| Paramètre | Fois dans `api.py` | Fois ailleurs | Total |
| --- | --- | --- | --- |
| `afgl_type` | 6 | 19 | 25 |
| `remove_rayleigh` | 6 | 19 | 25 |
| `cache` | 8 | 15 | 23 |
| `n_ph` | 3 | 16 | 19 |
| `species` | 5 | 9 | 14 |
| `dedup` | 3 | 8 | 11 |
| `res_km` | 6 | 4 | 10 |

La phrase `AFGL atmosphere profile (default "afgl_exp_h8km").` apparaît six
fois à l'identique dans `api.py` seul (lignes 441, 558, 650, 753, 835, 1166).

**Correctif.** numpydoc autorise le regroupement de plusieurs paramètres sur
une seule entrée, séparés par des virgules. Les 22 lignes du bloc de
`make_full_config` deviennent 7 :

```
    Parameters
    ----------
    bands : list[SensorBand]
        Sensor bands to simulate.
    aot, rh, h, href, species
        Atmospheric state, see :class:`~adjeff.atmosphere.AtmoConfig`.
    sza, vza, saa, vaa, sat_height
        Acquisition geometry, see :class:`~adjeff.atmosphere.GeoConfig`.
```

Pour les fonctions qui ne font que transmettre (`run_forward_pipeline`,
`load_scene`, `load_maja`, `run_radiatives_from_scene`), la section numpydoc
`Other Parameters` avec une ligne de renvoi remplace le bloc entier :

```
    Other Parameters
    ----------------
    remove_rayleigh, afgl_type, n_ph, batch_size, dedup
        Forwarded verbatim, see
        :class:`~adjeff.modules.samplers.RadiativePipeline`.
```

**Bénéfice** : environ 300 lignes sur les 534 de doc de `api.py`, et la fin
d'une classe entière de dérive silencieuse. Aujourd'hui, changer le défaut de
`n_ph` dans un sampler laisse trois docstrings mentir sans qu'aucun test ni
`ruff` ne le détecte.

### 2.3 Signatures recopiées

Les blocs `@overload` de `run_forward_pipeline` et `run_radiatives_from_scene`
coûtent **48 lignes** de signatures recopiées, uniquement pour exprimer
qu'une liste en entrée donne une liste en sortie. Un `TypeVar` borné supprime
les quatre blocs.

---

## 3. Expérience utilisateur

### 3.1 La meilleure source : `shim.py`

`../adjeff-article-rse/src/adjeff_article_1/shim.py` (214 lignes) est un
document de conception involontaire. Sa propre docstring l'annonce :

> Everything in this module exists because the corresponding operation is
> missing upstream. Each function names the adjeff addition that would
> delete it. This module is meant to shrink to nothing: its size measures
> how much of the article's plumbing adjeff still pushes onto its callers.

Chaque fonction porte une ligne `Would be deleted by:`. C'est la liste de
travaux, déjà validée par l'usage réel.

**Indicateur de réussite proposé : `shim.py` doit tomber à zéro ligne.**

### 3.2 Liste des irritants

| Irritant observé | Correctif proposé | Où ça mord |
| --- | --- | --- |
| `rho_s` est à la fois la vérité terrain et la sortie de `Unif2Surface` : une scène qui porte les deux perd la vérité | liaison de ports, voir section 4 | `shim.correct` (30 lignes), commentaire défensif dans `psf_comparison.py` |
| Les scalaires de config sont coercés en tableaux de longueur 1 par `to_arr` : toute sortie traîne des dimensions singleton `aot`, `rh`, `h`, `href` | garder un scalaire scalaire, ou `drop_singleton=True` sur les sorties | `shim.select_scalar`, appelé partout |
| `Metric` n'accepte que des tenseurs : quatre `.adjeff.to_tensor().to(device)` par appel | accesseur `pred.adjeff.rmse(truth, mask=...)`, ou `Metric` acceptant des `DataArray` | `shim.radial_rmse`, `hotspot.py` (5 appels) |
| Pas de recherche inverse longueur d'onde vers bande | `S2Band.from_wl(665.0)` | `shim.wl_to_band` |
| Profil radial non miroir, impossible de tracer un transect complet | `da.adjeff.radial(symmetric=True)` | `shim.sym_profile`, figures 4, 5, 7 à 17 |
| Les six quantités du 5S sont redéclarées une à une dans chaque module, jamais publiées | `adjeff.modules.samplers.RADIATIVE_VARS` | `shim.RADIATIVE_VARS` |
| `TrainingImages(images=s, weights=[1.0] * len(s))` réécrit à chaque appel | `weights=None` par défaut, poids uniformes | figures 2, 3, 7 à 17, `psf_comparison.py` |
| `make_model(...)` puis `fit(...)` : deux étapes toujours accolées | `fit_psf` existe mais impose ses propres scènes de disques. Exposer `fit(model_cls=..., psf_type=..., scenes=...)` | `figure7_17.optimised_kernel`, `psf_comparison_figure` |
| `run_forward_pipeline` bouclé à la main sur une liste de scènes générées | helper `training_scenes(band, shapes=..., cfg=...)` | `scenes.gauss_scenes` et `scenes.disk_scenes`, environ 80 lignes recopiées |

---

## 4. Le problème `rho_s` contre `rho_s_est` : liaison de ports

### 4.1 Le vrai diagnostic

Renommer la sortie de `Unif2Surface` en `rho_s_est` ne suffit pas : en aval,
`RhoToaSampler` exige `rho_s`. La collision se déplace, elle ne disparaît pas.

La cause est que `required_vars` et `output_vars` confondent deux choses
distinctes :

1. le **rôle physique** que la variable joue dans le module (« la réflectance
   de surface que je convolue »),
2. le **nom du slot** dans le `Dataset` où elle est rangée.

Il faut séparer les deux, et laisser l'instance relier l'un à l'autre.

### 4.2 API cible

La classe déclare des rôles canoniques, inchangés. L'instance choisit où
lire et où écrire.

```python
# La vérité terrain reste `rho_s`. L'estimation King prend son propre slot.
estim = Unif2Surface(psfs=psfs, rename={"rho_s": "rho_s_king"})
scene = estim(scene)          # écrit rho_s_king, ne touche pas rho_s

# Le sampler lit ce slot, sans savoir qu'il s'agit d'une estimation.
toa = RhoToaSymSampler(**cfg, rename={"rho_s": "rho_s_king"})
scene = toa(scene)            # rho_toa simulé depuis l'estimation

# La comparaison est directe, dans la même scène :
scene[band]["rho_s"], scene[band]["rho_s_king"]
```

Un seul dictionnaire suffit : aujourd'hui aucun module ne lit et n'écrit le
même rôle. Si le cas se présente un jour, scinder en `inputs=` et `outputs=`.

### 4.3 Surface de la modification

| Fichier | Changement | Lignes |
| --- | --- | --- |
| `modules/scene_module.py` | `__init__(rename=...)`. `required_vars`, `output_vars` et `optional_vars` deviennent des propriétés résolues sur des `ClassVar` renommés `_required_vars`, etc. | ~25 |
| `modules/scene_module.py` | ajouter `"rename"` à `_INFRA_PARAMS` | 1 |
| `modules/models/psf_conv_module.py` | `_compute` lit `ds[self._slot(k)]` et passe `k=` à `_formula`, qui reste inchangée car indexée par rôle | ~4 |
| `modules/pipeline.py`, `optim/fit.py` | rien, ils lisent déjà les propriétés | 0 |
| `optim/landscape.py` | noms en dur à remplacer par ceux du modèle, voir problème 2 en section 5 | ~15 |

**Point à ne pas rater** : `rename` ne doit pas entrer dans la clé de cache.
Deux exécutions qui ne diffèrent que par le nom du slot calculent exactement
la même chose et doivent tomber sur la même entrée. D'où l'ajout à
`_INFRA_PARAMS`.

**Précédent existant** : `ProductLoader.output_vars` est déjà une propriété
qui surcharge le `ClassVar` (`modules/loaders/product_loader.py:111`). Le
patron est donc admis par le design actuel, ce n'est pas une entorse.

### 4.4 Alternatives écartées

| Option | Pourquoi non |
| --- | --- |
| Module `Rename` intercalé dans le `Pipeline` | explicite mais bruyant, ajoute un saut de provenance et une copie pour un pur changement d'étiquette |
| Dimension `estimator=["truth", "king", "gauss", "wu"]` | séduisant, c'est exactement la structure de la figure 22, mais casse le broadcast et la clé de cache. À garder pour le post-traitement dans le futur `adjeff.analysis`, pas dans le pipeline |

---

## 5. Problèmes potentiels relevés

| # | Constat | Localisation | Action proposée |
| --- | --- | --- | --- |
| 1 | `GeoConfig.sun_le`, `sat_le`, `sun_sensor`, `sat_sensor` (environ 50 lignes) ne sont appelés que par les tests. `_smartg.py` reconstruit ces dictionnaires en local (lignes 168, 475, 616, 800). Deux sources de vérité pour la convention géométrique Smart-G, une seule exercée en production. À noter au passage : `sat_le` utilise `saa` là où `sat_sensor` utilise `vaa`, incohérence que personne ne peut détecter puisque le code n'est jamais appelé | `atmosphere/geo_config.py:44-100` | supprimer, ou faire consommer ces propriétés par `_smartg.py` |
| 2 | La docstring de `loss_landscape` annonce « any PSFModule, Gaussian, King, Voigt, Moffat, or any custom subclass », mais `_make_forward_fn` câble en dur la convolution de `Unif2Surface` et les noms `rho_unif`, `tdir_up`, `tdif_up`, `sph_alb`, `rho_s`. Impossible de tracer un paysage de perte pour un autre `PSFConvModule` | `optim/landscape.py:33-58` et `:105-107` | prendre le `model` en argument, appeler `model.forward_band`, lire les noms sur `model.required_vars` |
| 3 | Le chemin « vues zarr paresseuses » de `SceneModule.forward` ne s'arme que si un cache est configuré. Avec `cache=None` sur un grand sweep, tout reste en RAM et rien ne prévient | `modules/scene_module.py:144-151` | documenter explicitement, ou basculer sur un répertoire temporaire par défaut |
| 4 | `Loss.mask_on` n'accepte que `"rho_unif"` ou `None`, validé à la main dans `__post_init__`, alors qu'il s'agit simplement du nom de la variable qui pilote le masque | `optim/loss.py:32-38` | ouvrir à n'importe quel nom, ce qui supprime le `__post_init__` |
| 5 | `api.py` (1192 lignes) fait de la politique, pas seulement de la composition : `fit_psf` impose `train_radii=[1, 5, 50]`, `n_train=1999` et le nom `rho_s`. Ce sont les choix de l'article, pas ceux de la bibliothèque | `api.py:960-1010` | remonter ces constantes côté appelant, ou les nommer `ARTICLE_*` pour assumer le choix |
| 6 | Le même événement est émis deux fois par deux canaux distincts, et au mauvais niveau : `logger.info(msg)` suivi de `warnings.warn(msg, OptimizationWarning)` | `optim/lbfgs_optimizer.py:105-107` | un seul `logger.warning`, voir section 7.2 |

Le problème 2 et la question du renommage de `rho_s` se soignent avec le même
geste : cesser d'écrire des noms de variables en dur ailleurs que dans la
déclaration de rôle du module.

---

## 6. Sous-module `adjeff.analysis`

**Verdict : oui, mais comme consolidation de code déjà existant et
éparpillé, pas comme nouveau territoire fonctionnel.**

### 6.1 L'état actuel

| Fonction | Où elle vit aujourd'hui | Problème |
| --- | --- | --- |
| profil radial mean / std / cdf / adaptive | `accessor.py:118-243`, en ligne | logique métier dans un accesseur |
| énergie encerclée EE10 / EE50 / EE99 % | `optim/landscape.py:168-218` | réimplémente le binning de `radial.py`, explicitement « pour la vitesse » |
| RMSE radial sur `DataArray` | `shim.py`, côté utilisateur | hors de la bibliothèque |
| profil symétrique | `shim.py`, côté utilisateur | hors de la bibliothèque |
| `digitize` | `accessor.py:352-388` | mi-analyse, mi-préparation de config |

### 6.2 Contenu proposé

L'accesseur `.adjeff` ne serait plus qu'une façade mince qui délègue.

| Contenu | Justification |
| --- | --- |
| `radial_profile(stat=...)`, `transect`, `to_field` | déplacement, pas d'ajout |
| `encircled_energy(kernel, fractions)` | supprime la duplication `landscape.py` contre `radial.py` |
| `fwhm`, `mtf(kernel)` (transformée de Fourier de la PSF), `mtf_at(f)` | **le vrai manque**. La MTF est la quantité canonique pour comparer une PSF à la littérature instrumentale, et personne ne l'a écrite |
| `rmse`, `bias`, `mape` acceptant des `DataArray`, pondération radiale optionnelle | supprime `shim.radial_rmse` |

### 6.3 Garde-fou

Ne pas en faire une bibliothèque de traitement du signal générique. Le
critère d'admission tient en une phrase :

> Est-ce que cela quantifie une PSF ou un champ 2D d'effet d'adjacence ?

Une FFT 1D générique, une densité spectrale de puissance, un filtrage de
Wiener n'ont pas leur place ici. `scipy.signal` existe.

---

## 7. Logging

### 7.1 État des lieux

Le symptôme est sans appel : **les 6 notebooks sur 6 commencent par rediriger
structlog vers `/dev/null`**. La documentation officielle du projet désactive
les logs avant de faire quoi que ce soit. Ils ne sont donc pas seulement peu
clairs, ils sont perçus comme du bruit par leur propre auteur.

| Constat | Détail |
| --- | --- |
| Volume | 23 appels dans 10 552 lignes |
| Niveaux | 12 `debug`, 10 `info`, **1 `warning`**, **0 `error`**, 0 `exception` |
| Loggers déclarés jamais utilisés | `_smartg.py`, `rho_toa.py`, `rho_toa_sym.py` : 3 `get_logger()` pour 0 appel |
| Code mort | `MultilineConsoleRenderer` (75 lignes) n'est configuré nulle part, ni dans `src/`, ni dans les notebooks, ni dans les figures de l'article |
| Trous de couverture | `Pipeline` (234 lignes), `SweepSampler._sweep`, `PSFConvModule`, `optim/landscape` : zéro log, alors que ce sont les chemins qui durent des heures |
| Contexte | `structlog.contextvars` inutilisé. Un seul `logger.bind()` dans tout le paquet (`scene_module.py:130`) |

### 7.2 Couverture : une règle, pas une liste

Plutôt qu'un audit fichier par fichier, une règle unique et vérifiable :

> Tout ce qui peut durer plus d'une seconde, échouer, ou être sauté, produit
> une ligne.

Quatre points d'instrumentation obligatoires en découlent :

| Événement | Où c'est manquant | Niveau |
| --- | --- | --- |
| Entrée et sortie d'une unité de travail, avec sa durée | `Pipeline.__call__` par module, `SweepSampler._sweep`, `PSFConvModule._compute` | `info` |
| Décision de cache (hit, miss, write) | existe déjà mais en `debug`, donc invisible par défaut | `info` pour le hit, `debug` pour le détail |
| Travail sauté ou dégradé | `MajaLoader` qui met RH à 50 % par défaut, `dedup` qui collapse N états en M, `_pair_angles_with_points` qui ne garde qu'une diagonale | `warning` |
| Coût engagé avant de le payer | nombre d'états atmosphériques, de photons, taille de grille, annoncés avant l'appel Smart-G | `info` |

Le quatrième est le plus rentable : aujourd'hui rien ne dit, avant de lancer,
que le sweep va produire 4 000 appels Smart-G.

La couverture se mesure ensuite mécaniquement : chaque `SceneModule` doit
émettre au moins un `info` de début et un de fin. C'est testable dans
`conftest.py` avec `structlog.testing.capture_logs`.

### 7.3 Sémantique des niveaux

Le critère n'est pas la gravité, c'est qui lit la ligne et ce qu'il en fait.

| Niveau | Question à laquelle il répond | Destinataire | Exemples adjeff |
| --- | --- | --- | --- |
| `debug` | Pourquoi ce résultat précis ? | le développeur en train de déboguer | clé de cache complète, dims produites par Smart-G, paramètres d'une atmosphère individuelle, chemin zarr écrit |
| `info` | Où en est mon run, et combien cela coûte ? | l'utilisateur qui regarde son terminal pendant 3 heures | début et fin de module avec durée, `48/512 combos`, `cache hit`, `4 096 états → 128 après dedup` |
| `warning` | Le résultat est valide, mais ce n'est pas celui que tu croyais demander | l'utilisateur, à relire après le run | RH absente donc 50 % par défaut, line-search L-BFGS dégénéré, entrée de cache tronquée, extrapolation hors domaine |
| `error` | Cela a échoué, et je continue quand même | l'utilisateur, immédiatement | une bande sur six a échoué et le run continue sur les cinq autres |

**Règle de départage `info` contre `debug`** : si la ligne contient une valeur
numérique que l'utilisateur ne peut pas prévoir avant le run et qui l'aiderait
à décider d'attendre ou d'annuler, c'est `info`. Sinon c'est `debug`.

**Règle `warning` contre exception** : une `warning` n'interrompt rien et
laisse un résultat exploitable. Si le résultat est inexploitable, c'est une
exception, pas un log. adjeff respecte déjà cette règle (les erreurs passent
par `AdjeffError` et ses sous-classes), il ne manque que le niveau `warning`,
aujourd'hui quasi vide.

### 7.4 Améliorations des logs existants

#### a. Le message devient un événement stable, les données deviennent des clés

C'est la promesse de structlog, aujourd'hui à moitié tenue. Les messages sont
des phrases anglaises hétérogènes :

```
"done"                                  # scene_module, le plus fréquent
"cache miss"
"Creating Gaussian ImageDict."
"Merge all atmospheres with multi-profile."
"Computed atmospheric PSF."
"Scene was saved to cache."
```

Impératif, gérondif, passé, participe passé, avec et sans point final. Et
`"done"` seul, sur le chemin le plus chaud, ne dit ni quoi ni combien de temps.

Convention proposée : `objet.action`, en minuscules, sans ponctuation, jamais
interpolé.

| Avant | Après |
| --- | --- |
| `log.info("done", bands=[...], cached=True)` | `log.info("module.done", module="RhoAtmSampler", bands=6, cached=True, duration_s=0.02)` |
| `logger.debug("Creating Gaussian ImageDict.", bands=bands)` | `logger.debug("scene.generate", shape="gaussian", bands=6, n=1999)` |
| `logger.info(f"Adam {n}/{max} loss={loss:.4g}{delta}")` | `logger.info("fit.step", optimizer="adam", step=n, of=max, loss=loss, delta_pct=d)` |

Le troisième cas est le plus important : `adam_optimizer.py:76` et
`lbfgs_optimizer.py:113` construisent une f-string pré-formatée, ce qui annule
tout l'intérêt de structlog. Impossible de filtrer sur `loss`, de tracer la
courbe, ou de sortir en JSON.

#### b. Un contexte par run

`structlog.contextvars.bind_contextvars(run_id=..., band=..., combo=...)` en
tête de `fit()` et de `Pipeline.__call__`, et toute ligne émise en dessous
porte le contexte sans qu'aucun appelant n'ait à le transmettre. Cela supprime
les `band=band` répétés à la main et rend les logs corrélables sur un run à
512 combos.

#### c. Des durées, systématiquement

Aucun log ne porte de durée aujourd'hui. Un context manager de 10 lignes dans
`SceneModule.forward` suffit, et c'est ce qui fait passer les logs de
« informatiques » à « utiles ».

#### d. Supprimer `MultilineConsoleRenderer`, fournir `setup_logging()`

75 lignes de renderer mort, alors que `structlog.dev.ConsoleRenderer` fait le
travail. À remplacer par un helper opt-in exporté :

```python
adjeff.setup_logging(level="info")                # ConsoleRenderer, humain
adjeff.setup_logging(level="debug", json=True)    # pour un run SLURM
```

Une bibliothèque ne doit pas configurer le logging global d'elle-même, mais
elle doit fournir un chemin d'une ligne pour le faire. C'est ce qui manque, et
c'est pourquoi les 6 notebooks ont préféré tout couper.

---

## 8. Ordre de traitement suggéré

Du plus sûr au plus engageant.

| Ordre | Chantier | Gain | Risque |
| --- | --- | --- | --- |
| 1 | `torch.distributions.transforms` à la place de `utils/torchutils.Transform` et `ConstrainedParameter` | −140 lignes | faible |
| 2 | Liaison de ports `rename=` (section 4), qui débloque au passage le problème 2 de la section 5 | débloque le cas d'usage `rho_s_est` vers `RhoToaSampler` | faible |
| 3 | Problèmes 1, 3, 4 et 5 de la section 5 (code mort, documentation mémoire, `mask_on`, constantes d'article) | −50 lignes, clarté | faible |
| 4 | Consolidation `adjeff.analysis` plus `scipy.stats.binned_statistic` | −180 lignes, une seule implémentation du binning | faible |
| 5 | Résorption de `shim.py` jusqu'à zéro (section 3) | supprime 214 lignes chez l'utilisateur | faible |
| 5 bis | Dé-recopie des docstrings de `api.py` et suppression des `@overload` (sections 2.2 et 2.3) | −350 lignes, fin de la dérive silencieuse des valeurs par défaut | faible |
| 6 | Déduplication des samplers et de `image_generator` | −520 lignes | moyen, beaucoup de fichiers touchés |
| 7 | `scipy.optimize.least_squares` à la place des étages Adam et L-BFGS | −350 lignes | moyen, à prototyper sur une figure d'abord |
| 8 | `setup_logging()` plus suppression de `MultilineConsoleRenderer`, et correction du niveau dans `lbfgs_optimizer` (sections 7.3 et 7.4d) | −75 lignes, les notebooks peuvent rallumer les logs | faible |
| 9 | Convention `objet.action` et clés structurées sur les 23 appels existants, plus les durées (sections 7.4a et 7.4c) | logs exploitables et filtrables | faible |
| 10 | Couverture des 4 points d'instrumentation manquants, plus `contextvars` (sections 7.2 et 7.4b) | visibilité sur les runs longs | moyen |
| 11 | Trancher le recouvrement `ParamBatch` contre `xsweep` | à mesurer | à instruire |

La section 9 rassemble des points d'outillage et de tests qui sont **hors de
ce chemin de priorité**. Ils ne bloquent aucun des chantiers 1 à 11.

---

## 9. Outillage, tests et empaquetage (hors priorité)

Relevés en marge de la revue de `src/`, consignés pour mémoire.

### 9.1 `import adjeff` exige `SMARTG_DIR_AUXDATA`

`pixi run test` échoue tel quel : 10 erreurs de collecte, tous les tests non
intégration inclus.

```
src/adjeff/__init__.py:8                 → from .api import (...)
src/adjeff/api.py:42                     → from adjeff.atmosphere import ...
src/adjeff/atmosphere/atmo_factory.py:7  → from smartg.atmosphere import AerOPAC, AtmAFGL
smartg/config.py:26                      → NameError: 'SMARTG_DIR_AUXDATA' does not exist!
```

Deux imports de niveau module sont fautifs, alors que **23 autres imports
Smart-G sont soigneusement différés** dans le corps des fonctions. Ces deux-là
sont un oubli, pas un choix.

| Fichier | Import | Déclenche l'erreur ? |
| --- | --- | --- |
| `atmosphere/atmo_factory.py:7` | `from smartg.atmosphere import AerOPAC, AtmAFGL` | **oui** |
| `atmosphere/surface.py:14` | `from smartg.water import Albedo_cst` | **oui** |
| `modules/samplers/_smartg.py:15` | `from smartg.visualizegeo import ...` | non |
| `modules/samplers/_smartg.py:12` | `import geoclide as gc` | non |

**Preuve que la contrainte est fictive** : avec la variable pointant un
répertoire vide, les 183 tests passent en 26 secondes. Aucune donnée
auxiliaire n'est réellement nécessaire, seul l'import l'est.

Conséquences :

- La CI ne le voit pas, elle exporte `SMARTG_DIR_AUXDATA` juste avant l'étape
  `Test` (`.github/workflows/ci.yml`).
- Un `pip install adjeff` puis `import adjeff` casse pour quelqu'un qui ne veut
  que les parties CPU (accesseur, générateurs d'images, modèles de PSF,
  analyse radiale), soit la moitié de la bibliothèque qui n'a pas besoin de GPU.
- Un `pixi run test` sans la variable donne l'impression que la suite est
  cassée.

**Correctif** : différer les deux imports comme les 23 autres, et ajouter un
test `test_import_without_smartg` qui fait `import adjeff` dans un
sous-processus avec `SMARTG_DIR_AUXDATA` retirée de l'environnement. Deux
lignes de production, si un gain rapide est cherché un jour.

### 9.2 Outillage

| Constat | Détail |
| --- | --- |
| `line-length = 79` (`pyproject.toml:141`) | les guidelines de référence (`~/.claude/guidelines/python.md:10`) imposent 88. Divergence assumée ou oubli ? |
| `ruff exclude = ["tests/*", "docs/*", "scripts/*", "notebooks/*"]` | les tests ne sont ni formatés, ni lintés, ni typés (mypy ne cible que `src/adjeff`). La CI ne les vérifie donc pas non plus, alors qu'ils font environ 1 050 lignes |
| `[system-requirements]` | déprécié par pixi, avertissement affiché à chaque commande. Correctif indiqué par l'outil : déclarer `cuda` sur l'entrée `platforms` |

### 9.3 Couverture : 71 %, deux populations à ne pas confondre

| Légitimement bas (GPU requis) | Non légitime (pur CPU) |
| --- | --- |
| `_smartg.py` 19 %, `smartgutils.py` 22 %, `surface.py` 20 %, samplers environ 60 % | **`optim/landscape.py` 18 %**, **`optim/fit.py` 24 %**, `api.py` 33 %, `psf_conv_module.py` 42 %, `adam_optimizer.py` 44 %, **`optim/metrics.py` 49 %** |

Deux cas méritent une attention particulière :

- **`optim/landscape.py` à 18 %** : c'est le module qui a produit les figures 2
  et 3 de l'article, il tourne en `device="cpu"`, rien ne s'oppose à le tester.
  Et c'est celui dont la docstring est fausse (problème 2 de la section 5).
  Faible couverture plus documentation fausse est la combinaison qui laisse
  passer une régression inaperçue.
- **`optim/metrics.py` à 49 %** : le cœur scientifique de l'ajustement, moitié
  non testé.
