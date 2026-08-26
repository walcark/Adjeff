# Plan d'observabilité, adjeff 0.13.0

Établi le 2026-08-26, contre le code de la 0.12.0. Toutes les mesures citées
ont été prises sur ce code, pas reprises de la revue de simplification, qui
avait été écrite contre la 0.8.0.

---

## Le diagnostic, en trois chiffres

**94 %.** Sur un run forward complet de 13.8 s (7 appels Smart-G, 1 bande,
grille 201×201), le temps passé dans un silence de plus d'une seconde. Le plus
long : 2.15 s, soit 15.5 % du run. Chaque module ne parle qu'à la fin, jamais
avant ni pendant.

**8 fois `"done"`.** Les huit lignes de niveau `info` émises pendant ce run
portent toutes le même message, sans nom de module, sans durée, sans reste à
faire.

**21 lignes jetées.** `xsweep` émet pendant ce même run 21 enregistrements
`INFO` qu'adjeff ne voit pas :

```
INFO  sweep runs in memory: results are not persisted and no cache was consulted
INFO  sweep start points=1 calls=1 cached=0 skipped=0
INFO  sweep done ok=1 failed=0 skipped=0 cached=0 elapsed=2.2s
```

C'est exactement ce que la revue demandait d'ajouter à adjeff : le coût annoncé
avant d'être payé, le décompte, la durée, en clé-valeur. Les `elapsed=2.2s`
remplissent précisément les silences mesurés plus haut. Et la première ligne
est un avertissement sérieux pour un run GPU de plusieurs heures.

### Pourquoi elles sont jetées

adjeff n'appelle jamais `structlog.configure()` et hérite donc des défauts de
structlog 25.5.0 :

| | valeur | conséquence |
| --- | --- | --- |
| `logger_factory` | `PrintLoggerFactory` | écrit sur stdout, hors du `logging` standard |
| `wrapper_class` | `BoundLoggerFilteringAtNotset` | aucun filtrage, `debug` s'affiche |

Deux tuyaux disjoints, sans passerelle. Aucun niveau ne s'applique à adjeff, et
aucun réglage d'adjeff n'atteint `xsweep`. C'est la raison mécanique pour
laquelle les six notebooks sur six redirigent structlog vers `/dev/null` : il
n'existe aucun autre moyen de faire taire les `debug`.

À noter tout de même : `merge_contextvars` est déjà dans la chaîne par défaut.
Le contexte par run ne demande donc que de lier des variables, pas de
reconfigurer quoi que ce soit.

### L'état du `logging` standard pendant ce run

205 enregistrements, dont 35 utiles :

| logger | enregistrements | verdict |
| --- | --- | --- |
| `xsweep` | 21 `INFO` + 7 `DEBUG` | à faire remonter |
| `zarr.group` | 130 `DEBUG` | bruit |
| `numcodecs` | 27 `DEBUG` | bruit |
| `matplotlib`, `h5py`, `asyncio`, `trimesh`, `zarr.core.sync` | 12 `DEBUG` | bruit |

Plus 9 `warnings` hors du flux de log, dont 7 `ResourceWarning` levés par
`smartg.py:926`, un par appel Smart-G. Et 7 lignes écrites directement sur
stdout par Smart-G (`There is no current context to clear.`), qui ne sont pas
des enregistrements de log et qu'aucune configuration n'attrapera.

---

## Principe directeur

> Un seul transport, le `logging` de la bibliothèque standard. structlog reste
> l'API d'écriture ; il cesse d'être le transport.

```
adjeff   ─┐
xsweep   ─┼─→  logging  ─→  un handler  ─→  ConsoleRenderer  ou  JSON
zarr     ─┤
warnings ─┘
```

Trois propriétés qu'aucune autre disposition ne donne :

1. un seul niveau gouverne adjeff **et** `xsweep`, qui sont les deux moitiés du
   même travail ;
2. un utilisateur qui a déjà sa configuration `logging` reçoit les lignes
   d'adjeff dans ses handlers, à son format, sans rien changer ;
3. adjeff redevient silencieux à l'import, ce qu'une bibliothèque doit être.

Et un corollaire qui gouverne le contenu des tâches 5 et 6 :

> Ne pas redire ce que `xsweep` dit déjà. Il rapporte le sweep ; adjeff
> rapporte ce que `xsweep` ne peut pas voir.

---

## Tâches

Ordre choisi pour que chaque étape soit vérifiable seule et que la suivante
s'appuie sur elle. Les tâches 1 et 2 sont l'infrastructure, 3 et 4 la forme,
5 et 6 le contenu, 7 la sortie.

### Tâche 1 — Le transport

**Faire.** Un `adjeff/_logging.py` exposant `get_logger(name)` qui enveloppe
`logging.getLogger(name)` via `structlog.wrap_logger`, sans jamais toucher à la
configuration globale de structlog. `NullHandler` posé sur le logger `adjeff`.
Remplacer les points d'appel de `structlog.getLogger(__name__)`. Supprimer
`MultilineConsoleRenderer` (`utils/logger.py`, 75 lignes mortes, référencé
nulle part sauf dans le docstring de `utils/__init__.py`).

**Pourquoi.** C'est ce qui débranche `PrintLoggerFactory` et rend tout le reste
possible. Sans cette tâche, aucune des suivantes n'a d'effet sur `xsweep`.

**Vérifier.** Importer adjeff et lancer un pipeline n'écrit rien sur stdout ;
le `caplog` de pytest voit les enregistrements. Un test qui assert que
`logging.getLogger("adjeff").handlers` ne contient qu'un `NullHandler`.

**Risque.** Faible, mécanique. Attention au seul point non trivial : les
`log.bind()` de `scene_module.py` doivent continuer de fonctionner, ce que
`structlog.stdlib.BoundLogger` assure.

---

### Tâche 2 — `setup_logging()`

**Faire.** Exporter depuis `adjeff` :

```python
adjeff.setup_logging(level="info")                # console lisible
adjeff.setup_logging(level="debug", json=True)    # run SLURM
```

Elle attache un handler avec `structlog.stdlib.ProcessorFormatter`, et règle
trois choses qu'un utilisateur ne devrait pas avoir à écrire :

| | |
| --- | --- |
| niveau d'`adjeff` **et** de `xsweep` | celui demandé |
| `zarr`, `numcodecs`, `matplotlib`, `asyncio`, `h5py`, `trimesh` | `WARNING` |
| `logging.captureWarnings(True)` | les 9 warnings mesurés rejoignent le flux |

Le deuxième point vient de la mesure : ces loggers ont produit 170 des 205
enregistrements, tous du bruit. Prévoir un paramètre pour l'ajuster plutôt
qu'une liste figée.

**Pourquoi.** Une bibliothèque ne configure pas le `logging` global d'elle-même,
mais elle doit offrir un chemin d'une ligne pour le faire. C'est précisément ce
qui manque, et c'est pourquoi les notebooks ont préféré tout couper.

**Vérifier.** Un test qui assert qu'après `setup_logging(level="info")` un
enregistrement `xsweep` de niveau `INFO` atteint le handler et qu'un
enregistrement `zarr` de niveau `DEBUG` ne l'atteint pas.

---

### Tâche 3 — Durées et contexte

**Faire.** Un context manager d'une dizaine de lignes dans
`SceneModule.forward`, qui émet `module.start` en entrée et `module.done` en
sortie avec `duration_s`. Et `structlog.contextvars.bind_contextvars(run_id=,
band=, combo=)` en tête de `fit()` et de `Pipeline.__call__`.

**Pourquoi.** Aucun log ne porte de durée aujourd'hui, et c'est ce qui fait
passer les logs d'informatiques à utiles. Le contexte supprime les `band=band`
recopiés à la main et rend les lignes corrélables sur un run à 512 combos.
`merge_contextvars` étant déjà dans la chaîne, il n'y a rien à configurer.

**Vérifier.** Chaque `SceneModule` émet au moins un `info` d'entrée et un de
sortie, portant `duration_s`. Testable mécaniquement sur la liste des
sous-classes.

---

### Tâche 4 — Convention `objet.action` et niveaux

**Faire.** Renommer les messages en `objet.action`, minuscules, sans
ponctuation, jamais interpolés.

| avant | après |
| --- | --- |
| `log.info("done", bands=[...], cached=True)` | `log.info("module.done", module="RhoAtmSampler", bands=6, cached=True, duration_s=0.02)` |
| `logger.debug("Creating analytical ImageDict.", ...)` | `logger.debug("scene.generate", shape="gaussian", bands=6, n=1999)` |
| `logger.info(f"combo {done}/{total}", ...)` | `logger.info("fit.combo", done=done, of=total, band=...)` |

Le dernier est le seul reste de f-string (`optim/fit.py:129`) et il annule tout
l'intérêt de structlog : impossible de filtrer, de tracer la courbe, ou de
sortir en JSON.

Passer le `cache hit` de `debug` à `info`. Règle de départage : si la ligne
porte une valeur numérique que l'utilisateur ne peut pas prévoir avant le run
et qui l'aiderait à décider d'attendre ou d'annuler, c'est `info`.

**Vérifier.** Un test qui parcourt les appels de log du paquet et assert
qu'aucun message ne contient d'espace, de majuscule ou de point final, et
qu'aucun n'est une f-string.

---

### Tâche 5 — Combler les trous

**Faire.** Quatre fichiers sont aujourd'hui à zéro log, et ce sont les chemins
qui durent des heures :

| fichier | à ajouter |
| --- | --- |
| `modules/pipeline.py` | entrée et sortie de chaque module de la chaîne |
| `modules/sweep_sampler.py` | le coût agrégé **avant** le premier appel : états × bandes × photons |
| `modules/models/psf_conv_module.py` | entrée, sortie, durée |
| `optim/landscape.py` | progression du balayage |

Le deuxième est le plus rentable : rien ne dit aujourd'hui, avant de lancer,
que le sweep va produire des milliers d'appels Smart-G. `xsweep` annonce
`points=` et `calls=` par sweep, mais personne n'annonce le total.

**Ne pas ajouter** : le décompte des points, le cache consulté, la durée du
sweep. `xsweep` les émet déjà, et la tâche 2 les rend visibles.

**Pourquoi.** La règle de couverture : tout ce qui peut durer plus d'une
seconde, échouer, ou être sauté produit une ligne.

**Vérifier.** Rejouer la mesure de silence. Objectif : aucun silence de plus de
2 s sur le run de référence, contre 94 % du temps aujourd'hui.

---

### Tâche 6 — Le niveau `warning`, aujourd'hui quasi vide

**Faire.** Deux `warning` existent dans tout le paquet
(`optim/lbfgs_optimizer.py:110`, `utils/cache_store.py:148`). Les cas
manquants, identifiés :

- `MajaLoader` qui met RH à 50 % faute de mieux ;
- `dedup` qui collapse N états en M, sans le dire ;
- toute extrapolation hors du domaine d'une LUT.

**Pourquoi.** Un `warning` répond à « le résultat est valide, mais ce n'est pas
celui que tu croyais demander ». Si le résultat est inexploitable, c'est une
exception, pas un log, et adjeff respecte déjà cette règle via `AdjeffError`.
Il ne manque que l'étage intermédiaire.

**Vérifier.** Un test par cas, avec `caplog`.

---

### Tâche 7 — Sortie

**Faire.** Changelog. `makefig` du dépôt article appelle
`adjeff.setup_logging(level="info")`. Mesure de silence rejouée et consignée.

**Attention, changement visible.** Le dépôt article ne configure rien
aujourd'hui et reçoit le stdout de structlog tel quel. Après la tâche 1 il ne
recevra plus rien sans `setup_logging()`. C'est le comportement correct pour
une bibliothèque, mais il faut le traiter dans la même version.

---

## Hors périmètre

**Les notebooks.** Les six coupent structlog et pourront enfin retirer cette
redirection, mais ils ne sont pas à jour des modifications des versions 0.8.1 à
0.12.0. Chantier réservé à la 1.0.0.

**Le stdout de Smart-G.** `There is no current context to clear.`, 7 lignes pour
7 appels, écrit directement sur stdout. Aucune configuration de `logging` ne
l'attrape ; seul un `redirect_stdout` autour de l'appel le ferait. À traiter
séparément, si tant est que cela vaille la peine.

---

## Critère de réussite

Trois nombres à comparer aux mesures d'ouverture, sur le même run de référence :

| | 0.12.0 | cible 0.13.0 |
| --- | --- | --- |
| temps en silence de plus d'1 s | 94 % | sous 20 % |
| lignes `info` distinctes | 1 (`"done"`) | une par module et par étape |
| enregistrements `xsweep` visibles | 0 sur 21 | 21 sur 21 |
