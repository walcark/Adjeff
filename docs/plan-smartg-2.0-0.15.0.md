# Migration vers Smart-G 2.0, adjeff 0.15.0

Date : 2026-10-02. Branche : `adapt_smartg2`.
Cible : `smartg 2.0.1`, sur PyPI. L'API est celle de la `2.0.0b1`
relevée ici, la 2.0.1 n'ayant changé que des docstrings et le `th_deg`
par défaut du `Sensor`, passé de 0 à 180.

Smart-G 2.0 n'est pas une montée de version, c'est une réécriture de
l'API publique. Trois changements structurels, puis une longue liste de
renommages.

| Axe | `1.2.0` | `2.0.0b1` |
| --- | --- | --- |
| Retour de `Smartg.run` | `MLUT` | **`xarray.Dataset`** |
| Mots-clés de `run` | `CAPITALES` | **`snake_case`** |
| Découpage en modules | tout dans `smartg.smartg` | `sensor`, `surface`, `albedo`, `atmosphere` |
| `geoclide` | `>=3,<4` | **`>=4,<5`** |
| Distribution | conda-forge | **PyPI seulement**, pour l'instant |

## Ce que ça supprime d'adjeff

`utils/smartgutils.py` convertit un `MLUT` en `xarray` avant de le
normaliser. Smart-G 2.0 rend directement un `Dataset`, donc l'étape de
conversion disparaît. `adapt_smartg_output` et `collect_batched`, eux,
restent : le relevé ci-dessous montre que les noms de variables et de
dimensions sont inchangés, donc le travail de normalisation reste à
faire. Smart-G fournit l'inverse, `smartg.xarray.dataset_to_mlut`, pour
le code qui attend encore un `MLUT`.

Le bénéfice principal est ailleurs : la levée du pin `geoclide <4`, qui
débloque `PsfAtmSampler`.

## Table de correspondance des symboles

| adjeff utilise | `1.2.0` | `2.0.0b1` |
| --- | --- | --- |
| `Smartg` | `smartg.smartg` | `smartg.smartg`, inchangé |
| `Sensor` | `smartg.smartg` | **`smartg.sensor`** |
| `LambSurface` | `smartg.smartg` | **`smartg.surface`** |
| `RTLSSurface` | `smartg.smartg` | **`smartg.surface`** |
| `Environment` | `smartg.smartg` | **`smartg.surface`** |
| `Albedo_cst` | `smartg.water`, `smartg.albedo` | **`AlbedoCst`**, `smartg.albedo` |
| `Albedo_map` | `smartg.smartg` | **`AlbedoMap`**, `smartg.albedo` |
| `AtmAFGL` | `smartg.atmosphere` | **`Atm1D`**, `smartg.atmosphere` |
| `AerOPAC` | `smartg.atmosphere` | inchangé |
| `Entity`, `Plane`, `Transformation` | `smartg.visualizegeo` | **`smartg.objects3d`** |
| `multi_profiles` | `smartg.smartg` | inchangé |

## Table de correspondance des mots-clés de `run`

Relevé sur les 11 appels `.run()` d'adjeff.

| adjeff passe | `2.0.0b1` | Note |
| --- | --- | --- |
| `wl` | `wavelength` | |
| `atm` | `atmosphere` | accepte aussi un `xr.Dataset` |
| `surf` | `surface` | |
| `env` | `environment` | |
| `sensor` | `sensor` | inchangé |
| `le` | `le` | désormais un `LocalEstimate`, le `dict` reste accepté |
| `flux` | `flux` | inchangé |
| `NBPHOTONS` | `n_photons` | |
| `NF` | `n_icdf` | |
| `OUTPUT_LAYERS` | `output_layers` | |
| `THVDEG` | `th_deg` | |
| `PHVDEG` | `ph_deg` | |
| `RMIN` | `r_min` | |
| `RMAX` | `r_max` | |
| `myObjects` | à vérifier | voir `objects3d` |

## Mots-clés de `Sensor`

| `1.2.0` | `2.0.0b1` |
| --- | --- |
| `POSZ` | `pos_z` |
| `LOC` | `loc` |
| `TYPE` | `sensor_type` |
| `FOV` | `fov` |
| `THDEG`, `PHDEG` | `th_deg`, `ph_deg` |

## Changements de signature, pas seulement de nom

`AtmAFGL(atm_filename=..., RH_cst=..., P0=..., tauR=...)` devient
`Atm1D(fname=..., rh_cst=..., p0=..., tau_r=...)` : le premier argument
est positionnel et change de nom.

`Environment(ENV=5, ALB=alb_map)` devient `Environment(env=5,
alb=alb_map)`.

`RTLSSurface` garde son triple `kp=`, celui qu'`05fb0ae` avait dû
utiliser parce que la forme par argument levait une exception en 1.1. À
revérifier : la forme `k0=`, `k1p=`, `k2p=` est peut-être réparée.

## Points à valider par la mesure, pas par la lecture

0. **Les données auxiliaires sont à régénérer.** Constaté le 2026-10-05 :
   le portage du code est complet et les 448 tests hors GPU passent, mais
   les 17 tests d'intégration échouent tous, au même endroit et en 7
   secondes. `AerOPAC` seul suffit à reproduire, sans adjeff :

   ```
   ValueError: The scattering angles must be strictly increasing.
       AerOPAC.native_theta -> union_theta_grid -> as_theta_grid
   ```

   Dans `aerosols/OPAC/mixtures/sulphate_sol.nc` du jeu actuel, `theta`
   va de 180 à 0, donc décroissant : 1999 pas négatifs sur 2000. La 1.2
   l'acceptait, la 2.0 l'interdit.

   `smartg.auxdata.download(data_type="aer")` récupère le jeu au bon
   format. **Le répertoire pointé par `SMARTG_DIR_AUXDATA` est partagé
   avec `adjeff-article-rse`, où il est distribué comme donnée du
   papier** : le mettre à jour en place changerait la donnée publiée.
   Les deux versions doivent coexister le temps que l'article soit
   soumis.

1. ~~**La convention d'azimut.**~~ `41d81ad` corrige l'azimut du capteur
   satellite en `(vaa + 180) % 360`, établi en mesurant la fonction de
   phase Rayleigh contre l'azimut relatif. Le renommage `PHVDEG` en
   `ph_deg` peut s'accompagner d'un changement de convention : le test
   est à refaire, pas à supposer.

2. ~~**Le contenu du `Dataset` rendu.**~~ Relevé le 2026-10-05 sur un
   run minimal : les noms de variables et de dimensions sont **ceux de
   la 1.2**. Seul le conteneur change.

   ```
   TYPE   : Dataset
   DIMS   : Zenith angles 45, Azimuth angles 90, wavelength 2, z_atm 7,
            theta_atm 1801, iphase 2, nphamat 6
   VARS   : I_up (TOA), Q_up (TOA), U_up (TOA), V_up (TOA), N_up (TOA),
            direct transmission, n_atm, T_atm, OD_r, OD_p, OD_g, OD_atm,
            OD_sca_atm, OD_abs_atm, pmol_atm, ssa_atm, phase_atm, ...
   I_up (TOA) dims=('wavelength', 'Azimuth angles', 'Zenith angles')
   ```

   Conséquence : `adapt_smartg_output` garde tout son sens, seule la
   conversion `MLUT.to_xarray()` en amont disparaît.

3. ~~**Le `th_deg` par défaut du `Sensor` passe de 0 à 180.**~~
   Confirmé le 2026-10-05, et c'était bien le dernier défaut : les
   capteurs de `tdif_up` et `tdif_down` ne fixaient pas l'angle, donc ils
   se sont retournés vers le nadir et rendaient `0.0`. Aucun contrôle
   d'API ne pouvait l'attraper, le mot-clé existant dans les deux
   versions ; c'est la réciprocité 6S `tdif_up(θ) ≈ tdif_down(θ)` qui l'a
   révélé. L'angle est désormais explicite aux deux appels.

   **Ancien texte :** Tout appel
   qui ne le fixait pas regarde désormais vers le nadir au lieu du
   zénith. `tdif_up` et `tdif_down` construisent un `Sensor(POSZ=0.0,
   LOC="ATMOS", TYPE=1, FOV=90)` sans angle : ces deux-là changent de
   sens si le défaut n'est pas explicité.

## Ordre de travail

1. Table de correspondance, ce document.
2. Environnement : pin `smartg >=2.0.0b1`, `geoclide >=4,<5`, et le
   passage de conda-forge à PyPI pour smartg.
3. Un relevé de la sortie réelle de `run` sur un cas minimal, écrit dans
   ce document.
4. `atmosphere/`, le moins couplé.
5. `utils/smartgutils.py`, qui rétrécit.
6. `modules/samplers/_smartg.py`, le gros morceau.
7. Les tests d'intégration GPU, qui sont le seul juge.

## Résultat

Migration terminée le 2026-10-05 : **19 tests d'intégration GPU passent**,
448 hors GPU, lint et typage inclus.

Cinq défauts ont été trouvés, dans cet ordre, chacun masqué par le
précédent :

| Défaut | Trouvé par |
| --- | --- |
| `theta` décroissant des espèces CAMS | `AerOPAC` isolé |
| `afgl_exp_h8km.nc` absent du jeu 2.0 | `compute_optical_depth` isolé |
| `Environment(ENV=, ENV_SIZE=, ALB=)` | tests GPU |
| `Sensor(POSX=, POSY=)`, `Entity(TC=)`, `Smartg(obj3D=)` | `tests/test_smartg_api.py` |
| `th_deg` par défaut du `Sensor`, 0 → 180 | réciprocité 6S |

Les trois du milieu ont survécu au renommage automatique parce qu'il ne
couvrait que `run(`, `Sensor(` et `make_sensors(`. `tests/test_smartg_api.py`
compare désormais, par lecture de l'AST, chaque mot-clé passé aux quinze
appelables Smart-G à leur signature réelle : deux secondes au lieu de deux
minutes trente de GPU.

Le dernier n'était attrapable par aucun contrôle d'API, le mot-clé
existant dans les deux versions avec un défaut différent. Seule une
grandeur physique pouvait le révéler.

## Les données auxiliaires, régénérées

Les huit tables CAMS de l'article ont été reconstruites avec pymopsmap
le 2026-10-06, et comparées terme à terme aux originales. Deux défauts de
`to_smartg` sont sortis de cette comparaison, corrigés dans pymopsmap
(`3a0b69b`) :

| Défaut | Effet |
| --- | --- |
| `wav` écrit en micromètres | Smart-G lit ces tables en nanomètres et rejetait la table, « tabulated from 0.25 to 2.25 nm » |
| `stk = 1` au lieu de 6 | Smart-G complète par des zéros : F21, F33, F34 et F44 nuls, donc aucune polarisation transportée |

Après correction, l'accord est à la précision du float32 :

| Grandeur | Écart max |
| --- | --- |
| les six termes de `phase`, rapportés au pic de F11 | 4e-6 |
| `OD_atm`, `OD_p`, `ssa_atm`, `phase_atm` d'un `Atm1D` | 3e-6 |

Reste une différence qui **ne compte pas** : `ext` diffère d'un facteur
qui dépend de l'humidité (8 pour le sulphate entre 0 et 95 %), les deux
jeux ne normalisant pas par la même quantité. `dtau_ssa` ne lit `ext`
qu'à travers `ext_tmp / ext_ref_tmp`, les deux pris à la **même**
humidité, puis mis à l'échelle par `tau_ref` : toute normalisation
multiplicative s'annule. À humidité fixée, la forme spectrale est
identique à 7e-6.
