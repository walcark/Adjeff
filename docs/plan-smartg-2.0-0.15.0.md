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

`utils/smartgutils.py` existe en grande partie pour convertir un `MLUT`
en `xarray`. Smart-G 2.0 rend directement un `Dataset`, donc
`adapt_smartg_output` et une partie de `collect_batched`, livrés en
0.14.0, perdent leur raison d'être. Smart-G fournit même l'inverse,
`smartg.xarray.dataset_to_mlut`, pour le code qui attend encore un
`MLUT`.

C'est le principal bénéfice de la migration, au-delà de la levée du pin
`geoclide <4`.

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

1. **La convention d'azimut.** `41d81ad` corrige l'azimut du capteur
   satellite en `(vaa + 180) % 360`, établi en mesurant la fonction de
   phase Rayleigh contre l'azimut relatif. Le renommage `PHVDEG` en
   `ph_deg` peut s'accompagner d'un changement de convention : le test
   est à refaire, pas à supposer.

2. **Le contenu du `Dataset` rendu.** Les noms de variables et de
   dimensions conditionnent tout `smartgutils.py`. À relever sur une
   vraie sortie avant de réécrire quoi que ce soit.

3. **Le `th_deg` par défaut du `Sensor` passe de 0 à 180.** Tout appel
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
