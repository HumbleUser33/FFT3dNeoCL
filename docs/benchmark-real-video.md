# Comparatif réel — fft3dneocl vs fft3dcl vs neo-fft3d (CPU) sur vidéo 4K

(2026-10-21. Machine : Ryzen 7 5700X, RTX 3050 Laptop 6 Go, 32 Go RAM,
VapourSynth R81. Source : `D:\Test Monkeys\Sample.mkv` (HEVC 3840×2160,
YUV420P10, 1142 images). Deux séquences de 60 images : à partir des images
171 et 419. Scripts : `tools/bench_dump.py`, `tools/run_real_bench.ps1`,
`tools/bench_compare.py`.)

## Les trois comparés

| Filtre | Ce que c'est | Comment il est appelé |
|---|---|---|
| **neo-fft3d (CPU)** | moteur `neo_FFT3D` compilé localement (`fft3d_ref.exe`, nouvelle option `--in` pour lire des images réelles) | `fft3d_ref --in seq.raw ... --time` |
| **fft3dneocl (GPU)** | notre port OpenCL de neo-FFT3D (`fft3dneocl.dll`) | `plugin_smoke ... time=1` |
| **fft3dcl (GPU)** | port OpenCL de **FFT3DGPU** v0.8.6 (`FFT3DGPU VS\build\Release\fft3dcl.dll`), GPU forcé (`device=1`), sans notre plugin (`noplugin=1`) | `plugin_smoke ... ref=... refns=fft3dcl` |

Deux mises en garde de départ :

- **`neo-fft3d.dll` ne peut pas être testé tel quel** : il est compilé contre
  l'API 3 de VapourSynth, que R81 refuse (vérifié : « uses API 3, which is no
  longer supported »). Le côté CPU est donc mesuré avec `fft3d_ref.exe`,
  **le même moteur** `neo_FFT3D` compilé localement — c'est aussi notre
  référence de vérité terrain depuis le début du projet.
- **fft3dcl n'implémente pas le même filtre** que neo-FFT3D (lignée FFT3DGPU :
  recouvrement `bw/2` au lieu de `bw/3`, autres formules Wiener/degrid,
  `bt` limité à -1..4). Ses écarts de qualité mesurent une *différence
  d'algorithme*, pas une erreur.

## Les 5 presets (mêmes réglages partout, bw=bh=32, ow=oh=16, beta=1)

| Preset | Réglages |
|---|---|
| P1 léger | sigma=1,5, bt=3, tous plans |
| P2 moyen | sigma=3,0, bt=3, tous plans |
| P3 poussé | sigma=4,0, bt=3, sharpen=1,0, tous plans |
| P4 bt=5 | sigma=3,0, **bt=5**, tous plans (fft3dcl ne connaît pas bt=5) |
| P5 deux passes | passe 1 : sigma=2,5, bt=3, luma+chroma ; passe 2 : sigma=2,5, bt=3, luma seule |

## Stabilité

- **Déterminisme GPU : 100 %.** Chaque cas GPU est lancé deux fois de suite
  dans des processus séparés ; le hash MD5 de la sortie complète (1,49 Go) est
  identique entre les deux passages, pour les 10 cas × 2 filtres GPU.
- **Aucun plantage** sur les 30 lancements. fft3dcl refuse bt=5 proprement
  (« valid modes for bt are -1,0,1,2,3,4 »).
- **Reproductibilité inter-processus** : le preset P5 relancé plus tard dans
  d'autres processus donne des métriques de qualité rigoureusement identiques.
- Le banc d'essai a dû être corrigé : VapourSynth mettait les images en cache
  entre la boucle de récupération et la boucle chronométrée. `plugin_smoke`
  appelle maintenant `clearCoreCaches` avant chaque mesure. (Les chiffres de
  vitesse de l'ancien rapport synthétique `benchmark-report.md` étaient donc
  trop optimistes ; la qualité et la mémoire n'étaient pas concernées.)

## Performances (ms/image, 60 images, 2e passage GPU chaud, requêtes séquentielles)

### Séquence à partir de l'image 171

| Preset | neo-fft3d CPU | fft3dneocl GPU | fft3dcl GPU | neocl vs CPU | neocl vs cl |
|---|---|---|---|---|---|
| P1 léger | 242,0 | **55,6** | 66,4 | 4,4× | 1,19× |
| P2 moyen | 271,3 | **53,8** | 64,6 | 5,0× | 1,20× |
| P3 poussé | 314,2 | **60,3** | 65,8 | 5,2× | 1,09× |
| P4 bt=5 | 315,2 | **67,4** | non supporté | 4,7× | — |
| P5 2 passes | 458,6 | **103,7** (57,7+46,0) | 117,6 (66,8+50,7) | 4,4× | 1,13× |

### Séquence à partir de l'image 419

| Preset | neo-fft3d CPU | fft3dneocl GPU | fft3dcl GPU | neocl vs CPU | neocl vs cl |
|---|---|---|---|---|---|
| P1 léger | 272,2 | **56,0** | 67,7 | 4,9× | 1,21× |
| P2 moyen | 240,7 | **50,7** | 57,3 | 4,7× | 1,13× |
| P3 poussé | 273,1 | **52,8** | 57,6 | 5,2× | 1,09× |
| P4 bt=5 | 291,9 | **57,2** | non supporté | 5,1× | — |
| P5 2 passes | 411,0 | **106,2** (58,5+47,7) | 118,9 (67,1+51,7) | 3,9× | 1,12× |

À 4K, `fft3dneocl` est **régulièrement 9 à 21 % plus rapide que fft3dcl**
(moyenne ~14 %) et **4 à 5 fois plus rapide que le moteur CPU** multithread.
Le gain sur fft3dcl vient surtout du cache de spectres temporels (`fftcache`)
et du recouvrement bw/3 (moins de blocs que le bw/2 de fft3dcl).

## Empreinte mémoire (pic de processus)

### RAM système (DRAM)

| Preset | seq 171 : CPU / neocl / cl | seq 419 : CPU / neocl / cl |
|---|---|---|
| P1 | 3 049 / 5 706 / 8 485 | 3 049 / 5 910 / 8 939 |
| P2 | 3 049 / 5 766 / 8 266 | 3 049 / 5 337 / 7 999 |
| P3 | 3 056 / 5 157 / 8 697 | 3 049 / 5 234 / 8 872 |
| P4 | 3 476 / 5 720 / — | 3 468 / 5 284 / — |
| P5 | 3 053 / 5 870 / 8 991 | 3 049 / 6 058 / 8 420 |

Les deux bancs GPU gardent en RAM les mêmes tampons d'entrée et de sortie
(≈ 3 Go : 60 images d'entrée + 60 images de sortie à 24,9 Mo), communs aux
deux colonnes. À conditions identiques, **fft3dcl garde 2,5 à 3 Go de RAM de
plus que nous** : il conserve des copies float pleine image côté processeur
plus un cache de spectres côté hôte. Le CPU (`fft3d_ref`, banc séparé, images
également chargées) est le plus sobre.

### VRAM GPU (delta nvidia-smi global, pic)

| Preset | seq 171 : neocl / cl | seq 419 : neocl / cl |
|---|---|---|
| P1 | 2 451 / 1 556 | 2 440 / 1 543 |
| P2 | 2 440 / 1 554 | 2 439 / 1 555 |
| P3 | 2 439 / 1 555 | 2 439 / 1 554 |
| P4 (bt=5) | 2 851 / — | 2 863 / — |
| P5 (p1 → p2) | 2 440 → 1 540 / 1 558 → 1 100 | 2 439 → 1 538 / 1 545 → 1 046 |

C'est le seul point où **fft3dcl est plus économe : ~900 Mo de VRAM de moins**.
Explication : notre cache `fftcache` garde bt+3 spectres complets par plan
(comme le CPU), soit 2 142 MiB de tampons déclarés à 4K tous plans
(1 415,9 MiB luma + 2 × 363,2 MiB chroma) ; fft3dcl ne garde que le spectre
courant et travaille en float hôte (d'où sa RAM plus grosse — le choix est
déplacé, pas supprimé). Nos tampons sont fixes, alloués une fois : pas de
ballottement d'un cas à l'autre, et la passe 2 (luma seule) retombe à ~1,5 Go.
Sur les 6 Go de la RTX 3050, les deux passent largement ; à 4K avec bt=5 nous
sommes à ~2,9 Go, la marge reste confortable.

## Qualité de rendu (60 images = 62,2 M d'échantillons, vs référence CPU neo-FFT3D)

| Preset | fft3dneocl vs CPU | fft3dcl vs CPU |
|---|---|---|
| P1 (seq 171 / 419) | 0,0011 % / 0,0011 %, maxabs 1, PSNR 109,7 / 109,8 | 66,9 % / 69,7 %, maxabs 19 / 20, PSNR 56,1 / 54,9 |
| P2 | 0,0011 % / 0,0011 %, maxabs 1, PSNR 109,7 / 109,9 | 71,5 % / 75,7 %, maxabs 27 / 33, PSNR 53,6 / 50,9 |
| P3 | 0,0011 % / 0,0011 %, maxabs 1, PSNR 109,7 / 109,8 | 73,0 % / 79,0 %, maxabs 70 / 87, PSNR 51,6 / 47,0 |
| P4 | 0,0011 % / 0,0011 %, maxabs 1, PSNR 109,8 / 109,8 | non supporté |
| P5 (2 passes) | 0,008 % / 0,010 %, maxabs 1, PSNR 101,1 / 100,3 | 71,9 % / 77,6 %, maxabs 33 / 46, PSNR 52,4 / 48,8 |

- **fft3dneocl : conforme au CPU.** Environ 1 pixel pour 100 000 diffère,
  toujours d'exactement 1 unité de valeur sur 1024 (l'arrondi du FFT float32
  Stockham contre FFTW), PSNR ≈ 110 dB. En deux passes l'écart monte à
  ~1 pixel pour 10 000 (les deux passes cumulent les arrondis) mais
  **l'écart maximal reste 1 unité** — aucune dérive, aucun artefact.
- **fft3dcl : un autre filtre.** 67 à 79 % des échantillons diffèrent de la
  référence CPU, jusqu'à 87 unités en P3 (le `sharpen=1` de FFT3DGPU n'a pas
  la même formule que celui de neo-FFT3D). Son image n'est pas « fausse »,
  mais elle ne peut pas être rendue identique à neo-FFT3D.
- La comparaison directe fft3dneocl vs fft3dcl donne exactement les mêmes
  chiffres que « fft3dcl vs CPU » (puisque fft3dneocl ≈ CPU) — cohérence
  interne du banc.

## Qualité : écart à l'image d'origine (agressivité du débruitage)

Même mesure, mais chaque sortie est comparée à la **source non filtrée**
(`tools/run_vs_orig.ps1`, résultats `tests\bench_real\vs_orig.json`).
PSNR élevé = l'image reste proche de l'original (débruitage prudent) ;
MAE = valeur moyenne déplacée par pixel. Attention : sans image de
référence « parfaite », ces chiffres mesurent *jusqu'où le filtre modifie
l'image*, pas laquelle des deux images est la plus propre à l'œil.

| Preset | seq 171 : PSNR / MAE (CPU — neocl — cl) | seq 419 : PSNR / MAE (CPU — neocl — cl) |
|---|---|---|
| P1 léger | 51,2 — 51,2 — 49,1 / 2,00 — 2,00 — 2,60 | 50,2 — 50,2 — 48,1 / 2,18 — 2,18 — 2,82 |
| P2 moyen | 46,3 — 46,3 — 44,6 / 3,60 — 3,60 — 4,40 | 45,1 — 45,1 — 43,0 / 4,08 — 4,08 — 5,20 |
| P3 poussé | 44,9 — 44,9 — 43,3 / 4,25 — 4,25 — 5,05 | 43,8 — 43,8 — 41,5 / 4,71 — 4,71 — 6,13 |
| P4 bt=5 | 46,3 — 46,3 — n/a / 3,60 — 3,60 — n/a | 45,1 — 45,1 — n/a / 4,09 — 4,09 — n/a |
| P5 2 passes | 44,9 — 44,9 — 43,3 / 4,25 — 4,25 — 5,09 | 43,6 — 43,6 — 41,4 / 4,88 — 4,88 — 6,32 |

- **fft3dneocl et le CPU sont indiscernables** ici aussi (mêmes PSNR/MAE au
  centième près) : le port GPU déplace l'image exactement comme le CPU.
- **À sigma égal, fft3dcl modifie plus l'image que neo-FFT3D** : 1,5 à 2,3 dB
  de PSNR en moins, MAE 25 à 30 % plus élevé, et des écarts maximaux plus
  gros (jusqu'à 138 niveaux contre 116 pour nous en P3/seq 419). Ce n'est pas
  un défaut mesuré par hasard : « sigma=3 » ne veut pas dire la même chose
  dans les deux algorithmes (recouvrement et formules différents). À régler
  séparément si l'on veut comparer des forces équivalentes, il faudrait
  ajuster le sigma de fft3dcl (environ sigma×0,85 chez lui ≈ sigma chez nous
  d'après ces courbes).
- L'ordre d'agressivité est le même pour les deux moteurs : P1 le plus doux,
  P2/P4 au milieu, P3 et P5 les plus appuyés — les presets restent comparables
  d'un bord à l'autre.
- Les plans Y, U, V bougent de façon équilibrée chez les deux (PSNR par plan
  à moins de 0,2 dB d'écart) : pas de chroma « écrasée » d'un côté ou de
  l'autre.

## Images pour jugement visuel

`tools/bench_export_png.py` exporte, pour chaque séquence et chaque preset,
l'image 30 de la séquence (image absolue 201 ou 449) en 4 versions, dans
`tests\bench_real\png\` :

- `seqXXX_PN_fYYY_grille.png` — grille 2×2 étiquetée : en haut **Original**
  et **neo-fft3d CPU**, en bas **fft3dneocl GPU** et **fft3dcl** (panneaux
  réduits en 1080p pour la vue d'ensemble) ;
- `seqXXX_PN_fYYY_recadrage.png` — même grille mais en **recadrage 512×512 à
  la résolution native**, choisi automatiquement comme la zone où fft3dneocl
  et fft3dcl divergent le plus (jugement des détails fins) ;
- `seqXXX_PN_fYYY_{orig,cpu,neocl,cl}[_crop].png` — les images seules,
  pleine résolution, pour zoomer dans une visionneuse.

À l'œil, les panneaux CPU et fft3dneocl sont indiscernables (conforme au
±1 LSB mesuré) ; les différences visibles se lisent Original vs fft3dcl :
le grain est partout atténué par les trois filtres, fft3dcl lisse
légèrement plus fort à sigma égal (cohérent avec le MAE +25 à 30 %).

```
python tools\bench_export_png.py        # frame 30 par defaut
python tools\bench_export_png.py 10     # une autre frame de la sequence
```

`tools/bench_crop_compare.py <seq> <image_absolue> <x0> <y0> <taille> [P1,P3]`
produit la meme grille 2×2 sur un recadrage a position fixe, utile pour juger
une zone precise (exemple : le cadran « 87645 » de l'image 174,
`seq171_P*_f174_zone.png`, 768×768 en x=466 y=996).

## Commentaire en français simple

- **Stabilité** : rien n'a planté en 30 lancements ; nos sorties GPU sont
  identiques au bit près d'un lancement à l'autre (hash MD5), ce que le
  cahier des charges exigeait.
- **Vitesse** : à 4K, notre plugin est le plus rapide des trois, ~14 % devant
  fft3dcl et 4 à 5 fois devant le CPU.
- **Mémoire** : nous utilisons 2,5 à 3 Go de RAM de moins que fft3dcl, mais
  ~900 Mo de VRAM de plus (notre cache de spectres sur la carte, le sien dans
  la RAM). Les deux passent sans souci sur 6 Go de VRAM.
- **Qualité** : notre plugin reproduit le filtre CPU d'origine (1 unité de
  valeur sur un pixel sur 100 000). fft3dcl est un autre algorithme : il ne
  peut pas et ne rend pas la même image. Face à l'original, fft3dcl déplace
  plus les pixels que neo-FFT3D à sigma égal (MAE +25 à 30 %) — ses réglages
  ne sont pas interchangeables avec les nôtres.
- **Fonctions** : bt=5 n'existe que chez nous et au CPU ; fft3dcl s'arrête à
  bt=4.

## Reproduire

```
python tools\bench_dump.py                      # extrait les 2 séquences 4K (tests\bench_real\seq_*.raw)
powershell -File tools\run_real_bench.ps1       # toute la matrice (2 séquences x 5 presets x 3 moteurs)
powershell -File tools\run_real_bench.ps1 -Presets 5          # un seul preset (1..5)
powershell -File tools\run_real_bench.ps1 -Seqs 171           # une seule séquence
python tools\bench_compare.py A.raw B.raw 3840 2160 10 60 1 1  # qualité entre deux sorties brutes
powershell -File tools\run_vs_orig.ps1         # écart de chaque sortie a l'original (vs_orig.json)
```

Sorties brutes et JSON : `tests\bench_real\` (results.json = dernier run,
results_p1p4.json = passages P1-P4, logs `bench_log*.txt`). Les images de
test 5 images (`smoke_*.raw`) peuvent être supprimées sans conséquence.

## Nouveaux outils ajoutés pour ce comparatif

- `fft3d_ref.exe --in FILE` : le harnais CPU lit désormais de vraies images
  (avant : images synthétiques uniquement).
- `plugin_smoke.exe` : `clearCoreCaches` avant la boucle chronométrée
  (compilé en API 4.3) — sans cela on mesurait des caches VapourSynth, pas le
  filtre.
- `tools/bench_dump.py`, `tools/bench_compare.py`, `tools/run_real_bench.ps1`.
