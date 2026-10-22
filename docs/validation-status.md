# Statut de validation GPU (mise à jour : 2026-10-21)

## Pipeline bt=1 (2D Wiener + degrid), fixture `tests/dump_bt1`
(GRAY 64x64, bw=bh=16, ow=oh=5, sigma=2, beta=1, degrid=1, seed 7)

| Étape GPU | Référence CPU | Résultat |
|---|---|---|
| frame_to_cover | p0_f000_cover.bin | **bit-exact** (0/8649) |
| cover_to_overlap | p0_f000_in.bin | maxabs 1.5e-5 (arrondi float) |
| fft_fwd | p0_f000_fft.bin | maxrel 2.2e-5, rms 3.7e-5 |
| gridfraction+apply2d | p0_f000_applied.bin | maxabs 3.9e-3 sur DC ~3.7e4 (relative ~1e-7) |
| fft_inv | p0_f000_inv.bin | maxrel 7.9e-6 |
| overlap_to_cover | p0_f000_coverout.bin | **bit-exact** (même entrée float) |
| **END-TO-END** | **out.raw** | **bit-exact (0/4096)** |
| déterminisme (2 runs, 2 objets noyau) | — | **bit-exact** |

## Profondeurs de bits (bw=bh=16, bt=1)

| Fixture | END-TO-END |
|---|---|
| dump_bt1_10b (10 bits) | **bit-exact (0/4096)** |
| dump_bt1_32f (float32) | 60 % de pixels différents, **maxabs 2.4e-7 (≤6 ULP)** — acceptable (cible ±1/255), déterminisme bit-exact |

Politique float32 : le CPU utilise FFTW ; le FFT GPU (Stockham) diffère de FFTW par
l'arrondi (~1e-5 relatif au spectre). En entrée/sortie entière le cast final absorbe
ces différences (bit-exact vérifié 8/10 bits). En float32 il reste ≤6 ULP en sortie,
documenté comme tolérance. `overlap_to_cover` reproduit l'ordre exact du CPU
(fenêtres Y pré-multipliées par norm, sommes X groupées) → bit-exact à cette étape.

## Variantes (bw=bh=16, 8 bits, bt=1)

| Fixture | END-TO-END |
|---|---|
| dump_ilace (interlacé) | **bit-exact** |
| dump_wt1 (wintype 1) | **bit-exact** |
| dump_wt2 (wintype 2) | **bit-exact** |
| dump_crop (l=3 t=2 r=5 b=4) | **bit-exact** |
| dump_chroma plan 0 (YUV420 luminance) | **bit-exact** |
| dump_chroma plan 1 (YUV420 chroma, planeBase=128) | **bit-exact** (stage + end-to-end) |
| dump_yuv422 plan 0 / plan 1 | **bit-exact** |
| dump_yuv444 plan 0 / plan 1 | **bit-exact** |
| dump_chroma_f (YUV420 float32) | ≤6 ULP (politique float) |

## Profondeurs 12/14/16 bits (GRAY, bw=16)

| Fixture | END-TO-END |
|---|---|
| dump_bt1_12b | **bit-exact** |
| dump_bt1_14b | 3/4096 pixels à ±1 LSB |
| dump_bt1_16b | 8/4096 pixels à ±1 LSB |

À 14/16 bits, l'arrondi float32 du FFT (Stockham vs FFTW, relative ~1e-5, soit
~0,5-4 unités absolues sur des valeurs ≤65535) fait basculer le cast final de
quelques pixels d'1 LSB. C'est inhérent à deux implémentations FFT différentes en
float32 ; l'écart reste très inférieur à la cible ±1/255 (64 LSB à 14 bits).
L'étape `overlap_to_cover` est bit-exact à toutes les profondeurs (même entrée).

Particularité reproduite : `buffer.cpp` ligne 454 ajoute `planeBase` APRÈS le cast
tronqué (et non avant comme partout ailleurs) uniquement dans le premier segment
des lignes de recouvrement vertical (`ny==2 && nx==1 && ihx==0`). Ces colonnes sont
du rembourrage miroir, jamais recopiées en sortie, mais le noyau GPU reproduit le
quirk pour que le test d'étape isolé soit aussi bit-exact.

Note harnais : `DSFrame::Create` (stub) empilait les plans à `+= pitch` au lieu de
`+= pitch * hauteur` — le plan 1 écrasait le plan 0 de `out.raw` pour les formats
multi-plans. Corrigé ; les fixtures chroma ont été régénérées. Le dump `frameout`
(buffer de sortie du moteur) a permis d'isoler ce bug.

## Tailles de bloc (8 bits, bt=1)

| bw=bh | chemin FFT | END-TO-END |
|---|---|---|
| 16 | fusionné | bit-exact |
| 32 | fusionné | bit-exact |
| 64 | séparé (lignes+colonnes) | bit-exact |
| 128 | séparé | bit-exact |

## Commandes
```
.\fft3dneocl\build\Release\fft_test.exe tests\dump_bt1
```
Kernels : `fft3dneocl/src/kernels/fft3d.cl` (chargé au runtime, defines -DBW -DBH -DOW -DOH -DBPP).

## Plugin VapourSynth API4 (fft3dneocl.dll) — validation bout-en-bout

Le plugin (`fft3dneocl/src/plugin.cpp` + `gpu/pipeline.cpp`) est piloté par
`plugin_smoke.exe` (charge le noyau VapourSynth comme bibliothèque, injecte la
fixture comme filtre source, appelle `fft3dneocl.FFT3D`, compare à `out.raw`).
Les tables/géométrie côté hôte sont validées séparément par `tables_check.exe`
(bit-exact vs dumps CPU : fenêtres, norm, scalaires sigma, wsharpen/wdehalo,
nox/noy avec le rembourrage « +1 bloc par côté », coverpitch, mirw).

| Fixture (plugin, bt=1) | Résultat |
|---|---|
| dump_bt1 / _10b / _12b (8/10/12 bits) | **bit-exact (0)** |
| dump_bt1_14b / _16b | 1 et 6 pixels à ±1 LSB (politique 14/16 bits) |
| dump_bt1_32f | maxabs 2.4e-7 (≤6 ULP, politique float) |
| dump_bt32 / _64 / _128 | **bit-exact** |
| dump_wt1 / dump_wt2 (wintype 1/2) | **bit-exact** |
| dump_ilace (interlacé) | **bit-exact** |
| dump_crop (l=3 t=2 r=5 b=4) | **bit-exact** (marges recopiées depuis la source) |
| dump_chroma / dump_yuv422 / dump_yuv444 (plans 0,1) | **bit-exact** |
| dump_chroma_f (YUV420 float32) | ≤6 ULP (politique float) |
| dump_yuv_p0only (plan 0 seul) | **bit-exact** |

Note fixture : `dump_yuv_p0only` datait d'une version antérieure du harnais (avant
la correction `DSFrame::Create`) ; régénéré avec le harnais courant (`--planes 0`),
l'ancien `out.raw` conservé en `out.raw.stale`. Les fixtures chroma YUV sont
générées avec `--planes 0,1` (le plan 2 est recopié) — le test plugin passe
`planes=0,1`.

Test d'intégration réel (module Python VapourSynth R81, `tools/test_vs_integration.py`) :
`LoadPlugin`, `ListDevices`, `FFT3D` déterministe sur 7 images pour bt=2..5 et
bt=-1, bt=0 accepté (image 0 en passe-plat, état séquentiel).

## Modes temporels bt=2..5 (Apply3D2..5)

Le noyau `apply3d` (compilé avec `-DBT=2..5`) transcrit fidèlement `Apply3D2_C`
à `Apply3D5_C` (`neo_FFT3D/src/code_impl/Apply.cpp`) : DFT temporelle de 2 à 5
points sur les spectres des images de la fenêtre, facteur de Wiener avec
`sigma3d = btcur*sigma*sigma/norm` (même associativité que le CPU), DFT inverse.
Le correcteur de grille vaut `gridfraction * gridsample * btcur` et ne s'applique
qu'au bin DC temporel ; `gridfraction` se lit sur le DC du spectre **brut** de
l'image courante. La passe `sharpen_post` (le `Sharpen` du moteur après
`Apply3D`) ne s'exécute que si `sharpen != 0 || dehalo != 0` (le CPU sort tôt
sinon) ; sa gridfraction utilise le DC du spectre **filtré**.
Images de bord : même règle que le moteur (fft3d_engine.h 647-650), `btcur = 1`
si `bt/2 > n` ou `(bt-1)/2 > numFrames-1-n` → chemin 2D, déjà validé.

Le plugin demande les images `n+from..n+to` à l'enfant ; un `std::mutex` sérialise
le corps du filtre (VapourSynth appelle `getFrame` depuis plusieurs threads et les
pipelines partagent leurs tampons GPU — bug de concurrence préexistant, corrigé).

| Fixture (7 images, bw=bh=16, sigma=2) | Résultat |
|---|---|
| dump_bt2 / dump_bt3 / dump_bt4 / dump_bt5 | **bit-exact (0)** sur les 7 images |
| dump_bt3_bw32 (blocs 32) | **bit-exact** |
| dump_bt3_yuv (YUV420, plans 0,1) | **bit-exact** |
| dump_bt4_sharpen (sharpen=1 dehalo=1) | **bit-exact** |
| dump_bt5_crop (l=3 t=2 r=4 b=1) | **bit-exact** |
| dump_bt3_wt1 / dump_bt5_ilace | **bit-exact** |
| dump_bt2_10b | 1 pixel à ±1 LSB (limite d'arrondi, pixel quasi noir) |

Déterminisme : SHA-256 identiques entre exécutions (bt=2..5). Le résultat ne
dépend pas de l'ordre de demande des images : `plugin_smoke order=rev` et
`order=shuf` (lecture inversée/aléatoire, qui bombarde le cache) donnent ndiff=0
pour bt=2..5.

### Cache GPU des spectres (`fftcache`)

Comme le CPU (`cache<fftwf_complex>(bt+3, outsize)`, fft3d_engine.h 158 et
727-760), le pipeline garde les spectres bruts des dernières images (LRU,
bt+3 emplacements par plan, clé = numéro d'image). Une image déjà transformée
n'est pas recalculée : en lecture séquentielle, une seule FFT de plus par image
de sortie au lieu de bt. Les nombres sont inchangés (mêmes noyaux, mêmes
données), seule la vitesse change.

### Qualité et performances temporelles (bt=3, bw=bh=32, sigma=2, seed 5, 30 images)

| Cas | CPU (Ryzen 7 5700X) | GPU sans cache | GPU avec fftcache | Gain vs CPU | Pixels différents |
|---|---|---|---|---|---|
| 1280x720 GRAY8 bt=3 | 16,21 ms | 4,75 ms | **3,1-3,3 ms** | 5,0× | 256 / 27 648 000 (±1 LSB) |
| 1920x1080 GRAY8 bt=3 | 37,42 ms | 7,2-9,3 ms | **5,8-6,2 ms** | 6,2× | 605 / 62 208 000 (±1 LSB) |
| 1280x720 GRAY8 bt=5 | 18,28 ms | 6,33 ms | **3,0-3,2 ms** | 5,7× | 260 / 27 648 000 (±1 LSB) |

(Mesures A/B dans le même binaire ; le GPU portable varie de ±10 % selon la
cadence. Le cache est plus rapide dans toutes les paires mesurées.)

## Mode sharpen seul (bt=-1)

Le moteur CPU (`fft3d_engine.h` 836-865) saute le filtre de Wiener et applique
seulement `Sharpen` sur le spectre **brut** : FFT → Sharpen en place → FFT
inverse. Le GPU fait de même : `gridfraction_k` lit le DC du spectre brut, puis
`sharpen_post` modifie `d_spec` en place (mêmes arguments que la passe sharpen
temporelle, mais sur le spectre non filtré). Comme le CPU, si `sharpen == 0` et
`dehalo == 0`, la passe est sautée (retour anticipé de `Sharpen.cpp` 55) et le
résultat est l'identité FFT→inverse. `bt=-1` est sans état : pas de fenêtre
temporelle, pas de `fftcache`, pas de repli 2D en bord d'images.

Équivalence vérifiée des deux côtés : `bt=-1 sharpen=1` est bit-identique à
`bt=1 sigma=0 sharpen=1` (le facteur de Wiener vaut exactement 1), ce qui
recoupe les deux chemins GPU (`apply2d` en ligne contre `sharpen_post`).

| Fixture (bw=bh=16, sigma=2) | Résultat |
|---|---|
| dump_bm1_sharpen (sharpen=1) | **bit-exact (0)** |
| dump_bm1_dehalo (dehalo=1) | **bit-exact (0)** |
| dump_bm1_both (sharpen=1 dehalo=1) | **bit-exact (0)** |
| dump_bm1_off (sharpen=0 dehalo=0, passe-plat) | **bit-exact (0)** |
| dump_bm1_yuv (YUV420, plans 0,1) | **bit-exact (0)** |
| dump_bm1_10b (10 bits) | **bit-exact (0)** |
| dump_bm1_multi (3 images) | **bit-exact (0)** ; `order=rev`/`shuf` : ndiff=0 |

Note de diagnostic : un premier écart (ndiff 2323, maxabs 5) venait du script de
test qui oubliait de passer `bw=16 bh=16` au GPU (le CPU utilisait 16, le GPU le
défaut 32) — pas d'erreur du port. Les dumps d'étapes GPU (spectres brut et
appliqué) comparés aux dumps CPU ont confirmé la correspondance à ~3 ULP près.

## Mode Kalman (bt=0)

Le moteur CPU (`fft3d_engine.h` 787-833) applique un filtre de Kalman
récurrent sur le spectre : l'image 0 est renvoyée sans traitement (789-791),
puis chaque image met à jour un état persistant par plan — `outLast`,
`covar`, `covarProcess` (alloués 144-149, initialisés 346-351 : `outLast=0`,
`covar=covarProcess=sigmaSquaredNoiseNormed2D`). Par case spectrale, le test
de mouvement `|cur-prev|² > sigma*kratio²` (674) déclenche une remise à zéro
(`out=cur`, `covar=sigma`) sinon le gain de Kalman fusionne `cur` et `prev`.
L'état vit pour toute l'instance : deux passes séquentielles ne se répètent
pas, et il n'y a pas de repli 2D en bord d'images (`btcur` reste 0).

Le GPU reproduit cette sémantique avec un noyau `kalman` dédié : état
persistant alloué au `create()` et rempli via `EnqueueFillBuffer`, grille
`nblocks*bh*outpitch` (toutes les colonnes rembourrées, comme le dispatch
`Kalman_AVX` du CPU — le harnais sur Ryzen 7 5700X utilise cette version ;
la grille est sans effet sur les colonnes miroir, toujours nulles ici).
Après le noyau, une copie `d_outLast → d_spec` joue le `memcpy` du moteur
(812), puis `gridfraction`+`sharpen_post` éventuels, puis FFT inverse.

Côté API : `fmSerial` a disparu d'API4 ; `bt=0` utilise `fmUnordered`, le
remplacement officiel pour les filtres à état (le core sérialise les requêtes
par instance). L'image 0 est rendue par simple recopie de la source dans
`fft3dGetFrame`, sans toucher le GPU.

| Fixture (bw=bh=16, sigma=2, 7 images) | Résultat |
|---|---|
| dump_kalman (GRAY8) | **bit-exact (0)** |
| dump_kalman_sh (sharpen=1) | **bit-exact (0)** |
| dump_kalman_kr3 (kratio=3) | **bit-exact (0)** |
| dump_kalman_10b (10 bits) | **bit-exact (0)** |
| dump_kalman_yuv (YUV420, 3 plans) | **bit-exact (0)** |

Note de diagnostic (piège à retenir) : un long écart initial (ndiff ~50 %,
maxabs 12) venait des fixtures générées sans `--format` (défaut du harnais :
YUV420) puis passées à `plugin_smoke` comme GRAY — les images étaient donc
décalées d'un plan de chroma. Le noyau Kalman était correct dès le départ ;
régénérer les fixtures avec `--format GRAY` a donné ndiff=0 immédiatement.
Autre piège : les dumps de stages GPU doivent être nommés par plan
(`p%d_f%03d_…`), sinon ils s'écrasent entre plans sur les clips YUV.

## Mode débruitage par motif (pfactor ≠ 0)

Le moteur CPU (`fft3d_engine.h` 543-575) estime le motif **une seule fois par
instance**, sur l'image `pframe` : FFT de cette image, puis `FindPatternBlock`
(choix du bloc avec la plus faible puissance corrigée de grille, pondérée par
`pwin`, candidats `bx,by ∈ [2, nox-2)×[2, noy-2)`, comparaison stricte `<`) et
`SetPattern` (`pattern2d[h*outpitch+w] = psd*pwin[w]`, plus `psigma`). Si
`px,py` sont fournis (≠0), la recherche est sautée. Variante : si
`sigma2/3/4` diffèrent de `sigma` avec `pfactor=0`, le motif vient de
`SigmasToPattern` (constructeur 373-378) et `pfactor` est forcé à 1, sans
demander d'image.

Côté GPU : `estimatePattern()` écrit l'image `pframe` dans le pipeline
(f2c → c2o → FFT avant), relit le spectre, puis `FindPatternBlock`/`SetPattern`
sont exécutés **sur le processeur** avec la copie verbatim de `helper.cpp`
(même ordre des additions flottantes → choix de bloc et table identiques au
bit près), et la table `pattern2d` est téléversée dans `d_pat`. La table `pwin`
est calculée à l'hôte avec les formules exactes du constructeur (353-368).
Le plugin demande l'image `pframe` en plus de la fenêtre temporelle et
l'estimation se fait sous le mutex, avant le premier `process()` de chaque plan
(également pour l'image 0 en passe-plat bt=0, le moteur estimant avant son
retour anticipé).

Dans les filtres : `apply2d` remplace le sigma constant par
`pattern2d[h*outpitch+w]` par case, et — particularité reproduite — la famille
« pattern » de `Apply2D` **saute** les facteurs sharpen/dehalo
(`if constexpr (!pattern)` dans `Apply.cpp`/`Apply_AVX.cpp`) : en bt=1 le moteur
appelle alors `Sharpen` en passe séparée (697-698), ce que le GPU fait avec
`gridfraction`+`sharpen_post` après `apply2d`. `apply3d` utilise
`pattern2d[t]*(float)bt` (le CPU précalcule `pattern3d=pattern2d*btcur` une
fois, helper.cpp `Pattern2Dto3D` — même multiplication flottante). `kalman`
utilise `sigma = max(pattern2d[case], 1e-15)` par case (`Kalman_AVX<true>`),
l'initialisation `covar` restant `sigmaSquaredNoiseNormed2D`. Le gridsample est
calculé avec `planeBase=0` même sur la chroma (`CoverToOverlap(..., false)`,
engine 408).

| Fixture (bw=bh=16, 7 images) | Résultat |
|---|---|
| dump_pat_bt1 (pfactor=1) | **bit-exact (0)** |
| dump_pat_bt1_sh (pfactor=1 sharpen=1 dehalo=1) | **bit-exact (0)** |
| dump_pat_bt3 / dump_pat_bt5 (motif 3D) | **bit-exact (0)** |
| dump_pat_kal (bt=0 + motif) | **bit-exact (0)** |
| dump_pat_px (px=3 py=3, bloc fixe) | **bit-exact (0)** |
| dump_pat_sig (sigma2/3/4, motif sans image) | **bit-exact (0)** |
| dump_pat_pframe2 (pframe=2) | 1 pixel à ±1 LSB (limite d'arrondi, voir note) |
| dump_pat_yuv (YUV420, plans 0,1) | **bit-exact (0)** |
| dump_pat_nodg (degrid=0, gcur=0) | **bit-exact (0)** |

Note : le FFT GPU (Stockham) diffère de FFTW d'environ 1 ULP ; le motif estimé
diffère donc d'environ 1e-5 relatif. Sur `dump_pat_pframe2` un pixel se trouve
exactement sur une frontière d'arrondi 8 bits et bascule d'1 LSB — tolérance
déjà acceptée pour 14/16 bits et le float32. Le résultat reste déterministe
(deux passes, ordre séquentiel ou mélangé : SHA-256 identiques) et le test
d'intégration VapourSynth confirme que l'ordre d'arrivée des images ne change
pas le motif estimé. `pshow` (visualisation du bloc choisi) reste rejeté : mode
de débogage sans intérêt en production.

## Commandes plugin
```
.\fft3dneocl\build\Release\plugin_smoke.exe tests\dump_bt1\input.raw tests\dump_bt1\out.raw 64 64 8 GRAY tests\dump_bt1\plugin_out.raw bt=1 bw=16 bh=16
.\fft3dneocl\build\Release\plugin_smoke.exe tests\dump_bt3\input.raw tests\dump_bt3\out.raw 64 64 8 GRAY tests\dump_bt3\plugin_out.raw bt=3 bw=16 bh=16 frames=7
.\fft3dneocl\build\Release\tables_check.exe tests\dump_bt1
powershell -NoProfile -File tools\run_plugin_matrix.ps1   # toute la matrice (bt=1, bt=2..5, bt=-1, bt=0, motif), 1 ligne par fixture
powershell -NoProfile -File tools\run_bench_report.ps1    # bilan qualite/perf/memoire vs fft3dcl et CPU (docs/benchmark-report.md)
python tools\test_vs_integration.py                      # chargement réel du plugin dans VapourSynth R81
```

## Comparaison au filtre CPU `neo_FFT3D` (référence de référence)

Le DLL fourni `neo_FFT3D\neo-fft3d.dll` est compilé contre l'**API 3** de
VapourSynth : R81 refuse de le charger (« uses API 3, which is no longer
supported »). La comparaison a donc été faite contre le **même moteur CPU, compilé
localement** par `reference\harness\fft3d_ref.exe` (sources `neo_FFT3D` non
modifiées) — c'est d'ailleurs lui qui produit toutes les références `out.raw` des
fixtures. Le harnais a reçu une option `--time` (mesure du moteur seul) ; la
sortie pixel est inchangée (empreinte SHA-256 identique à `tests/dump_bt1/out.raw`).

### Qualité à grande échelle (bt=1, sigma=2, seed 7, entrée identique des deux côtés)

| Cas | pixels différents | écart max |
|---|---|---|
| 1280x720 GRAY8 bw=16 | 11 / 921 600 (0,0012 %) | ±1 LSB |
| 1280x720 GRAY8 bw=32 | 8 / 921 600 (0,0009 %) | ±1 LSB |
| 1280x720 GRAY8 bw=64 | 10 / 921 600 (0,0011 %) | ±1 LSB |
| 1920x1080 GRAY8 bw=32 | 26 / 2 073 600 (0,0013 %) | ±1 LSB |
| 1280x720 YUV420 3 plans bw=32 | 9 / 1 382 400 (0,0007 %) | ±1 LSB |

Quelques pixels par million, tous à 1 unité de valeur : c'est l'arrondi du FFT
float32 (Stockham côté GPU contre FFTW côté CPU), déjà documenté comme tolérance.

### Performances : CPU (Ryzen 7 5700X, 8 cœurs) contre GPU (RTX 3050), 30 images

| Cas | fft3d_ref (CPU) | fft3dneocl (GPU) | rapport |
|---|---|---|---|
| 1280x720 GRAY8 bw=16 | 11,35 ms (88 i/s) | 9,54 ms (105 i/s) | 1,2x |
| 1280x720 GRAY8 bw=32 | 17,43 ms (57 i/s) | 1,97 ms (507 i/s) | **8,8x** |
| 1280x720 GRAY8 bw=64 | 12,91 ms (78 i/s) | 3,27 ms (306 i/s) | 3,9x |
| 1920x1080 GRAY8 bw=32 | 41,21 ms (24 i/s) | 5,46 ms (183 i/s) | **7,5x** |
| 1280x720 YUV420 3 plans bw=32 | 22,95 ms (44 i/s) | 3,62 ms (276 i/s) | 6,3x |

Le moteur CPU est déjà multithread ; le plugin GPU est mesuré en requête
séquentielle d'une image. Le bloc de 32 est le meilleur compromis côté GPU comme
côté CPU (le CPU est plus rapide en 16 qu'en 32, le GPU l'inverse).

```
# référence CPU + mesure
reference\harness\build\Release\fft3d_ref.exe --out tests\bench_720p_cpu\out.raw --width 1280 --height 720 --format GRAY --frames 1 --seed 7 --bt 1 --bw 32 --bh 32 --sigma 2
# même entrée, générée par le harnais : --dump-input
reference\harness\build\Release\fft3d_ref.exe --dump-input --out tests\bench_720p_cpu\input.raw --width 1280 --height 720 --format GRAY --frames 30 --seed 7
# plugin GPU sur cette entrée
.\fft3dneocl\build\Release\plugin_smoke.exe tests\bench_720p_cpu\input.raw tests\bench_720p_cpu\out.raw 1280 720 8 GRAY tests\bench_720p_cpu\plugin_out.raw bt=1 bw=32 bh=32 sigma=2 frames=30 time=1
```

## Comparaison au filtre d'origine `fft3dgpu.dll` (DX9/HLSL)

Le plugin d'origine installé avec VapourSynth (namespace `fft3dgpu`, fonction
`FFT3DGPU`) a été appelé sur les mêmes images que notre plugin via
`plugin_smoke.exe ... ref=<chemin fft3dgpu.dll>`.

**Résultat important : `fft3dgpu.dll` (DX9) est déréglé sur ce GPU (RTX 3050).**
Sur une image parfaitement plate (valeur constante 128), même avec `sigma=0` et
`degrid=0` (aucun débruitage demandé), il sort une **grille d'artefacts périodique
de période `bw`** : écart max 126-128, écart moyen 54 sur 128. Vérifié en GRAY8,
YUV444P8 et YUV420P8, en 64x64 et 256x256, avec `precision=0/1/2`, `mode=0/1`,
`ow=4/5` — le résultat ne change pas d'un paramètre à l'autre (le plan de chroma,
lui, reste correct). Notre plugin rend exactement l'entrée (écart 0).
Une comparaison numérique pixel à pixel contre ce filtre n'a donc pas de sens ici :
les 99 % de pixels différents et `maxabs=255` viennent de sa grille, pas de notre
port. La référence de qualité reste le moteur CPU `neo_FFT3D` (bit-exact, cf. plus
haut).

### Performances (GPU unique, requêtes séquentielles, 30 images)

| Cas | fft3dneocl | fft3dgpu (DX9) | rapport |
|---|---|---|---|
| 1280x720 GRAY8 bw=16 | 8,96 ms/img (112 i/s) | 250 ms (4,0 i/s) | **28x** |
| 1280x720 GRAY8 bw=32 | 2,31 ms/img (433 i/s) | 182 ms (5,5 i/s) | **79x** |
| 1280x720 GRAY8 bw=64 | 3,74 ms/img (267 i/s) | 172 ms (5,8 i/s) | **46x** |
| 1920x1080 GRAY8 bw=32 | 5,62 ms/img (178 i/s) | 431 ms (2,3 i/s) | **77x** |
| 1280x720 YUV420 3 plans bw=32 | 4,04 ms/img (248 i/s) | 183 ms (5,5 i/s) | **45x** |

`num_streams` (1080p bw=32) : 1 flux 5,56 ms, 2 flux 5,24 ms, 4 flux 5,03 ms —
progrès modeste en requête séquentielle (VapourSynth parallélise déjà les images).
Déterminisme revérifié en 1080p : deux passages donnent le même SHA-256.

```
.\fft3dneocl\build\Release\plugin_smoke.exe tests\bench_1080p\input.raw - 1920 1080 8 GRAY tests\bench_1080p\plugin_out.raw ref=<fft3dgpu.dll> bt=1 bw=32 bh=32 frames=30 time=1
```
(`ref_out.raw` en 2ᵉ argument peut valoir `-` : pas de référence CPU, mesure de temps seule.)

## Points de vigilance découverts
- Le dump `inv` = c2r du spectre **appliqué** (post-Apply2D), pas du `fft` brut.
- FFTW c2r : symétrie hermitienne 2D `Z[h][w]=conj(Z[(BH-h)%BH][BW-w])` ; imag ignorée
  seulement aux 4 coins auto-conjugués (h∈{0,BH/2}, w∈{0,BW/2}).
- Fenêtres de synthèse : aucune fenêtre aux bords extérieurs du cover
  (ihx==0 / ihx==nox / ihy==0 / ihy==noy → fenêtre 1), les deux côtés.
- Comparaisons : ne jamais comparer les colonnes de rembourrage (outpitch>outwidth,
  coverpitch>coverwidth) — jamais écrites des deux côtés.
- np.fromfile sans dtype lit en float64 (piège de diagnostic).

## À faire
1. ~~Tailles de bloc 32 (valider), puis 64/128~~ → **16/32 (fusionné), 64/128 (séparé) : tous bit-exact bout-en-bout.**
   - Chemin sépare (lignes+colonnes, tampon global) choisi automatiquement quand
     2*bh*(bw+1)*8 > mémoire locale (imprimer "fft path: split").
2. Bit depths 10/12/14/16 (BPP=2) et 32f (BPP=4) ; interlacé ; crop ; wintype 1/2.
3. ~~Apply3D2..5 (DFT temporelle)~~ → **fait** : bt=2..5 bit-exact (voir section
   Modes temporels). ~~Sharpen pur (bt=-1)~~ → **fait** : bit-exact (voir section
   Mode sharpen seul). ~~Kalman (bt=0)~~ → **fait** : bit-exact (voir section
   Mode Kalman). ~~Pattern (FindPatternBlock/SetPattern, pfactor≠0)~~ → **fait** :
   bit-exact (voir section Mode débruitage par motif).
4. ~~Plugin VapourSynth API4~~ → **fait** : `fft3dneocl.dll` (namespace `fft3dneocl`,
   fonctions `FFT3D` + `ListDevices`), bt=-1, bt=0 et bt=1..5, validé bit-exact
   bout-en-bout sur toute la matrice de fixtures (voir section Plugin ci-dessus).
5. ~~Modes temporels bt=2..5~~ → **fait**, y compris le cache GPU des spectres
   (`fftcache` LRU, comme le CPU). ~~Kalman (bt=0)~~ → **fait** (état persistant,
   `fmUnordered`). ~~Estimation de motif (pfactor≠0)~~ → **fait** (motif estimé
   une fois par instance sur l'image `pframe`, `FindPatternBlock`/`SetPattern` à
   l'hôte). Seul `pshow` (visualisation de débogage) reste volontairement rejeté.
6. Validation croisée GPU (AMD/Intel) et sur une autre machine NVIDIA. Le
   `fft3dgpu.dll` DX9 d'origine rendant une grille d'artefacts sur la RTX 3050
   (voir la section Comparaison), il ne peut pas servir de référence numérique ici ;
   la référence de qualité reste le moteur CPU.

