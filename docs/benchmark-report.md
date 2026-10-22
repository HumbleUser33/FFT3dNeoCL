# Bilan qualité / performances / mémoire — fft3dneocl vs fft3dcl vs neo-fft3d CPU

(2026-10-21. Machine : Ryzen 7 5700X, RTX 3050 Laptop 6 Go, VapourSynth R81.
Script : `tools/run_bench_report.ps1`. Entrées identiques des trois côtés
(fixtures `tests/bench_*`, GRAY8, bw=bh=32, sigma=2, seed 5, 30 images).)

> **Note (même jour)** : la section Performances ci-dessous a été mesurée avant
> la correction `clearCoreCaches` de `plugin_smoke` — les 30 images tenaient
> dans le cache VapourSynth et étaient en partie resservies, donc ces ms/image
> sont trop optimistes. La comparaison sur vidéo réelle 4K
> (`docs/benchmark-real-video.md`) utilise la méthode corrigée et fait foi
> pour la vitesse. Qualité et mémoire ne sont pas concernées.

## Les trois comparés

| Filtre | Ce que c'est | Comment il est appelé |
|---|---|---|
| **neo-fft3d (CPU)** | moteur `neo_FFT3D` compilé localement (`fft3d_ref.exe`), multithread, mesure du moteur seul | `fft3d_ref --time` |
| **fft3dneocl (GPU)** | notre port OpenCL de neo-FFT3D (`fft3dneocl.dll`) | `plugin_smoke ... time=1` |
| **fft3dcl (GPU)** | port OpenCL de **FFT3DGPU** v0.8.6 pour VapourSynth (`FFT3DGPU VS\build\Release\fft3dcl.dll`), GPU forcé (`device=1`), sans notre plugin (`noplugin`) | `plugin_smoke ... ref=... refns=fft3dcl` |

Attention de départ : **fft3dcl n'implémente pas le même filtre** que neo-FFT3D.
C'est la lignée FFT3DGPU : recouvrement par défaut `ow = bw/2` (neo : `bw/3`),
formules de Wiener/degrid légèrement différentes, `bt` limité à -1..4 (**pas de
bt=5**), mais Kalman (bt=0) et sharpen seul (bt=-1) implémentés. La comparaison
de qualité ci-dessous mesure donc une *différence d'algorithme*, pas une erreur.

## Qualité (pixels différents vs la référence CPU neo-FFT3D, 30 images)

| Cas | fft3dneocl vs CPU | fft3dcl vs CPU |
|---|---|---|
| 720p bt=1 | **278 / 27 648 000 (0,001 %), maxabs 1** | 19 566 240 (70,8 %), maxabs 9 |
| 720p bt=3 | **256 / 27 648 000 (0,001 %), maxabs 1** | 19 537 387 (70,7 %), maxabs 7 |
| 720p bt=5 | **260 / 27 648 000 (0,001 %), maxabs 1** | *bt=5 non supporté* |
| 1080p bt=3 | **605 / 62 208 000 (0,001 %), maxabs 1** | 43 970 905 (70,7 %), maxabs 7 |

- **fft3dneocl** : quelques pixels par million, toujours à 1 unité de valeur —
  l'arrondi du FFT float32 (Stockham contre FFTW). C'est l'objectif du port :
  identique au CPU.
- **fft3dcl** : ~70 % des pixels diffèrent de 0 à 7-9 unités. C'est le filtre
  FFT3DGPU (autre recouvrement, autres formules), pas un défaut de notre mesure.
  (Rappel : le `fft3dgpu.dll` DX9 d'origine rend une grille d'artefacts sur cette
  RTX 3050 — `fft3dcl` OpenCL, lui, rend une image saine.)

## Performances (ms/image, 30 images, GPU chaud, requêtes séquentielles)

| Cas | neo-fft3d CPU | fft3dneocl GPU | fft3dcl GPU | fft3dneocl vs CPU | fft3dcl vs CPU |
|---|---|---|---|---|---|
| 720p bt=1 | 13,85 ms | **2,06 ms** | 1,90 ms | 6,7× | 7,3× |
| 720p bt=3 | 16,16 ms | **3,30 ms** | 3,12 ms | 4,9× | 5,2× |
| 720p bt=5 | 18,53 ms | **3,48 ms** | non supporté | 5,3× | — |
| 1080p bt=3 | 37,16 ms | **5,65 ms** | 7,30 ms | 6,6× | 5,1× |

Les deux ports GPU sont au coude-à-coude : `fft3dcl` gagne de quelques % en 720p
(son recouvrement bw/2 fait plus de blocs mais ses noyaux sont compacts),
`fft3dneocl` gagne de ~23 % en 1080p grâce au cache de spectres `fftcache`.
Les deux sont 5 à 7 fois plus rapide que le moteur CPU multithread.

## Empreinte mémoire

### RAM système (pic du processus, inclut les tampons du banc d'essai)

| Cas | neo-fft3d CPU | fft3dneocl GPU | fft3dcl GPU |
|---|---|---|---|
| 720p bt=1 | 60 MB | **168 MB** | 295 MB |
| 720p bt=3 | 159 MB | **294 MB** | 302 MB |
| 720p bt=5 | 191 MB | **261 MB** | — |
| 1080p bt=3 | 335 MB | **402 MB** | 531 MB |

Le banc d'essai ajoute les mêmes tampons des deux côtés GPU (entrée + sorties en
mémoire, ~55-120 MB) : à conditions identiques, `fft3dcl` garde ~100-130 MB de
RAM de plus que nous (plans float pleine image côté hôte + cache de spectres).

### VRAM GPU

| Cas | fft3dneocl (tampons exacts) | fft3dneocl (delta pilote) | fft3dcl (delta pilote) |
|---|---|---|---|
| 720p bt=1 | **51 MiB** | +127 MB | +457 à 553 MB (3 mesures) |
| 720p bt=3 | **166 MiB** | +255 MB | +185 MB (3 mesures identiques) |
| 720p bt=5 | **199 MiB** | +291 MB | — |
| 1080p bt=3 | **359 MiB** | +441 MB | +320 MB |

- « tampons exacts » = somme déclarée par notre plugin (ligne journal
  `fft3dneocl: N MiB of device buffers`, comme `fft3dcl` en loggue une).
- « delta pilote » = mémoire GPU totale vue par `nvidia-smi` pendant le run ;
  il inclut le contexte GPU + VapourSynth (~80-90 MB, commun aux deux).
  `nvidia-smi` ne sait pas attribuer la VRAM aux processus OpenCL, d'où le delta.
- Anomalie mesurée côté **fft3dcl à bt=1** : ~500 MB reproductibles (3 runs),
  contre 185 MB à bt=3. Son cache de spectres GPU est vidé à chaque image en
  bt=1 (`pruneCache(-1,-2)`) ; le pilote conserve les tampons libérés. À bt=3 le
  cache est glissant et la VRAM reste petite. Chez nous les tampons sont fixes
  (alloués une fois), pas de ballottement.
- Nos tampons sont plus gros à bt élevé (cache `fftcache` = bt+3 spectres par
  plan, comme le CPU) : 359 MiB en 1080p bt=3, contre ~230 MB estimés pour
  `fft3dcl` (il garde le spectre pleine image + un second + les plans float).

## Commentaire en français simple

- **Qualité** : notre plugin reproduit le filtre CPU d'origine à 1 unité de
  valeur près sur environ un pixel par million. `fft3dcl` est un *autre* filtre
  (FFT3DGPU) : il donne une image différente, pas moins « propre », mais pas
  comparable pixel à pixel avec neo-FFT3D.
- **Vitesse** : les deux ports GPU se valent en 720p ; en 1080p nous sommes
  ~23 % devant. Les deux font 5 à 7 fois le débit du CPU.
- **Mémoire** : nous utilisons moins de RAM système que `fft3dcl` (100-130 MB de
  moins) et notre VRAM est stable et prévisible ; `fft3dcl` en utilise un peu
  moins en temporel (bt=3) mais affiche un pic anormal d'environ 500 MB en 2D
  (bt=1) dû au va-et-vient de ses tampons GPU.
- **Fonctions** : `fft3dcl` fait Kalman (bt=0) et sharpen seul (bt=-1) mais pas
  bt=5 ; nous faisons bt=-1, bt=0 (Kalman) et bt=1..5, seul le mode motif reste à faire.

## Reproduire

```
powershell -NoProfile -File tools\run_bench_report.ps1
# ou au cas par cas :
.\fft3dneocl\build\Release\plugin_smoke.exe tests\bench_720p_bt3_cpu\input.raw tests\bench_720p_bt3_cpu\out.raw 1280 720 8 GRAY tests\bench_720p_bt3_cpu\plugin_out.raw bt=3 bw=32 bh=32 sigma=2 frames=30 time=1
.\fft3dneocl\build\Release\plugin_smoke.exe tests\bench_720p_bt3_cpu\input.raw tests\bench_720p_bt3_cpu\out.raw 1280 720 8 GRAY x.raw bt=3 bw=32 bh=32 sigma=2 frames=30 ref="C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\build\Release\fft3dcl.dll" refns=fft3dcl reffn=FFT3DCL device=1 noplugin=1 refout=tests\bench_720p_bt3_cpu\fft3dcl_out.raw time=1
reference\harness\build\Release\fft3d_ref.exe --width 1280 --height 720 --format GRAY --bitdepth 8 --frames 30 --seed 5 --sigma 2 --bw 32 --bh 32 --bt 3 --time --out %TEMP%\cpu_out.raw
```

Outils ajoutés pour ce bilan : `plugin_smoke` accepte `refout=` (sortie complète
du filtre de référence), `noplugin=` (reference seule, sans notre instance) et
passe `device=` au seul filtre de référence ; le pipeline GPU journalne sa VRAM
(`fft3dneocl: N MiB of device buffers`).
