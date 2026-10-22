# Fuite mémoire DRAM à l'ouverture/fermeture d'aperçu StaxRip — diagnostic et correctif

(2026-10-22. Machine : Ryzen 7 5700X, RTX 3050 Laptop 6 Go, 32 Go RAM,
VapourSynth R81. Symptôme signalé par l'utilisateur : à chaque ouverture puis
fermeture d'un aperçu StaxRip en 4K (`sigma=3 bt=4 bw=32 bh=32 ow=16 oh=16
sharpen=1.8`), la mémoire vive (RAM, pas la VRAM) augmentait d'environ 2,5 Go
par cycle, jusqu'à l'épuisement.)

## Outil de reproduction : `leak_harness`

`fft3dneocl/tools/leak_harness.cpp` (construit avec le plugin) reproduit le
cycle dans un seul processus : à chaque cycle il crée un core VapourSynth,
charge le plugin, filtre quelques images, puis détruit le core, et mesure la
mémoire privée du processus (psapi) à chaque étape.

```
leak_harness.exe W H bits fmt cycles [key=val ...]
  key=val spéciaux (avec le =) :
    frames=N    nombre d'images du clip synthétique (défaut 4)
    fetch=N     ne traiter que N images
    nofilter=1  sans filtre (plancher core+source seuls)
    nofetch=1   créer le filtre sans traiter d'image
    stdcopy=1   filtre std.Invert à la place (VS pur, zéro GPU)
    onecore=1   un seul core pour tous les cycles (filtres recréés seuls)
    fetchstart=N  commencer à la frame N (évite le clamp des bords)
    threads=N   setThreadCount du core VapourSynth
    vmmap=1   histogramme VirtualQuery des régions privées engagées (fin)
    ref=<dll> refns=fft3dcl reffn=FFT3DCL   mesurer fft3dcl à la place
```

Variables d'environnement du plugin : `FFT3DNEOCL_LOG=1` écrit
`%TEMP%\fft3dneocl_pool.log` (acquire REUSE/CREATE, création/libération
d'instance, MiB de tampons) ; `FFT3DNEOCL_XFER=map|copy` choisit la stratégie
de transfert hôte↔GPU (par défaut `copy` via `h_stage` persistant).

## Chaîne de diagnostic (1080p10 YUV420, 8 images, 8 cycles)

| Configuration | Résidu par cycle |
|---|---|
| `nofilter=1` (core + source seuls) | plat (±1 Mo) |
| `stdcopy=1` (filtre VapourSynth pur, zéro GPU) | plat (±1 Mo) |
| `nofetch=1` (contexte + programme + tampons GPU, sans traitement) | plat |
| neocl complet, **avant correctif** | +47 Mo |
| neocl, tampons d'images CPU persistants seuls | +45 Mo (inchangé) |
| neocl, contexte OpenCL partagé seul | +45 Mo (inchangé) |
| neocl, contexte + file de commandes partagés | +45 Mo (inchangé) |
| neocl, transferts par map/unmap au lieu de Write/ReadBuffer | +45 Mo (inchangé) |
| **neocl, pipelines GPU mutualisés (pool)** | +44 Mo, mais brassage ramené de ~1 Go à ~120 Mo |
| **neocl, `onecore=1` (core conservé)** | **plat (±1 Mo)** |
| fft3dcl (référence) dans le même harnais | **+165 Mo** |

Conclusion du diagnostic : la fuite visible dans StaxRip était dominée par le
**brassage des tampons GPU par instance** (~1,5 Go de tampons recréés et
libérés à chaque aperçu en 4K, plus la recompilation des programmes). Ce
brassage, combiné à la destruction du core VapourSynth, fragmente les arènes
mémoire du pilote NVIDIA et du tas, qui ne rendent pas toute la mémoire.
`onecore=1` (filtres recréés mais core conservé) est plat : c'est la
combinaison des deux qui fuit.

## Correctifs appliqués (tous dans le plugin, aucun changement d'algorithmes)

1. **Chargeur OpenCL partagé** (`ocl::shared_api()`) : `OpenCL.dll` chargé une
   seule fois ; `Api::load` est idempotent.
2. **Pool de contextes par périphérique** (`ocl::shared_context()`) : le
   contexte survit aux instances ; les tampons/programmes restent par pipeline.
3. **Pool de files de commandes par (contexte, périphérique)**
   (`ocl::shared_queue()`).
4. **Pool de pipelines en liste libre** (`GpuPipeline::acquire()`) : chaque
   instance reçoit son propre pipeline (pas de course), mais un pipeline libéré
   **revient dans le pool** au lieu d'être détruit ; la prochaine instance le
   réutilise. Clé = signature complète des paramètres + **index de plan** +
   périphérique (sans l'index de plan, U et V — paramètres identiques —
   partageaient la même entrée et le second recréait un jeu complet à chaque
   fois). Plafonds : 2 entrées inactives par clé, 6 au total (changer souvent
   de paramètres ne fait pas croître la mémoire sans limite).
   Une première version « une entrée par clé, séparation quand en usage » a
   empiré les choses dans StaxRip : les évaluations de script se chevauchent
   (script principal + aperçu), chaque chevauchement créait un jeu séparé de
   ~4,3 Go de tampons 4K — c'est l'escalier ~3-4 Go/cycle observé par
   l'utilisateur. La liste libre supprime ce brassage (journal réel StaxRip :
   24 REUSE / 9 CREATE, plus aucun SEPARATE).
5. **`reset_state()` à la réutilisation** : cache de spectres (fftcache),
   drapeau de pattern et état Kalman sont réinitialisés exactement comme à la
   création — les images calculées par un pipeline réutilisé sont identiques
   octet pour octet.
6. **Transferts hôte↔GPU par `EnqueueMapBuffer`/`Unmap`** au lieu de
   `EnqueueWriteBuffer`/`EnqueueReadBuffer` synchrones : mêmes octets, mais le
   pilote ne recrée pas de zone épinglée par transfert. (Au passage :
   correction de la signature du chargeur, `EnqueueMapBuffer` retourne `void*`
   et non `cl_int`.)
7. **Tampon de lecture `h_stage` persistant** par pipeline (plus
   d'allocation de ~6 Mo par plan et par image).
8. **`vsapi->freeNode(d->child)` dans `fft3dFree`** — la vraie cause racine.
   En API4, `createVideoFilter2` ne consomme **pas** la référence sur le nœud
   dépendance obtenue par `mapGetNode` : le filtre en est propriétaire et doit
   la rendre dans son callback de libération (squelette officiel
   `sdk/filter_skeleton.c`). Nous ne la rendions jamais : à la fermeture de
   l'aperçu, le nœud FFT3D était détruit mais le nœud source (décodeur ffms2 +
   cache d'images décodées) restait référencé et n'était jamais détruit —
   tout était retenu jusqu'à la fermeture de StaxRip, exactement le symptôme
   décrit. Le plugin fft3dcl avait eu le même bug, corrigé dans le commit
   [`bf54ad3`](https://github.com/HumbleUser33/FFT3dCL/commit/bf54ad3b4b8e33f15379f9051be47baf8dafc635)
   (+830 MiB/cycle → +0,5 MiB/cycle en 4K, pixels inchangés). Nos chemins
   d'erreur de `fft3dCreate` libéraient déjà la référence (lambda `fail`).
9. **Destructeur du pool de pipelines** (`~PipePool`) : VapourSynth décharge le
   DLL du plugin (`FreeLibrary`) à la destruction du core. Les pipelines au
   repos dans le pool statique n'étaient jamais détruits → à chaque cycle
   core-créé/détruit, ~2,1 Go de tampons GPU + le `h_stage` fuiraient à la
   décharge. Le destructeur libère tout (le chargeur OpenCL, POD, reste
   valide pendant la décharge ; le comptage de références OpenCL garde les
   contextes vivants jusqu'à la libération de nos tampons).

Effets secondaires positifs : ouverture d'aperçu beaucoup plus rapide (pas de
recréation de contexte, de recompilation, ni de réallocation des tampons :
`nodes: 0` et `fetch: 1000→120 Mo` dès le 2ᵉ cycle). La VRAM des tampons est
conservée entre aperçus et réutilisée.

## Vérifications

- **Matrice de 51 fixtures** (`tools/run_plugin_matrix.ps1`) après tous les
  changements : 0 fixture hors tolérance (le ±1 LSB connu de `pat_pframe2`
  accepté).
- **Non-régression qualité/perf sur vidéo réelle 4K** (après les correctifs
  8+9, même méthode que `docs/benchmark-real-video.md`) : presets P1-P4 sur
  la séquence 171 et P2 sur la séquence 419, 60 images, relancés avec le
  nouveau DLL — les 5 sorties (1,4 Go chacune) ont un **MD5 identique** aux
  sorties de l'ancien DLL (21/10). Vitesse sans régression : P1 53,2 (avant
  55,6), P2 52,0/52,7 (53,8/50,7), P3 53,8 (60,3), P4 bt=5 58,1 (67,4)
  ms/image.
- **Empreinte par cycle** (`fnv=` dans `leak_harness`) : identique sur tous les
  cycles pour bt=4 (fenêtre temporelle + fftcache), bt=0 (Kalman) et
  pfactor (pattern) → la réutilisation des pipelines ne change aucun pixel.
- **Éviction LRU** : 8 cycles avec des paramètres différents à chaque fois,
  sans crash ni croissance non bornée.
- **`onecore=1`** avec le code final : plat (±1 Mo/cycle).

## Plancher résiduel — résolu (la section précédente était fausse)

Le résidu « +262 Mo/cycle en 4K » attribué à tort à VapourSynth/au pilote
était en réalité **deux fuites de notre code**, trouvées grâce au commit
`bf54ad3` de fft3dcl signalé par l'utilisateur :

1. la référence du nœud source jamais rendue (correctif 8 ci-dessus) — c'est
   elle qui retenait les images VapourSynth après `freeCore` (les régions
   15,8/4,0 Mo de `vmmap`, le résidu égal au jeu d'images travaillées) ;
2. le pool de pipelines non nettoyé à la décharge du DLL (correctif 9) —
   visible seulement quand le pool existe, ~936 Mo/cycle en 4K dans le
   harnais qui recharge le DLL à chaque cycle.

Mesure finale, harnais 4K YUV420 10 bits, `frames=8`, 3 cycles (core +
chargement du plugin recréés à chaque cycle) :

| Cycle | Avant correctifs | Après correctifs 8+9 |
|---|---|---|
| 1 | +262 | +302 (jeu de démarrage) |
| 2 | +262 | **+0,2** |
| 3 | +262 | **−6,1** |

`fnv` identique sur les trois cycles ; `corefree: −2437 Mo` : tout se
désengage. La matrice de 51 fixtures reste à 0 hors tolérance.

Reste « retenu mais non cumulé » (volontaire, pas une fuite) : les pipelines
au repos dans le pool gardent leurs tampons GPU/DRAM tant que le DLL reste
chargé, pour réutilisation à l'aperçu suivant ; tout part à la fermeture de
StaxRip ou à la décharge du DLL.

## Test dans StaxRip réel (automatisation UIA)

Lancement de `StaxRip.exe "<projet>.srip"`, ouverture de l'aperçu via l'élément
de menu « Preview » (InvokePattern UIA), mesure de `PrivateMemorySize64`.
Limite découverte : **StaxRip v2.52.6 meurt quand la fenêtre d'aperçu est
fermée programmatiquement** (WM_CLOSE ou bouton « Fermer » via UIA) — y compris
avec le projet CPU qui n'appelle pas notre DLL. La fermeture manuelle de
l'utilisateur fonctionne ; l'automatisation ne peut pas enchaîner les cycles
ouverture/fermeture. Le journal `fft3dneocl_pool.log` du build à liste libre
montre le bon comportement dans StaxRip réel : 24 REUSE / 9 CREATE, plus aucun
jeu séparé de 4,3 Go.
