# Équivalence sigma / sharpen — fft3dneocl vs fft3dcl

Comment régler `fft3dcl` pour qu'il donne le même résultat visuel que
`fft3dneocl` (et l'inverse), à blocs identiques.

## Méthode

- Contenu réel : 10 images de `seq_171` (Sample.mkv, 4K HEVC, zone très bruitée).
- Réglages communs aux deux filtres : `bw=32 bh=32 ow=16 oh=16 beta=1`,
  plans Y+U+V, degrid par défaut. Balayage `bt=3` et `bt=4`.
- **Phase A** : balayage du sigma (sharpen=0). Deux critères d'équivalence :
  1. *même force* = même écart moyen (MAE) à l'image d'origine, interpolé ;
  2. *différence minimale* = sigma fft3dcl dont la sortie ressemble le plus
     pixel à pixel à notre sortie.
- **Phase B** : balayage du sharpen (0,25 → 1,0) aux sigma appariees, mêmes
  deux critères.
- Outils : `tools/run_sigma_match.ps1` (lancements), `tools/sigma_match_report.py`
  et `tools/sharpen_match_report.py` (analyse), résultats JSON dans
  `tests\bench_real\match\`.

## Phase A — sigma

### bt=3 : les sigma sont quasi interchangeables

| sigma fft3dneocl | fft3dcl même force | fft3dcl différence min | différence résiduelle |
|---|---|---|---|
| 1,5 | 1,54 | **1,5** | 0,22 |
| 2,0 | 2,05 | **2,0** | 0,26 |
| 2,5 | 2,57 | **2,5** | 0,28 |
| 3,0 | 3,10 | **3,0** | 0,30 |
| 3,5 | 3,63 | **3,5** | 0,31 |
| 4,0 | 4,0 | **4,0** | 0,32 |

À blocs égaux, la force des deux filtres suit la même courbe : il faut à peine
+3 à +10 % de sigma côté fft3dcl pour la même « force » mesurée, et le sigma
qui donne les images les plus identiques est **le même sigma**. La différence
résiduelle (0,2–0,3 sur 1023 niveaux) est invisible.

### bt=4 : les moteurs divergent

| sigma fft3dneocl | fft3dcl même force | fft3dcl différence min | différence résiduelle |
|---|---|---|---|
| 1,5 | 1,64 | **1,5** | 0,86 |
| 2,0 | 2,23 | **2,0** | 1,11 |
| 2,5 | 2,88 | **2,75** | 1,32 |
| 3,0 | 3,59 | **3,5** | 1,50 |
| 3,5 | 4,0 | **4,0** | 1,63 |
| 4,0 | 4,0 | **4,0** | 1,79 |

Explication : chez nous bt=4 donne quasiment le même résultat que bt=3
(l'écart à l'original passe de 4,405 à 4,406 à sigma 3), alors que fft3dcl
bt=4 débruite **moins** que son bt=3 (4,33 → 4,01). Il faut donc pousser sa
sigma d'environ +15 à +20 % pour retrouver notre niveau. La différence
résiduelle (≈1,5) reste 5 fois plus grande qu'en bt=3 : à bt=4 les deux
algorithmes ne font pas le même traitement temporel, l'équivalence n'est
que approximative.

## Phase B — sharpen

### bt=3 (sigma 3,0 des deux côtés) : sharpen identique des deux côtés

Différence directe entre sorties (lignes = sharpen neocl, colonnes = sharpen fft3dcl) :

|  | 0,25 | 0,5 | 0,75 | 1,0 |
|---|---|---|---|---|
| **0,25** | **0,34** | 0,47 | 0,65 | 0,85 |
| **0,5** | 0,48 | **0,38** | 0,48 | 0,66 |
| **0,75** | 0,66 | 0,51 | **0,43** | 0,51 |
| **1,0** | 0,84 | 0,68 | 0,54 | **0,47** |

Le minimum est toujours sur la diagonale : **gardez la même valeur de sharpen
des deux côtés**. (Note : le sharpen réduit l'écart à l'origine des deux côtés
— il réinjecte du détail là où le débruitage en avait enlevé — ce critère seul
ne permet pas de les départager ; c'est la comparaison directe qui tranche.)

### bt=4 (sigma 3,0 chez nous / 3,5 chez fft3dcl) : sharpen fft3dcl un cran plus fort

|  | 0,25 | 0,5 | 0,75 | 1,0 |
|---|---|---|---|---|
| **0,25** | **1,55** | 1,62 | 1,74 | 1,87 |
| **0,5** | **1,56** | 1,59 | 1,67 | 1,78 |
| **0,75** | 1,62 | **1,61** | 1,64 | 1,72 |
| **1,0** | 1,71 | 1,67 | **1,66** | 1,69 |

Le minimum glisse d'un cran vers le bas côté fft3dcl : notre sharpen 0,5 ↔
fft3dcl 0,25 ; notre 1,0 ↔ fft3dcl 0,75. L'écart entre les cellules est faible
(1,55 à 1,87), donc garder la même valeur reste acceptable ; viser un cran de
moins côté fft3dcl affine légèrement.

## Confirmation visuelle

Recadrage 768 px sur le cadran « 87645 » (image 174), sharpen 0,5 :

- `tests\bench_real\match\grille_bt3.png` : neocl s3,0 et fft3dcl s3,0 sont
  indiscernables ; fft3dcl s3,5 est visiblement plus lisse (trop fort).
- `tests\bench_real\match\grille_bt4.png` : fft3dcl s3,0 garde plus de grain
  que notre s3,0 ; fft3dcl s3,5 correspond.

## Table pratique (bw=bh=32, ow=oh=16)

| Réglage fft3dneocl | Réglage fft3dcl équivalent |
|---|---|
| bt=3, sigma S, sharpen K | bt=3, sigma S (ou S×1,05), sharpen K |
| bt=4, sigma S, sharpen K | bt=4, sigma S×1,15 à 1,20, sharpen K (ou un cran moins) |

## Limites

- Équivalence établie sur une seule source 4K très bruitée ; le bt=4 des deux
  moteurs diffère structurellement (différence résiduelle ≈1,5/1023 contre
  ≈0,3 en bt=3) — l'équivalence y est indicative.
- Les critères sont des moyennes statistiques ; un jugement subjectif peut
  préférer un réglage légèrement différent.
- Rappel : avec les recouvrements *par défaut* (ow non précisé), fft3dneocl
  utilise ow=bw/3 et fft3dcl ow=bw/2 — l'écart de « force » observé
  précédemment venait surtout de là, pas du sigma.
