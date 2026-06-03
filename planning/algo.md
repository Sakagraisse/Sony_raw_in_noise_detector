# Algorithme de detection de la charte

## Objectif

Detecter automatiquement la charte de mesure, meme avec defocus, parallax,
exposition variable et montee ISO. L'algorithme ne doit pas simplement chercher
des pixels rouges/verts/bleus/jaunes. Il doit exploiter toute la connaissance
du pattern:

- 4 marqueurs de coin connus.
- Anneaux colores avec centre noir.
- Couleurs assignees a des coins fixes.
- Geometrie complete sauvegardee dans le JSON du pattern.
- Grille de patches connue, ici 11 x 7.
- Serie ISO souvent prise sur trepied.

## Pattern recommande

La version recommandee actuellement est:

```text
DR_Grid_4.3_ColorRings.png
```

Principe:

- Anneau rouge = haut gauche.
- Anneau vert = haut droit.
- Anneau bleu = bas droit.
- Anneau jaune = bas gauche.
- Centre noir dans chaque anneau.
- Liseré blanc externe comme aide secondaire.
- Patches de mesure separes des marqueurs.

Le role de l'anneau colore est d'identifier le coin.
Le role du centre noir est de fournir une ancre geometrique stable.

## Pipeline de detection cible

### 1. Pretraitement image

Utiliser plusieurs representations de l'image:

- RGB lineaire ou normalise.
- HSV pour la teinte et la saturation.
- Lab pour separer luminance et chroma.
- Image grayscale pour contours, ellipse et centre noir.

Ne pas utiliser seulement des seuils absolus de couleur. Avec la montee ISO,
les couleurs peuvent devenir plus claires, se desaturer ou clipper.

Preferer:

- ratios entre canaux;
- dominance locale d'une couleur;
- contraste local;
- position attendue;
- forme attendue.

### 2. Recherche par zones attendues

Diviser l'image en quatre zones larges:

- haut gauche;
- haut droit;
- bas droit;
- bas gauche.

Dans chaque zone, chercher prioritairement le marqueur attendu:

```text
haut gauche  -> rouge
haut droit   -> vert
bas droit    -> bleu
bas gauche   -> jaune
```

Cette contrainte rend l'algorithme beaucoup plus robuste qu'une detection globale:
un patch sature ou une derive de couleur ne devrait pas pouvoir remplacer un
marqueur si sa position est incoherente.

### 3. Detection de l'anneau colore

Pour chaque zone:

1. Construire un masque couleur adaptatif.
2. Chercher les composants connexes plausibles.
3. Garder les candidats ayant une taille plausible.
4. Evaluer leur circularite ou leur ellipse ajustee.
5. Rejeter les formes trop petites, trop allongees ou trop loin du coin attendu.

Score candidat:

```text
score_anneau =
  score_couleur
  + score_position
  + score_taille
  + score_circularite
```

La couleur ne doit pas etre le seul critere. Le marqueur est une forme circulaire
attendue a une position attendue.

### 4. Detection du centre noir

Une fois l'anneau trouve:

1. Faire un crop local autour de l'anneau.
2. Chercher une zone sombre proche du centre de l'anneau.
3. Nettoyer le masque sombre par morphologie.
4. Calculer le centroide du composant sombre principal.
5. Utiliser ce centroide comme point geometrique.

Fallback:

- Si le trou noir n'est pas trouve, utiliser le centre de l'ellipse ajustee.
- Si l'ellipse est mauvaise, utiliser le centroide couleur.
- Si le marqueur est absent mais que 3 marqueurs sont fiables, predire le 4e par geometrie.

Le centre noir est important parce qu'il reste plus stable que l'anneau quand les
couleurs commencent a clipper.

### 5. Homographie

Quand les 4 points sont disponibles:

1. Recuperer les centres sources depuis le JSON du pattern.
2. Associer les centres detectes par couleur.
3. Calculer l'homographie pattern -> image.
4. Projeter la grille theorique.
5. Extraire les rectangles internes des patches.

La grille ne doit pas etre detectee patch par patch. Elle doit etre deduite de la
geometrie du pattern apres homographie.

### 6. Validation geometrique

Apres homographie, calculer un score global:

- Les 4 marqueurs forment-ils un quadrilatere plausible?
- Les bords opposes ont-ils des longueurs plausibles?
- L'aire projetee est-elle suffisante?
- La grille projetee tombe-t-elle dans la zone attendue?
- Les centres de patches restent-ils dans l'image?
- Les rectangles internes ne croisent-ils pas les gouttieres noires?

Score global:

```text
global_score =
  score_marqueurs
  + score_homographie
  + score_projection_grille
  + score_qualite_patches
```

Si le score global est faible, l'image doit etre marquee comme douteuse, meme si
une homographie a ete calculee.

## Gestion de la serie ISO

### Detection directe

La detection directe peut fonctionner aux ISO bas ou moyens. Mais aux ISO eleves,
les patches et parfois les marqueurs peuvent clipper. Les couleurs deviennent alors
moins fiables.

Donc la detection directe ne doit pas etre obligatoire pour chaque ISO.

### Propagation depuis ISO 100

Si la serie est prise sur trepied:

1. Detecter la charte sur l'image la mieux exposee, typiquement ISO 100.
2. Calculer l'homographie de reference.
3. Reutiliser cette homographie sur les autres ISO.
4. Optionnellement appliquer une correction locale si un micro-deplacement est detecte.

La propagation est la methode recommandee pour les campagnes multi-ISO.

### Correction de micro-mouvement

Si la camera bouge legerement:

- utiliser phase correlation;
- ou optical flow sur le liseré / bords de grille;
- ou recalage par template matching autour des marqueurs;
- puis ajuster l'homographie de reference.

## Methodes alternatives possibles

### 1. Hough Circle / ellipse fitting

Utiliser contours, Canny ou Hough Circle pour trouver des formes circulaires.
Bon pour detecter les anneaux si les couleurs deviennent difficiles.

### 2. Template matching

Creer un template de marqueur floute a plusieurs echelles.
Utiliser normalized cross-correlation dans chaque quadrant.

### 3. Detection par modele generatif synthetique

Generer plusieurs versions synthetiques du pattern:

- flou variable;
- parallax;
- clipping;
- bruit;
- white balance;
- crop;
- rotation;
- compression.

Puis tester les algorithmes classiques sur ce dataset avant d'utiliser des photos reelles.

### 4. Mini modele vision entraine

Entrainer un petit modele pour predire directement les 4 centres:

- entree: image downsamplee;
- sortie: heatmaps des 4 marqueurs ou coordonnees normalisees;
- dataset: principalement synthetique, puis fine tuning avec vraies photos.

Ce n'est pas un LLM, mais un petit modele de vision specialise.
Sur MacBook Air M3, une preuve de concept peut etre envisagee en quelques heures.

### 5. Detection hybride classique + ML

Utiliser l'algorithme classique comme premier detecteur.
Utiliser le modele ML uniquement quand le score classique est faible.

Cela garde le pipeline rapide, explicable et robuste.

## Criteres de succes

L'algorithme sera considere fiable quand:

- Il detecte correctement la charte a ISO bas en vue plate et parallax.
- Il extrait les 77 patches avec rectangles internes bien centres.
- Il rejette les faux positifs au lieu de produire une mauvaise homographie.
- Il propage correctement la grille sur les ISO eleves en scenario trepied.
- Il produit un score de confiance lisible.
- Les overlays permettent une verification humaine rapide.

## Prochaine implementation recommandee

Remplacer la detection actuelle trop simple par:

```text
quadrants
-> detection anneau colore
-> detection centre noir
-> ellipse fallback
-> homographie
-> projection JSON
-> scoring qualite
```

Ensuite seulement, envisager un modele vision leger si les cas reels restent trop
variables pour une methode classique.

