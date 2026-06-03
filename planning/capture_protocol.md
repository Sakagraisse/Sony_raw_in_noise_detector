# Protocole de capture pour contributeurs

Ce protocole sert a produire des fichiers RAW exploitables pour mesurer read noise,
gain, full well et Photographic Dynamic Range. L'ecran n'a pas besoin d'etre calibre,
mais il doit etre stable, uniforme et utilise de facon reproductible.

## 1. Materiel minimum

- Camera capable de produire des fichiers RAW.
- Objectif monte sur la camera.
- Ecran pour afficher la mire en plein ecran.
- Trepied ou support stable.
- Bouchon d'objectif pour les dark frames.

Materiel recommande:

- Declencheur retardateur, telecommande ou retardateur 2 s.
- Piece sombre ou lumiere ambiante controlee.
- Ecran haute resolution, idealement 4K/Retina ou mieux.

## 2. Reglages de l'ecran

Avant la prise de vue:

- Desactiver auto-brightness.
- Desactiver True Tone, Night Shift, filtre lumiere bleue et modes similaires.
- Desactiver HDR, local dimming dynamique et contrast enhancer si possible.
- Fixer la luminosite manuellement.
- Afficher la mire en plein ecran.
- Utiliser la resolution native de l'ecran si possible.
- Nettoyer les reflets: pas de lampe ou fenetre visible dans l'ecran.

L'ecran n'a pas besoin d'etre calibre en couleur. Le calcul utilise les valeurs RAW
capturees par la camera, pas une luminance absolue connue. Le point critique est
d'eviter que l'ecran ajoute de la texture ou des variations locales qui ressemblent
a du bruit capteur.

## 3. Reglages camera

Reglages recommandes:

- Mode manuel complet.
- RAW non compresse si disponible.
- Desactiver les corrections automatiques si le boitier le permet.
- Balance des blancs fixe.
- Stabilisation desactivee si la camera est sur trepied.
- Meme ouverture et meme vitesse pour le chart frame et le dark frame d'un meme ISO.
- ISO natifs et intermediaires selon la campagne de test.

Serie ISO recommandee pour un premier test:

```text
100, 200, 400, 800, 1600, 3200, 6400, 12800
```

Ajouter les ISO intermediaires si l'objectif est de detecter dual gain, digital gain
ou comportements speciaux du boitier.

## 4. Cadrage et focus

Pour le chart frame:

- La mire doit remplir largement l'image.
- Les marqueurs ou coins de grille doivent rester visibles.
- La camera doit etre aussi parallele que possible a l'ecran.
- Le defocus doit etre leger et volontaire.

Regle pratique pour le defocus:

- Trop net: on voit les pixels/sous-pixels de l'ecran, ce qui fausse la variance.
- Trop flou: les patches se melangent et la detection de grille devient fragile.
- Bon compromis: les pixels de l'ecran disparaissent, mais les separations de patches
  restent lisibles.

## 5. Exposition du chart frame

Objectif:

- Exposer a droite sans clipping.
- Le patch le plus clair doit etre proche de la saturation RAW, mais non clippe.
- Les patches sombres doivent rester au-dessus du black level.

Procedure:

1. Afficher la mire.
2. Choisir l'ISO.
3. Regler vitesse/ouverture pour approcher la saturation sans l'atteindre.
4. Verifier l'histogramme si disponible.
5. Prendre le chart frame.
6. Sans changer ISO/vitesse/ouverture, mettre le bouchon.
7. Prendre le dark frame.

Pour chaque ISO, il faut au minimum:

```text
1 chart frame + 1 dark frame
```

Pour une campagne plus robuste:

```text
2 chart frames + 2 dark frames par ISO
```

## 6. Nommage des fichiers

Le pipeline peut trier par ISO, mais les contributeurs doivent idealement garder un
nommage lisible.

Exemple:

```text
sony_a7iv_sample001_iso100_chart_001.ARW
sony_a7iv_sample001_iso100_dark_001.ARW
sony_a7iv_sample001_iso200_chart_001.ARW
sony_a7iv_sample001_iso200_dark_001.ARW
```

Si le nommage n'est pas possible, conserver tous les fichiers d'une session dans un
meme dossier, sans melanger plusieurs cameras ou plusieurs sessions.

## 7. Metadonnees a fournir

Remplir le template `session_metadata_template.json` avant ou apres la capture.

Champs importants:

- Marque et modele camera.
- Sample ID du boitier.
- Format RAW.
- Objectif.
- Mode RAW compresse/non compresse.
- Ecran utilise.
- Resolution de l'ecran.
- Luminosite ecran si connue.
- Pattern version.
- Liste des ISO captures.
- Notes sur defocus, reflets, conditions et problemes.

## 8. Checklist d'acceptation rapide

Un set est probablement bon si:

- Tous les ISO ont au moins un chart frame et un dark frame.
- Les dark frames sont noirs et pris avec les memes reglages que les chart frames.
- Le chart frame n'est pas clippe sur le patch le plus clair.
- Les pixels de l'ecran ne sont pas clairement visibles a 100 %.
- Les patches restent detectables.
- L'ecran n'a pas change de luminosite pendant la serie.
- Les metadonnees de session sont remplies.

Un set doit etre marque douteux si:

- Auto-brightness/HDR/True Tone etait actif.
- Il y a des reflets visibles.
- Le chart frame est fortement clippe.
- Le defocus melange les patches.
- La camera a change de mode RAW ou d'exposition entre chart et dark.
- La session melange plusieurs boitiers sans metadata claire.

## 9. Controle qualite logiciel a ajouter

Le pipeline devra calculer automatiquement:

- Nombre de patches valides.
- Nombre de patches rejetes pour clipping.
- Score de texture ecran intra-patch.
- Score d'uniformite locale des patches.
- Score de detection de grille.
- Erreur de reprojection homographie.
- `R2` de la Photon Transfer Curve.
- Warnings black level et white level.

Ces scores permettront d'accepter, refuser ou etiqueter les samples participatifs
sans imposer le meme ecran a tout le monde.

