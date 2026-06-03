# Plan d'evolution du projet

## 1. Objectif general

Ce document fixe les prochaines evolutions du projet Sony Sensor Analysis Toolchain.
Le but est de passer d'un prototype fonctionnel centre sur Sony a une base de mesure
plus robuste, automatisable, comparable aux references PhotonsToPhotos / William J.
Claff, et suffisamment structuree pour accumuler ensuite beaucoup de samples par
marque, modele et exemplaire de camera.

Priorites demandees:

1. Creer ce dossier de planification et documenter la feuille de route.
2. Lister les mesures et references PhotonsToPhotos / William J. Claff.
3. Rediger un white paper de methode et de calcul.
4. Structurer le projet pour automatiser le run malgre le defocus et les pixels de l'ecran.
5. Archiver proprement les resultats par marque, camera, session et sample.
6. Garder Python comme coeur, puis preparer une couche HTML participative.

Documents de travail:

- `planning/capture_protocol.md`: protocole de capture pour les contributeurs.
- `planning/session_metadata_template.json`: template metadata d'une session.

## 2. References PhotonsToPhotos / William J. Claff

PhotonsToPhotos est la reference principale pour comparer les performances capteur.
Le projet doit viser des mesures compatibles ou au moins documentees par rapport aux
concepts publies par Bill Claff.

References utiles:

- Site principal: https://www.photonstophotos.net/
- Photographic Dynamic Range: https://www.photonstophotos.net/Charts/PDR.htm
- Photon Transfer Curves: https://www.photonstophotos.net/Charts/PTC.htm
- Read Noise in DNs: https://www.photonstophotos.net/Charts/RN_ADU.htm
- Input-referred Read Noise: https://www.photonstophotos.net/Charts/RN_e.htm
- Sensor Characteristics: https://www.photonstophotos.net/Charts/Sensor_Characteristics.htm
- Gain Collaboration: https://www.photonstophotos.net/Collaborations/Gain_Collaboration.htm

Mesures a documenter et, a terme, produire:

- Photographic Dynamic Range (PDR): plage dynamique photographique, exprimee en EV.
- PDR normalise print: valeur comparable apres normalisation de resolution, comme dans l'esprit PhotonsToPhotos.
- Pixel PDR: dynamique au niveau pixel, avant normalisation de sortie.
- Engineering Dynamic Range (EDR): dynamique SNR=1, utile comme reference technique.
- Low Light ISO / Low Light EV: seuils derives d'un niveau minimal de PDR acceptable.
- Read Noise in DNs / ADU: bruit de lecture dans l'unite brute numerique du RAW.
- Input-referred Read Noise: bruit de lecture rapporte en electrons.
- Gain ou conversion gain: electrons par ADU, derive de la Photon Transfer Curve.
- Full Well Capacity: capacite maximale avant saturation, en electrons.
- Saturation / white level: niveau RAW maximal utile avant clipping.
- Black level: pedestal RAW mesure et compare aux metadonnees.
- Dynamic range vs ISO: courbe principale par sensibilite.
- Read noise vs ISO: courbe utile pour detecter les changements de mode capteur.
- Gain vs ISO: courbe utile pour detecter dual gain, conversion gain et ruptures.
- Photon Transfer Curve (PTC): variance en fonction du signal moyen.
- ISO invariance / shadow improvement: analyse des variations de bruit et de PDR selon l'ISO.
- Heatmaps capteur: visualisation avancee des non-uniformites, a traiter plus tard.
- Energy spectra: analyse avancee des frequences spatiales du bruit, a traiter plus tard.

## 3. White paper methode et calcul

### 3.1 Protocole de capture

Pour chaque ISO mesure, le protocole minimal reste:

- Un dark frame: objectif bouche, meme ISO et idealement memes parametres d'exposition.
- Un chart frame: mire affichee a l'ecran, exposee a droite sans clipping.
- Une legere defocalisation volontaire: assez forte pour casser la texture des pixels de l'ecran et le moire, mais pas assez pour effacer les separations entre patches.
- RAW non compresse ou le plus lineaire possible si le boitier le permet.
- Stabilite: trepied, pas de reflet ecran, luminosite stable, temperature si possible documentee.

### 3.2 Entrees de calcul

Les entrees necessaires sont:

- RAW chart par ISO.
- RAW dark par ISO.
- Metadonnees EXIF/RAW: marque, modele, ISO, shutter, aperture, white level, black level si disponible.
- Geometrie de grille: nombre de colonnes/lignes, homographie, centres des patches, taille des zones internes.
- Version de la mire: couleurs, resolution cible, marges, espacement, markers.

### 3.3 Extraction des patches

La mire actuelle utilise une grille 11 x 7. Chaque patch fournit un niveau de signal.
L'analyse doit utiliser uniquement une zone interne du patch pour eviter les bords,
la diffusion due au defocus et les erreurs d'homographie.

Regle cible:

- Detecter ou ajuster la grille.
- Calculer l'homographie vers la geometrie theorique.
- Extraire les rectangles internes, par exemple 60 a 75 % de la taille du patch.
- Rejeter les patches clipses, trop sombres, ou trop instables.
- Enregistrer les overlays de controle pour audit visuel.

### 3.4 Bruit de lecture

Le bruit de lecture est mesure sur le dark frame.

Calcul de base:

1. Extraire les pixels verts du RAW Bayer, car ils sont majoritaires et souvent les plus stables.
2. Mesurer le black level comme moyenne d'une zone representative.
3. Mesurer le read noise en ADU comme ecart-type robuste.
4. Comparer le black level mesure au black level metadata.
5. Signaler un warning si l'ecart est important.

Notation:

- `RN_ADU = std(dark_green_pixels)`
- `black_level = mean(dark_green_pixels)`

Le projet actuel fait deja une premiere version de ce calcul dans `step3_analyze.py`.

### 3.5 Gain et Photon Transfer Curve

La Photon Transfer Curve relie le signal moyen et la variance du bruit.
Dans un modele lineaire simplifie:

```text
variance_ADU = slope * signal_ADU + intercept
gain_e_per_ADU = 1 / slope
```

Etapes:

1. Pour chaque patch non clipse, calculer `signal_ADU = mean(patch) - black_level`.
2. Calculer une variance robuste du patch.
3. Ajuster une regression lineaire `variance_ADU` vs `signal_ADU`.
4. Convertir la pente en gain `e-/ADU`.
5. Controler la qualite par `R2`, nombre de patches valides et rejet d'outliers.

Le defocus et la texture ecran peuvent creer une variance spatiale qui n'est pas du
bruit capteur. Il faut donc privilegier une variance robuste, des zones internes,
et potentiellement une methode pair-difference ou median absolute deviation.

### 3.6 Full Well et dynamique

Le full well est estime depuis le niveau utile maximal:

```text
full_well_ADU = white_level - black_level
full_well_e = full_well_ADU * gain_e_per_ADU
RN_e = RN_ADU * gain_e_per_ADU
```

Engineering Dynamic Range:

```text
EDR = log2(full_well_e / RN_e)
```

Photographic Dynamic Range:

Le PDR PhotonsToPhotos est lie a un seuil photographique de bruit plus exigeant que
SNR=1. La version actuelle du projet utilise un seuil SNR=20.

On cherche le signal `S` en electrons tel que:

```text
S / sqrt(RN_e^2 + S) = 20
```

Donc:

```text
S^2 - 400*S - 400*RN_e^2 = 0
PDR_pixel = log2(full_well_e / S)
```

Puis normalisation print:

```text
PDR_print = PDR_pixel + log2(sqrt(total_pixels) / sqrt(8_000_000))
```

Cette normalisation rend les boitiers de resolutions differentes plus comparables.

### 3.7 Controles qualite

Chaque session doit produire des indicateurs de confiance:

- Nombre de patches valides.
- Nombre de patches rejetes pour clipping.
- Ecart black level mesure vs metadata.
- Score de detection de grille.
- Erreur de reprojection homographie.
- `R2` de la Photon Transfer Curve.
- Courbes debug: PTC, read noise vs ISO, gain vs ISO, PDR vs ISO.
- Previews: source, rectified, overlay source, overlay rectified.

## 4. Architecture cible Python

Le coeur reste Python. La GUI peut rester utile, mais le pipeline doit devenir
automatisable en ligne de commande.

Architecture cible:

```text
sony_corr/
  patterns/       generation et description des mires
  ingest/         detection RAW, EXIF, tri par ISO, pairing dark/chart
  geometry/       detection grille, homographie, manual fit, scoring
  extraction/     extraction patches, masques, rejet clipping/outliers
  metrics/        read noise, PTC, gain, full well, PDR
  storage/        schemas resultats, archivage, aggregation
  reports/        graphs, exports, white paper figures
  web/            future interface HTML/API
```

Etapes de migration:

1. Garder les scripts actuels fonctionnels.
2. Extraire progressivement les fonctions stables depuis `step1_sort.py`,
   `step2_rectify.py` et `step3_analyze.py`.
3. Creer un runner CLI unique, par exemple:

```bash
python3 -m sony_corr run --input samples/raw --output data/results --project sony_a7iv_test
```

4. Garder la GUI comme outil de correction manuelle et de diagnostic.
5. Ajouter des tests unitaires sur les calculs et des tests integration sur images synthetiques.

## 5. Detection robuste malgre defocus et pixels d'ecran

Le defocus est une contrainte voulue, pas une erreur. La detection doit donc chercher
des structures robustes a basse frequence, pas des pixels nets.

Strategie cible:

- Garder des marqueurs larges aux coins et des marqueurs d'orientation.
- Utiliser une detection multi-strategie:
  - markers larges si visibles;
  - profils 1D horizontaux/verticaux;
  - detection des separations sombres;
  - fallback homographie manuelle.
- Mesurer un score de confiance pour chaque strategie.
- Choisir automatiquement la meilleure grille valide.
- Autoriser plusieurs patterns:
  - grille standard 11 x 7;
  - grille minimaliste avec fond noir;
  - mire plus espacee pour defocus fort;
  - mire haute densite pour ecrans haute resolution.

Critere de reussite:

- Detection automatique sur la majorite des chart frames bien exposes.
- Correction manuelle possible quand le score est faible.
- Temps de traitement raisonnable par image, en evitant les recherches exhaustives.
- Extraction de patches stable meme si les bords sont flous.

## 6. Archivage propre et donnees exploitables big data

Le projet doit pouvoir accumuler plusieurs samples pour la meme camera.
Il faut separer les fichiers bruts, les artefacts de traitement et les resultats
normalises.

Arborescence proposee:

```text
data/
  raw/
    sony/
      a7iv/
        sample_001/
          session_2026-06-03/
  processed/
    sony/
      a7iv/
        sample_001/
          session_2026-06-03/
            iso_100/
            iso_200/
  results/
    sony/
      a7iv/
        sample_001/
          session_2026-06-03/
            session_results.json
            session_results.csv
            quality_report.json
  aggregate/
    sony/
      a7iv/
        aggregate_results.csv
        aggregate_results.parquet
```

Metadonnees minimales par session:

- `brand`: marque camera.
- `model`: modele camera.
- `body_serial_hash`: identifiant anonymise du boitier si possible.
- `sample_id`: identifiant local du sample.
- `session_id`: date ou UUID.
- `raw_format`: ARW, DNG, etc.
- `lens`: objectif si disponible.
- `screen_model`: modele ecran si connu.
- `screen_resolution`: resolution d'affichage de la mire.
- `pattern_version`: version de la mire.
- `pipeline_version`: commit git ou version package.
- `temperature`: optionnel.
- `operator_notes`: notes libres.

Resultats par ISO:

- ISO.
- Black level mesure et metadata.
- White level.
- Read noise ADU.
- Read noise e-.
- Gain e-/ADU.
- Full well e-.
- EDR.
- PDR pixel.
- PDR print.
- Scores qualite et warnings.

Aggregation multi-samples:

- Moyenne par ISO.
- Mediane par ISO.
- Ecart-type.
- Nombre de samples.
- Detection d'outliers.
- Comparaison entre boitiers du meme modele.

## 7. Future couche HTML participative

La partie HTML ne doit pas remplacer Python. Elle doit servir de couche simple pour
ajouter des samples, renseigner les metadonnees et visualiser les resultats.

V1 locale:

- Formulaire HTML pour ajouter un dossier ou des fichiers RAW.
- Champs metadata: marque, modele, sample, ecran, notes.
- Bouton pour lancer le pipeline Python.
- Page statut: fichiers detectes, paires ISO, warnings.
- Page resultats: graphiques PDR, read noise, gain, PTC.

V2 participative:

- Upload de samples par contributeurs.
- Validation des metadonnees.
- Controle qualite automatique avant publication.
- Export public des resultats agreges.
- Possibilite de comparer plusieurs cameras.

Contraintes:

- Ne jamais exposer les RAW sans consentement explicite.
- Anonymiser les numeros de serie.
- Garder une trace du protocole de capture.
- Associer chaque resultat a une version du pipeline et de la mire.

## 8. Roadmap proposee

### Phase 1 - Documentation et schemas

- Finaliser ce plan.
- Ajouter un schema JSON pour les sessions et resultats.
- Documenter les calculs dans un white paper plus formel.
- Ajouter une section README pointant vers `planning/plan.md`.

### Phase 2 - Pipeline automatique

- Creer un package Python structure.
- Ajouter un runner CLI de bout en bout.
- Stabiliser le tri dark/chart.
- Ajouter un score de detection de grille.
- Sauvegarder tous les artefacts dans une arborescence normalisee.

### Phase 3 - Robustesse scientifique

- Ameliorer la variance robuste.
- Ajouter outlier rejection sur patches.
- Ajouter scores PTC et warnings.
- Comparer plusieurs samples d'une meme camera.
- Produire CSV/Parquet agreges.

### Phase 4 - Interface HTML

- Creer une petite interface locale.
- Ajouter upload/selection samples.
- Afficher progression et resultats.
- Preparer une architecture participative.

## 9. Definition de succes

Le projet sera considere comme pret pour une premiere vraie campagne de mesure quand:

- Un dossier de RAW peut etre traite en une commande.
- Les resultats par ISO sont sauvegardes dans un format stable.
- Les overlays permettent de valider rapidement la grille.
- Les courbes PDR, read noise et gain sont produites automatiquement.
- Les metadonnees permettent d'agreger plusieurs samples d'une meme camera.
- Les calculs sont suffisamment documentes pour etre compares aux references PhotonsToPhotos.
