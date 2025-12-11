import rawpy
import numpy as np
import cv2
import tifffile
import sys
import os
from scipy.spatial import ConvexHull

def detect_grid_features(image_rgb):
    """
    Détecte la grille de patchs colorés.
    Retourne :
    - corners_geom : Les coordonnées (x,y) des 4 patchs aux coins géométriques de l'image [TL, TR, BR, BL]
    - debug_img : Image avec dessins pour debug
    - valid_contours : Liste des contours validés comme patchs
    - indices : Indices dans valid_contours correspondant aux 4 coins
    """
    # 1. Preprocessing
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Normalisation pour maximiser le contraste patchs/fond
    p_low, p_high = np.percentile(gray, (1, 99))
    if p_high > p_low:
        gray_norm = np.clip((gray - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
    else:
        gray_norm = gray
    
    # Thresholding
    # Otsu pour séparer les patchs (gris/blanc) du fond (noir)
    # Si Otsu échoue (ex: trop de noir), on peut essayer un seuil adaptatif
    # Block size très grand pour couvrir un patch entier (qui fait ~100-300px)
    thresh = cv2.adaptiveThreshold(gray_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 201, -2) 
    print(f"Adaptive Threshold utilisé (Block=201)")
    
    # Nettoyage morphologique (utile si defocus important)
    # On ferme pour combler les trous dans les patchs, on ouvre pour virer le bruit
    # Avec un gros defocus, les patchs peuvent se toucher ou être très flous
    # On réduit le kernel pour éviter de fusionner les patchs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cv2.imwrite("debug_grid_thresh.jpg", thresh)
    
    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patches = []
    valid_contours = []
    debug_img = image_rgb.copy()
    
    # Filtrage des contours
    # On filtre d'abord les tout petits trucs
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        print("Aucun contour significatif trouvé.")
        return None, debug_img, None, None
        
    median_area = np.median(areas)
    print(f"Aire médiane des contours : {median_area}")
    
    # Si l'aire médiane est énorme (ex: tout est fusionné), c'est un problème
    # Si l'aire médiane est minuscule (bruit), c'est un problème
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # Critères larges car defocus change la taille apparente
        # On accepte une plage très large autour de la médiane
        if area > median_area * 0.1 and area < median_area * 10:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                patches.append([cX, cY])
                valid_contours.append(cnt)
                cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
        else:
             cv2.drawContours(debug_img, [cnt], -1, (0, 0, 255), 2) # Rejeté en rouge
    
    print(f"Patchs candidats trouvés : {len(patches)}")
    
    if len(patches) < 10: 
        print("Pas assez de patchs trouvés pour former une grille.")
        return None, debug_img, None, None
        
    # 2. Trouver les 4 coins de la grille
    points = np.array(patches)
    
    # On cherche les points extrêmes géométriquement
    # TL: min(x+y), BR: max(x+y)
    # TR: min(y-x) ou max(x-y), BL: max(y-x) ou min(x-y)
    # Note: y est en bas dans opencv
    
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1) # y - x (si points est [x, y]) -> non diff fait col(i+1) - col(i)
    # points est shape (N, 2) -> [[x, y], ...]
    # diff(points, axis=1) donne y - x
    
    diff_yx = points[:, 1] - points[:, 0]
    
    idx_tl = np.argmin(s)          # x+y min
    idx_br = np.argmax(s)          # x+y max
    idx_tr = np.argmin(diff_yx)    # y-x min => y petit, x grand
    idx_bl = np.argmax(diff_yx)    # y-x max => y grand, x petit
    
    # Les 4 coins géométriques [TL, TR, BR, BL]
    corners_geom = np.array([patches[idx_tl], patches[idx_tr], patches[idx_br], patches[idx_bl]], dtype="float32")
    indices = [idx_tl, idx_tr, idx_br, idx_bl]
    
    # --- RAFFINEMENT DES COINS ---
    # Parfois le min(x+y) chope un patch qui n'est pas vraiment le coin si la grille est tordue
    # On va essayer de trouver les coins de l'enveloppe convexe
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # On cherche dans hull_points ceux qui sont les plus proches des coins de la bounding box
        x, y, w, h = cv2.boundingRect(hull_points.astype(np.int32))
        bbox_corners = np.array([
            [x, y],         # TL
            [x+w, y],       # TR
            [x+w, y+h],     # BR
            [x, y+h]        # BL
        ])
        
        refined_indices = []
        refined_corners = []
        
        for target in bbox_corners:
            # Trouver le point de l'enveloppe le plus proche de ce coin de bbox
            dists = np.linalg.norm(hull_points - target, axis=1)
            best_idx_in_hull = np.argmin(dists)
            best_pt = hull_points[best_idx_in_hull]
            
            # Retrouver l'index original dans 'patches'
            # C'est un peu lourd, on compare les coords
            for i, p in enumerate(patches):
                if np.array_equal(p, best_pt):
                    refined_indices.append(i)
                    refined_corners.append(p)
                    break
        
        if len(refined_corners) == 4:
            print("Coins raffinés via ConvexHull.")
            corners_geom = np.array(refined_corners, dtype="float32")
            indices = refined_indices
            
    except Exception as e:
        print(f"Erreur ConvexHull: {e}, fallback sur méthode simple.")

    # Dessin des coins détectés
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)] # RGB: R, G, B, Cyan
    labels = ["TLg", "TRg", "BRg", "BLg"]
    for i, pt in enumerate(corners_geom):
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 20, colors[i], -1)
        cv2.putText(debug_img, labels[i], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 3)
        
    cv2.imwrite("debug_grid_detected.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    return corners_geom, debug_img, valid_contours, indices

def get_patch_luminance(image_gray, contour):
    mask = np.zeros(image_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(image_gray, mask=mask)[0]
    return mean_val

def rectify_raw_grid(raw_path, output_path):
    print(f"Traitement de : {raw_path}")
    
    # 1. Lecture et Démosaïcage rapide
    with rawpy.imread(raw_path) as raw:
        # Auto-bright False pour garder la linéarité relative des couleurs si possible, 
        # mais pour la détection on veut voir quelque chose.
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0, user_sat=None)
        raw_data = raw.raw_image_visible.astype(np.float32)
        
    h, w = raw_data.shape
    print(f"Dimensions RAW : {w}x{h}")
    
    # 2. Détection
    corners_geom, debug_img, contours, indices = detect_grid_features(rgb)
    
    if corners_geom is None:
        print("Echec de la détection de la grille.")
        return

    # 3. Analyse Orientation (Luminosité)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lums = []
    labels_geom = ["TL_geom", "TR_geom", "BR_geom", "BL_geom"]
    
    print("\nAnalyse de luminosité des coins :")
    for i, idx in enumerate(indices):
        lum = get_patch_luminance(gray, contours[idx])
        lums.append(lum)
        print(f"  {labels_geom[i]} : {lum:.2f}")
        
    # Le plus sombre = Start Color (Haut-Gauche théorique)
    # Le plus clair = End Color (Bas-Droite théorique)
    idx_darkest = np.argmin(lums) 
    idx_lightest = np.argmax(lums)
    
    print(f"  -> Plus sombre (Start/TL) : {labels_geom[idx_darkest]}")
    print(f"  -> Plus clair (End/BR)    : {labels_geom[idx_lightest]}")
    
    # Rotation pour mettre le Darkest en premier (TL)
    # corners_geom est [TL, TR, BR, BL] (sens horaire)
    # On veut reordonner pour que src_pts[0] soit le patch sombre
    
    # Si idx_darkest = 0 (TLg), shift = 0 -> [TL, TR, BR, BL]
    # Si idx_darkest = 1 (TRg), shift = 1 -> [TR, BR, BL, TL] (Rotation 90°)
    # etc.
    
    # np.roll décale vers la droite (dernier devient premier) si shift positif
    # Ici on veut que l'élément à idx_darkest devienne l'élément 0.
    # Donc on shift vers la gauche de idx_darkest positions.
    src_pts = np.roll(corners_geom, -idx_darkest, axis=0)
    
    # Vérif cohérence
    # Le plus clair doit être à l'index 2 (BR) dans la liste tournée
    # On recalcule l'index du plus clair dans la nouvelle liste
    # Si idx_lightest était 3 et idx_darkest 1 -> new_idx = (3-1)%4 = 2. Correct.
    expected_lightest_idx = (idx_lightest - idx_darkest) % 4
    
    if expected_lightest_idx != 2:
        print(f"ATTENTION : Le patch le plus clair n'est pas opposé au plus sombre (idx={expected_lightest_idx} vs 2).")
        print("L'orientation risque d'être incorrecte ou l'éclairage est très inégal.")
        
        # Tentative de correction heuristique :
        # Si on a détecté TL et TR comme extrêmes (idx 0 et 1), c'est peut-être juste une inversion gauche/droite ou haut/bas locale
        # Mais si on a TL et BL (0 et 3), c'est bizarre.
        # On fait confiance au plus sombre comme ancre (Start Color).
    else:
        print("Orientation confirmée par la position du patch le plus clair.")

    # 4. Homographie
    # Définition de la grille cible redressée
    # On veut une image finale qui contient juste la grille proprement
    # Grille 11 cols x 7 rows
    cols = 11
    rows = 7
    
    # Paramètres de sortie arbitraires (mais haute def)
    patch_size = 100
    spacing = 0 # On peut coller les patchs ou garder l'espace. Gardons l'espace proportionnel.
    # Dans la grille générée : patch ~ 300px, spacing ~ 20px -> spacing ~ 0.06 * patch
    # Simplifions : on mappe les CENTRES des patchs.
    
    # On définit une marge
    margin = 100
    
    # Coordonnées des CENTRES cibles
    # TL (0,0) -> x = margin + patch_size/2, y = margin + patch_size/2
    # TR (10,0) -> x = margin + patch_size/2 + 10*step_x
    
    step_x = patch_size * 1.1 # 10% spacing
    step_y = patch_size * 1.1
    
    # Centres théoriques
    dst_tl = [margin, margin]
    dst_tr = [margin + (cols-1)*step_x, margin]
    dst_br = [margin + (cols-1)*step_x, margin + (rows-1)*step_y]
    dst_bl = [margin, margin + (rows-1)*step_y]
    
    dst_pts = np.array([dst_tl, dst_tr, dst_br, dst_bl], dtype="float32")
    
    # Taille image finale
    out_w = int(dst_tr[0] + margin)
    out_h = int(dst_bl[1] + margin)
    
    print(f"Dimensions sortie : {out_w}x{out_h}")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # 5. Application aux plans RAW
    planes = [
        raw_data[0::2, 0::2], # R
        raw_data[0::2, 1::2], # G1
        raw_data[1::2, 0::2], # G2
        raw_data[1::2, 1::2]  # B
    ]
    
    warped_planes = []
    
    # Adaptation matrice pour demi-résolution
    S = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    M_scaled = S @ M @ np.linalg.inv(S)
    
    # La sortie est aussi divisée par 2
    out_w_half = out_w // 2
    out_h_half = out_h // 2
    
    print("Rectification des plans RAW...")
    for p in planes:
        warped = cv2.warpPerspective(p, M_scaled, (out_w_half, out_h_half), flags=cv2.INTER_LINEAR)
        warped_planes.append(warped)
        
    # 6. Export
    stack = np.stack(warped_planes, axis=0)
    
    description = (
        "Rectified RAW Grid (Grid Detection)\n"
        "Method: Corner Patches Luminance Detection\n"
        "Channels: 0=R, 1=G1, 2=G2, 3=B\n"
    )
    
    tifffile.imwrite(output_path, stack, photometric='minisblack', description=description)
    print(f"Sauvegardé : {output_path}")
    
    # Preview
    r = warped_planes[0]
    g = (warped_planes[1] + warped_planes[2]) / 2
    b = warped_planes[3]
    preview = np.dstack((r, g, b))
    preview = preview / np.max(preview) * 255
    cv2.imwrite(output_path.replace(".tiff", "_preview.jpg"), preview.astype(np.uint8))
    print(f"Preview : {output_path.replace('.tiff', '_preview.jpg')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rectify_raw_grid.py <fichier_raw.ARW>")
    else:
        input_file = sys.argv[1]
        output_file = input_file + ".rectified.tiff"
        rectify_raw_grid(input_file, output_file)
