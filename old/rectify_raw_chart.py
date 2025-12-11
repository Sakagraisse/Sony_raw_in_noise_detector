import rawpy
import numpy as np
import cv2
import tifffile
import sys
import os

def detect_features(image_rgb):
    """
    Détecte les 4 coins (gros cercles) et les 2 marqueurs d'orientation (petits cercles).
    Retourne les coordonnées des centres.
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    print(f"Stats Gray: Mean={np.mean(gray):.1f}, Std={np.std(gray):.1f}, Min={np.min(gray)}, Max={np.max(gray)}")
    
    # Normalisation robuste (Percentile stretching)
    p_low, p_high = np.percentile(gray, (1, 99))
    print(f"Percentiles: 1%={p_low}, 99%={p_high}")
    
    if p_high > p_low:
        gray_norm = np.clip((gray - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
    else:
        gray_norm = gray
        
    # Binarisation (les marqueurs sont noirs sur fond blanc/gris)
    # On inverse pour avoir les marqueurs en blanc
    # Utilisation d'Adaptive Threshold pour gérer le vignettage et l'éclairage inégal
    thresh = cv2.adaptiveThreshold(gray_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 41, 5)
    print(f"Adaptive Threshold utilisé")
    
    # Sauvegarde debug
    cv2.imwrite("debug_gray.jpg", gray_norm)
    cv2.imwrite("debug_thresh.jpg", thresh)
    
    # Détection des contours
    # RETR_LIST pour tout récupérer, même si c'est imbriqué
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    large_blobs = []
    small_blobs = []
    
    debug_img = image_rgb.copy()
    
    print(f"Nombre total de contours : {len(contours)}")
    
    # Filtrage par taille
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # On cherche des cercles
        # Circularité : 1.0 pour un cercle parfait, ~0.785 pour un carré
        # On met le seuil à 0.82 pour exclure les patchs carrés (qui sont nombreux)
        if circularity > 0.82: 
            # Calcul du centre
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Classification par taille
                # Les coins sont ~3500px, les marqueurs ~850px (sur l'image test)
                # On abaisse le seuil min pour supporter des cadrages plus larges
                if area > 1000:
                    large_blobs.append((cX, cY))
                    cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 3) # Vert pour les gros
                    print(f"Contour {i}: Area={area:.1f}, Circ={circularity:.2f} -> LARGE (Coin)")
                elif 100 < area <= 1000:
                    small_blobs.append((cX, cY))
                    cv2.drawContours(debug_img, [cnt], -1, (0, 0, 255), 3) # Rouge pour les petits
                    print(f"Contour {i}: Area={area:.1f}, Circ={circularity:.2f} -> SMALL (Marqueur)")
                else:
                     # Trop petit
                     pass
        else:
            # Debug pour voir ce qu'on rejette (si c'est un gros truc)
            if area > 500:
                print(f"Contour {i} rejeté (forme): Area={area:.1f}, Circ={circularity:.2f}")

    cv2.imwrite("debug_contours.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                    
    return large_blobs, small_blobs

def order_corners(corners):
    """
    Ordonne les coins : [Haut-Gauche, Haut-Droit, Bas-Droit, Bas-Gauche]
    Basé sur la somme et la différence des coordonnées.
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(corners)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # HG
    rect[2] = pts[np.argmax(s)] # BD
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # HD
    rect[3] = pts[np.argmax(diff)] # BG
    
    return rect

def rectify_raw(raw_path, output_path):
    print(f"Traitement de : {raw_path}")
    
    # 1. Lecture et Démosaïcage rapide pour la détection
    with rawpy.imread(raw_path) as raw:
        # On utilise l'image visible (souvent un JPEG intégré ou un demosaic rapide)
        # Pour la détection, on veut une image bien exposée, donc on laisse l'auto-bright
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0, user_sat=None)
        
        # Données RAW Bayer
        raw_data = raw.raw_image_visible.astype(np.float32)
        
    h, w = raw_data.shape
    print(f"Dimensions RAW : {w}x{h}")
    
    # 2. Détection des marqueurs
    corners, markers = detect_features(rgb)
    
    if len(corners) != 4:
        print(f"ERREUR : 4 coins attendus, {len(corners)} trouvés.")
        return
    
    if len(markers) < 2:
        print(f"ERREUR : Au moins 2 marqueurs d'orientation attendus, {len(markers)} trouvés.")
        # On continue peut-être juste avec les coins, mais l'orientation sera incertaine
        # return 

    print(f"Coins détectés : {len(corners)}")
    print(f"Marqueurs orientation : {len(markers)}")

    # 3. Identification de l'orientation
    # On ordonne les coins géométriquement d'abord (HG, HD, BD, BG dans le repère de l'image actuelle)
    ordered_corners = order_corners(corners)
    (tl, tr, br, bl) = ordered_corners
    
    # On cherche quel marqueur est "Top" et lequel est "Right"
    # Le marqueur "Top" est proche du milieu du segment TL-TR
    # Le marqueur "Right" est proche du milieu du segment TR-BR
    
    # Centres théoriques des bords
    mid_top = (tl + tr) / 2
    mid_right = (tr + br) / 2
    mid_bottom = (bl + br) / 2
    mid_left = (tl + bl) / 2
    
    # Trouver le marqueur le plus proche de chaque milieu
    def get_closest_marker(target_pt, markers_list):
        if not markers_list: return None, float('inf')
        dists = [np.linalg.norm(np.array(m) - target_pt) for m in markers_list]
        min_dist = min(dists)
        return markers_list[np.argmin(dists)], min_dist

    # On teste les 4 rotations possibles pour voir laquelle aligne les marqueurs
    # Configuration cible : Top Marker en haut, Right Marker à droite.
    
    # Simplification : On va déterminer la rotation nécessaire
    # Si le marqueur "Top" (le petit point seul sur un bord long) est en fait en bas, il faut tourner de 180°.
    # Si il est à gauche, +90°, etc.
    
    # On identifie les marqueurs présents
    # Il y a un marqueur au milieu du bord "Haut" (dans le repère de la mire)
    # Il y a un marqueur au milieu du bord "Droit" (dans le repère de la mire)
    
    # On cherche les marqueurs proches des 4 bords
    m_top, d_top = get_closest_marker(mid_top, markers)
    m_right, d_right = get_closest_marker(mid_right, markers)
    m_bottom, d_bottom = get_closest_marker(mid_bottom, markers)
    m_left, d_left = get_closest_marker(mid_left, markers)
    
    rotation = 0 # 0, 90, 180, 270 (sens horaire)
    
    # Logique de détection d'orientation
    # On suppose que d_top < seuil signifie "Il y a un marqueur en haut"
    threshold = 100 # pixels
    
    has_top = d_top < threshold
    has_right = d_right < threshold
    has_bottom = d_bottom < threshold
    has_left = d_left < threshold
    
    print(f"Marqueurs trouvés aux bords : Haut={has_top}, Droite={has_right}, Bas={has_bottom}, Gauche={has_left}")
    
    if has_top and has_right:
        print("Orientation : Correcte (0°)")
        src_pts = ordered_corners
    elif has_right and has_bottom:
        print("Orientation : 90° Horaire (Mire tournée vers la droite)")
        # Le "Haut" de la mire est à Droite de l'image
        # TL de la mire est TR de l'image
        src_pts = np.array([tr, br, bl, tl], dtype="float32")
    elif has_bottom and has_left:
        print("Orientation : 180° (Mire à l'envers)")
        src_pts = np.array([br, bl, tl, tr], dtype="float32")
    elif has_left and has_top:
        print("Orientation : 270° Horaire (Mire tournée vers la gauche)")
        src_pts = np.array([bl, tl, tr, br], dtype="float32")
    else:
        print("ATTENTION : Orientation ambiguë. On suppose 0°.")
        src_pts = ordered_corners

    # 4. Calcul de la transformation (Homographie)
    # Dimensions cibles (Ratio 16:9, on peut prendre une taille fixe ou celle de l'image)
    # On va garder la largeur max détectée pour maximiser la résolution
    widthA = np.linalg.norm(src_pts[2] - src_pts[3])
    widthB = np.linalg.norm(src_pts[1] - src_pts[0])
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(src_pts[1] - src_pts[2])
    heightB = np.linalg.norm(src_pts[0] - src_pts[3])
    maxHeight = max(int(heightA), int(heightB))
    
    # On force un ratio proche de 16:9 si on veut être strict, mais gardons la géométrie détectée
    # Points destination (Rectangle parfait)
    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Matrice de perspective
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # 5. Application au RAW (Plan par Plan)
    # Pour ne pas casser le Bayer, on sépare les 4 phases CFA
    # R G
    # G B
    # On suppose un pattern 2x2 standard.
    
    # Extraction des sous-plans (H/2, W/2)
    planes = [
        raw_data[0::2, 0::2], # R (ou autre selon pattern)
        raw_data[0::2, 1::2], # G1
        raw_data[1::2, 0::2], # G2
        raw_data[1::2, 1::2]  # B
    ]
    
    warped_planes = []
    
    # On doit adapter la matrice M pour les plans qui sont 2x plus petits
    # M map (x,y) -> (x',y'). Pour (x/2, y/2) -> (x'/2, y'/2), on divise les translations par 2
    # et on garde l'échelle.
    # M_scaled = S * M * inv(S) où S est une matrice de scale 0.5
    S = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    M_scaled = S @ M @ np.linalg.inv(S)
    
    out_w = maxWidth // 2
    out_h = maxHeight // 2
    
    print("Rectification des plans RAW...")
    for p in planes:
        # Interpolation : INTER_LINEAR est acceptable ici car on est DANS un plan monochrome
        # Cela lisse un peu le bruit (attention pour l'analyse de bruit pur), 
        # mais c'est nécessaire pour la rectification géométrique.
        # Pour une analyse de bruit pure sans interpolation, il faudrait ne faire que crop/rotate 90.
        # Mais l'utilisateur demande "redressé" (perspective).
        warped = cv2.warpPerspective(p, M_scaled, (out_w, out_h), flags=cv2.INTER_LINEAR)
        warped_planes.append(warped)
        
    # 6. Export
    # On sauvegarde en TIFF multi-canal (4 canaux) float32
    # Ordre : R, G1, G2, B (conventionnel, mais dépend du pattern CFA initial)
    stack = np.stack(warped_planes, axis=0) # Shape (4, H, W)
    
    # Métadonnées pour expliquer pourquoi on fait ça
    description = (
        "Rectified RAW Data (CFA Planes Separated)\n"
        "Purpose: Automated SNR and Dynamic Range Analysis (Photons to Photos style)\n"
        "Channels: 0=Phase00, 1=Phase01, 2=Phase10, 3=Phase11\n"
        "Geometry: Perspective corrected based on chart markers."
    )
    
    tifffile.imwrite(output_path, stack, photometric='minisblack', description=description)
    print(f"Sauvegardé : {output_path} (TIFF 4 canaux float32)")
    
    # Génération d'une preview RGB pour vérification
    # On moyenne les 2 verts et on assemble
    r = warped_planes[0]
    g = (warped_planes[1] + warped_planes[2]) / 2
    b = warped_planes[3]
    
    # Normalisation pour affichage
    preview = np.dstack((r, g, b))
    preview = preview / np.max(preview) * 255
    cv2.imwrite(output_path.replace(".tiff", "_preview.jpg"), preview.astype(np.uint8))
    print(f"Preview sauvegardée : {output_path.replace('.tiff', '_preview.jpg')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rectify_raw_chart.py <fichier_raw.ARW>")
    else:
        input_file = sys.argv[1]
        output_file = input_file + ".rectified.tiff"
        rectify_raw(input_file, output_file)
