import rawpy
import numpy as np
import cv2
import tifffile
import sys
import os
from scipy.spatial import ConvexHull, distance

# Couleurs de référence (RGB)
REF_START_COLOR = np.array([26, 11, 18])   # Haut-Gauche
REF_END_COLOR = np.array([254, 151, 223])  # Bas-Droite

def get_roi_mean_color(image_rgb, contour):
    """Calcule la couleur moyenne (R,G,B) à l'intérieur d'un contour"""
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(image_rgb, mask=mask)
    return np.array(mean_val[:3])

def detect_grid_and_orient(image_rgb):
    """
    Détecte la grille, identifie les 4 coins et l'orientation.
    """
    h, w = image_rgb.shape[:2]
    debug_img = image_rgb.copy()
    
    # 1. Preprocessing
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Normalisation
    p_low, p_high = np.percentile(gray, (1, 99))
    if p_high > p_low:
        gray_norm = np.clip((gray - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
    else:
        gray_norm = gray
        
    # Adaptive Threshold (Block size large pour les patchs flous)
    thresh = cv2.adaptiveThreshold(gray_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 151, -2)
    
    # Nettoyage
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite("debug_hybrid_thresh.jpg", thresh)
    
    # 2. Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrage initial
    min_area = (h * w) / 10000 # Ex: 12MP -> 1200px min
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        print("Aucun contour significatif.")
        return None, debug_img
        
    # Filtrage par médiane (on cherche les patchs de la grille)
    areas = [cv2.contourArea(c) for c in valid_contours]
    median_area = np.median(areas)
    print(f"Aire médiane estimée des patchs : {median_area:.0f}")
    
    # On garde les contours qui sont proches de la médiane (les patchs)
    # Et aussi les un peu plus gros (les marqueurs coins)
    patch_candidates = []
    patch_centers = []
    
    for c in valid_contours:
        area = cv2.contourArea(c)
        # Patchs : 0.5x à 2x médiane
        # Marqueurs : > 2x médiane
        if 0.2 * median_area < area < 5 * median_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                patch_candidates.append(c)
                patch_centers.append([cX, cY])
                cv2.drawContours(debug_img, [c], -1, (0, 255, 0), 2)
    
    print(f"Candidats patchs retenus : {len(patch_candidates)}")
    
    if len(patch_candidates) < 10:
        print("Pas assez de patchs pour une grille.")
        return None, debug_img

    # 3. Identification des 4 coins de la grille via Convex Hull
    points = np.array(patch_centers)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # L'enveloppe convexe contient les coins extérieurs.
    # Sur une grille rectangulaire projetée, l'enveloppe a souvent 4 sommets principaux,
    # mais peut en avoir plus si les bords sont courbés ou si on a chopé du bruit.
    # On va chercher les 4 points de l'enveloppe qui forment le quadrilatère d'aire maximale
    # Ou plus simple : on prend l'approximation polygonale à 4 sommets de l'enveloppe.
    
    # Conversion pour approxPolyDP
    hull_cnt = hull_points.reshape((-1, 1, 2)).astype(np.int32)
    
    # On simplifie le contour jusqu'à avoir 4 points
    epsilon = 0.01 * cv2.arcLength(hull_cnt, True)
    approx = cv2.approxPolyDP(hull_cnt, epsilon, True)
    
    while len(approx) > 4:
        epsilon *= 1.5
        approx = cv2.approxPolyDP(hull_cnt, epsilon, True)
        if epsilon > 1000: break # Sécurité
        
    if len(approx) != 4:
        print(f"Impossible de réduire l'enveloppe à 4 coins (trouvé {len(approx)}).")
        # Fallback : Bounding Rect orienté ? Non, on prend les 4 points de l'enveloppe les plus proches des coins de l'image
        img_corners = np.array([[0,0], [w,0], [w,h], [0,h]])
        grid_corners = []
        for corner in img_corners:
            dists = np.linalg.norm(hull_points - corner, axis=1)
            grid_corners.append(hull_points[np.argmin(dists)])
        grid_corners = np.array(grid_corners)
    else:
        grid_corners = approx.reshape(4, 2)

    # Dessin des coins bruts
    for pt in grid_corners:
        cv2.circle(debug_img, (pt[0], pt[1]), 15, (255, 0, 0), -1)

    # 4. Orientation par couleur
    # On doit associer chaque coin géométrique à un patch réel pour mesurer sa couleur
    ordered_corners = []
    corner_colors = []
    
    for pt in grid_corners:
        # Trouver le contour le plus proche de ce point géométrique
        dists = [np.linalg.norm(np.array(c) - pt) for c in patch_centers]
        idx_best = np.argmin(dists)
        best_cnt = patch_candidates[idx_best]
        best_center = patch_centers[idx_best]
        
        ordered_corners.append(best_center)
        
        # Mesure couleur
        color = get_roi_mean_color(image_rgb, best_cnt)
        corner_colors.append(color)
        
    ordered_corners = np.array(ordered_corners)
    corner_colors = np.array(corner_colors)
    
    # Comparaison avec REF_START et REF_END
    # On cherche l'index qui minimise la distance à START
    dists_start = [np.linalg.norm(c - REF_START_COLOR) for c in corner_colors]
    idx_tl = np.argmin(dists_start)
    
    # On cherche l'index qui minimise la distance à END
    dists_end = [np.linalg.norm(c - REF_END_COLOR) for c in corner_colors]
    idx_br = np.argmin(dists_end)
    
    print(f"Index supposé TL (Start Color) : {idx_tl} (Dist: {dists_start[idx_tl]:.1f})")
    print(f"Index supposé BR (End Color)   : {idx_br} (Dist: {dists_end[idx_br]:.1f})")
    
    # Réorganisation : On veut [TL, TR, BR, BL]
    # On sait qui est TL et qui est BR.
    # Il faut trouver TR et BL.
    # Dans un ordre cyclique (convex hull), TR est après TL (ou avant), et BR est opposé.
    
    # On va trier les points pour avoir un ordre géométrique connu (ex: tri par angle autour du centre)
    center = np.mean(ordered_corners, axis=0)
    angles = np.arctan2(ordered_corners[:,1] - center[1], ordered_corners[:,0] - center[0])
    # Sort clockwise starting from top-left?
    # Plus simple : on a identifié TL.
    # On prend le point TL.
    # On cherche le point BR (le plus loin de TL ? ou celui identifié par couleur).
    # On utilise celui identifié par couleur c'est le plus sûr.
    
    pt_tl = ordered_corners[idx_tl]
    pt_br = ordered_corners[idx_br]
    
    # Les deux autres points
    other_indices = [i for i in range(4) if i != idx_tl and i != idx_br]
    p1 = ordered_corners[other_indices[0]]
    p2 = ordered_corners[other_indices[1]]
    
    # Pour distinguer TR de BL :
    # Produit vectoriel ou position relative.
    # Vecteur TL -> BR.
    # TR doit être "au dessus" (ou à gauche/droite selon repère).
    # Plus simple : TR est celui qui a le x le plus grand (dans le repère tourné) ?
    # Non, si l'image est tournée à 90°.
    
    # Utilisons le produit vectoriel (Cross Product)
    # Vec A = BR - TL
    # Vec B = P - TL
    # Cross = Ax * By - Ay * Bx
    # Le signe nous dit de quel côté est le point.
    
    vec_main = pt_br - pt_tl
    vec_p1 = p1 - pt_tl
    cross_p1 = vec_main[0] * vec_p1[1] - vec_main[1] * vec_p1[0]
    
    # Dans le repère image (y vers le bas) :
    # TR devrait être "au dessus" de la diagonale TL-BR -> Cross négatif (ou positif selon convention)
    # BL devrait être "en dessous" -> Signe opposé.
    
    # Testons visuellement : TL(0,0), BR(10,10). TR(10,0).
    # Main=(10,10). P=(10,0).
    # Cross = 10*0 - 10*10 = -100.
    # BL(0,10). P=(0,10).
    # Cross = 10*10 - 10*0 = 100.
    # Donc TR a un cross négatif, BL a un cross positif.
    
    if cross_p1 < 0:
        pt_tr = p1
        pt_bl = p2
    else:
        pt_tr = p2
        pt_bl = p1
        
    final_corners = np.array([pt_tl, pt_tr, pt_br, pt_bl], dtype="float32")
    
    # Debug labels
    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(final_corners):
        cv2.putText(debug_img, labels[i], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
    cv2.imwrite("debug_hybrid_detected.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    return final_corners, debug_img

def rectify_raw_hybrid(raw_path, output_path):
    print(f"Traitement de : {raw_path}")
    
    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0, user_sat=None)
        raw_data = raw.raw_image_visible.astype(np.float32)
        
    src_pts, debug_img = detect_grid_and_orient(rgb)
    
    if src_pts is None:
        print("Echec détection.")
        return

    # Dimensions cibles
    # On veut mapper la grille (11x7 patchs)
    # On définit une taille de patch arbitraire pour la sortie
    PATCH_SIZE = 100
    COLS = 11
    ROWS = 7
    
    # On ajoute une petite marge autour de la grille pour ne pas couper les patchs du bord
    MARGIN = PATCH_SIZE // 2
    
    # Largeur/Hauteur de la zone GRILLE (de centre TL à centre BR)
    # TL est à (MARGIN, MARGIN)
    # TR est à (MARGIN + (COLS-1)*STEP, MARGIN)
    # STEP = PATCH_SIZE + SPACING.
    # Dans la mire générée, le spacing est petit. Disons STEP = PATCH_SIZE * 1.1
    STEP = PATCH_SIZE * 1.1
    
    width_grid = (COLS - 1) * STEP
    height_grid = (ROWS - 1) * STEP
    
    dst_tl = [MARGIN, MARGIN]
    dst_tr = [MARGIN + width_grid, MARGIN]
    dst_br = [MARGIN + width_grid, MARGIN + height_grid]
    dst_bl = [MARGIN, MARGIN + height_grid]
    
    dst_pts = np.array([dst_tl, dst_tr, dst_br, dst_bl], dtype="float32")
    
    out_w = int(MARGIN * 2 + width_grid)
    out_h = int(MARGIN * 2 + height_grid)
    
    print(f"Dimensions sortie : {out_w}x{out_h}")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Application RAW
    planes = [
        raw_data[0::2, 0::2],
        raw_data[0::2, 1::2],
        raw_data[1::2, 0::2],
        raw_data[1::2, 1::2]
    ]
    
    warped_planes = []
    S = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    M_scaled = S @ M @ np.linalg.inv(S)
    
    out_w_half = out_w // 2
    out_h_half = out_h // 2
    
    for p in planes:
        warped = cv2.warpPerspective(p, M_scaled, (out_w_half, out_h_half), flags=cv2.INTER_LINEAR)
        warped_planes.append(warped)
        
    stack = np.stack(warped_planes, axis=0)
    
    tifffile.imwrite(output_path, stack, photometric='minisblack')
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
        print("Usage: python rectify_raw_hybrid.py <fichier_raw.ARW>")
    else:
        input_file = sys.argv[1]
        output_file = input_file + ".rectified.tiff"
        rectify_raw_hybrid(input_file, output_file)
