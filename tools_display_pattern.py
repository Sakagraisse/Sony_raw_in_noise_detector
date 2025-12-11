import pygame
import sys

# --- CONFIGURATION ---
# Couleurs calibrées (identiques au script de génération)
START_COLOR = (26, 11, 18)
END_COLOR = (254, 151, 223)
BG_COLOR = (255, 255, 255) # Fond blanc pour l'écran
GRID_BG_COLOR = (0, 0, 0)  # Fond noir sous les patchs
FLAT_COLOR = (212, 96, 146) # Couleur pour le flat field (P2P)

# Grille
COLS = 11
ROWS = 7
TOTAL_PATCHES = COLS * ROWS

# Ratio cible de la zone utile (hors bandes noires de l'écran)
TARGET_ASPECT_RATIO = 16 / 9

def interpolate_color(start_rgb, end_rgb, index, total_steps):
    """Calcule la couleur RGB intermédiaire"""
    if total_steps <= 1:
        return start_rgb
    
    r_start, g_start, b_start = start_rgb
    r_end, g_end, b_end = end_rgb
    
    ratio = index / (total_steps - 1)
    
    r = int(r_start + (r_end - r_start) * ratio)
    g = int(g_start + (g_end - g_start) * ratio)
    b = int(b_start + (b_end - b_start) * ratio)
    
    return (r, g, b)

def create_checkerboard_surface(width, height, square_size):
    """
    Génère une surface Pygame contenant le damier.
    Optimisé pour ne pas recalculer à chaque frame.
    """
    surface = pygame.Surface((width, height))
    surface.fill((0, 0, 0)) # Fond noir
    
    # On dessine les carrés blancs
    # Pour éviter les artefacts, on dessine pixel par pixel ou bloc par bloc aligné
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                rect = pygame.Rect(x, y, square_size, square_size)
                pygame.draw.rect(surface, (255, 255, 255), rect)
                
    return surface

def main():
    pygame.init()
    
    # --- GESTION RÉSOLUTION & RETINA (MAC OS) ---
    # Sur macOS Retina, (0,0) utilise souvent la résolution "logique" (mise à l'échelle par l'OS).
    # Pour avoir du pixel-perfect (netteté maximale), on doit demander explicitement
    # la plus haute résolution disponible dans la liste des modes.
    modes = pygame.display.list_modes()
    
    if modes:
        # La liste est triée du plus grand au plus petit. Le premier est le max natif.
        target_res = modes[0]
        print(f"Modes détectés. Sélection de la résolution native maximale : {target_res}")
        screen = pygame.display.set_mode(target_res, pygame.FULLSCREEN)
    else:
        # Fallback si la détection échoue
        print("Aucun mode détecté, utilisation de la résolution bureau par défaut.")
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    
    # On récupère les dimensions réelles du buffer d'affichage
    screen_w, screen_h = screen.get_size()
    
    pygame.display.set_caption("Sony Calibration Grid - Dynamic")
    print(f"Résolution active : {screen_w}x{screen_h}")
    
    # --- CALCUL DES DIMENSIONS (Letterboxing) ---
    # On veut garder le ratio 16:9 pour la grille, quitte à avoir des bandes noires
    screen_ratio = screen_w / screen_h
    
    if screen_ratio > TARGET_ASPECT_RATIO:
        # L'écran est plus large que 16:9 (ex: 21:9) -> Bandes sur les côtés
        render_h = screen_h
        render_w = int(screen_h * TARGET_ASPECT_RATIO)
        offset_x = (screen_w - render_w) // 2
        offset_y = 0
    else:
        # L'écran est plus haut que 16:9 (ex: 16:10 ou 4:3) -> Bandes en haut/bas
        render_w = screen_w
        render_h = int(screen_w / TARGET_ASPECT_RATIO)
        offset_x = 0
        offset_y = (screen_h - render_h) // 2

    # Marges relatives à la zone de rendu (basées sur la largeur rendue)
    # Dans le script original 4K (3840px), marge = 200px -> ratio ~ 0.052
    margin_x = int(render_w * 0.052)
    margin_y = int(render_h * 0.092) # ratio ~ 200/2160
    
    # Espace entre patchs
    patch_spacing = int(render_w * 0.0052) # ratio ~ 20/3840
    
    # Dimensions de la grille de patchs
    grid_area_w = render_w - (2 * margin_x)
    grid_area_h = render_h - (2 * margin_y) - int(render_h * 0.07) # - espace pour le damier
    
    patch_w = (grid_area_w - ((COLS - 1) * patch_spacing)) / COLS
    patch_h = (grid_area_h - ((ROWS - 1) * patch_spacing)) / ROWS
    
    # --- PRÉPARATION DU DAMIER (Pixel Perfect) ---
    # Le damier doit être net. On calcule sa taille en pixels écran.
    check_w = int(grid_area_w * 0.5)
    check_h = int(render_h * 0.037) # ~80px sur 2160
    
    # Taille des cases du damier :
    # Pour une netteté optimale, on veut un nombre entier de pixels.
    # 4 pixels écran est un bon compromis : assez fin pour le focus, assez gros pour éviter le moiré violent.
    # Si l'écran est très haute def (Retina/4K), on peut monter à 6 ou 8.
    checker_square_size = max(2, int(screen_w / 1000) * 2) 
    
    checker_surface = create_checkerboard_surface(check_w, check_h, checker_square_size)
    checker_pos_x = offset_x + (render_w - check_w) // 2
    checker_pos_y = offset_y + render_h - int(render_h * 0.055) - check_h

    # --- BOUCLE PRINCIPALE ---
    running = True
    show_flat = True # Commence par le flat field

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    show_flat = not show_flat

        if show_flat:
            # Mode Flat Field
            screen.fill(FLAT_COLOR)
        else:
            # Mode Grille
            # 1. Fond
            screen.fill(BG_COLOR)
            
            # 2. Fond noir de la grille (Guard bands)
            grid_rect_x = offset_x + margin_x
            grid_rect_y = offset_y + margin_y
            pygame.draw.rect(screen, GRID_BG_COLOR, 
                             (grid_rect_x, grid_rect_y, grid_area_w, grid_area_h))
            
            # 3. Dessin des patchs
            current_patch = 0
            for row in range(ROWS):
                for col in range(COLS):
                    px = grid_rect_x + (col * (patch_w + patch_spacing))
                    py = grid_rect_y + (row * (patch_h + patch_spacing))
                    
                    color = interpolate_color(START_COLOR, END_COLOR, current_patch, TOTAL_PATCHES)
                    
                    # On utilise des float pour le calcul mais int pour l'affichage
                    # +1 sur la taille pour éviter les micro-lignes noires dues à l'arrondi si nécessaire
                    pygame.draw.rect(screen, color, (int(px), int(py), int(patch_w)+1, int(patch_h)+1))
                    
                    current_patch += 1
                    
            # 4. Marqueurs (Coins + Orientation)
            marker_radius = int(render_w * 0.0104) # ~40px sur 3840
            small_marker_radius = int(marker_radius * 0.5)
            
            # Positions relatives au render_rect
            corners = [
                (offset_x + int(render_w * 0.026), offset_y + int(render_h * 0.046)), # HG
                (offset_x + render_w - int(render_w * 0.026), offset_y + int(render_h * 0.046)), # HD
                (offset_x + int(render_w * 0.026), offset_y + render_h - int(render_h * 0.046)), # BG
                (offset_x + render_w - int(render_w * 0.026), offset_y + render_h - int(render_h * 0.046)) # BD
            ]
            
            for cx, cy in corners:
                pygame.draw.circle(screen, (0, 0, 0), (cx, cy), marker_radius)
                
            # Marqueur Haut (Top Center)
            top_cx = offset_x + render_w // 2
            top_cy = offset_y + int(render_h * 0.046)
            pygame.draw.circle(screen, (0, 0, 0), (top_cx, top_cy), small_marker_radius)

            # Marqueur Droite (Right Center)
            right_cx = offset_x + render_w - int(render_w * 0.026)
            right_cy = offset_y + render_h // 2
            pygame.draw.circle(screen, (0, 0, 0), (right_cx, right_cy), small_marker_radius)
            
            # 5. Damier de Focus
            # On blit la surface pré-calculée
            screen.blit(checker_surface, (checker_pos_x, checker_pos_y))
            # Cadre autour du damier
            pygame.draw.rect(screen, (0,0,0), (checker_pos_x-2, checker_pos_y-2, check_w+4, check_h+4), 2)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
