from PIL import Image, ImageDraw

# --- CONFIGURATION ---
WIDTH = 3840  # 4K
HEIGHT = 2160 
ASPECT_RATIO = "16:9"

# Grille
COLS = 11
ROWS = 7
TOTAL_PATCHES = COLS * ROWS

# Marges et espacements (Guard Bands)
MARGIN_X = 200  
MARGIN_Y = 200  
PATCH_SPACING = 20 # Espace noir entre les cases

# --- COULEURS CALIBRÉES (Selon vos mesures) ---
# Format (Rouge, Vert, Bleu)
# Case la plus sombre (Haut-Gauche)
START_COLOR = (26, 11, 18)   
# Case la plus claire (Bas-Droite)
END_COLOR = (254, 151, 223)  

# --- FONCTIONS ---

def interpolate_color(start_rgb, end_rgb, index, total_steps):
    """Calcule la couleur RGB intermédiaire pour l'étape donnée"""
    if total_steps <= 1:
        return start_rgb
    
    r_start, g_start, b_start = start_rgb
    r_end, g_end, b_end = end_rgb
    
    ratio = index / (total_steps - 1)
    
    r = int(r_start + (r_end - r_start) * ratio)
    g = int(g_start + (g_end - g_start) * ratio)
    b = int(b_start + (b_end - b_start) * ratio)
    
    return (r, g, b)

def draw_checkerboard(draw, rect, size=4):
    """Dessine le damier de contrôle de flou"""
    x1, y1, x2, y2 = rect
    for y in range(y1, y2, size):
        for x in range(x1, x2, size):
            # Damier noir et blanc
            if ((x // size) + (y // size)) % 2 == 0:
                draw.rectangle([x, y, x+size, y+size], fill=(255, 255, 255))
            else:
                draw.rectangle([x, y, x+size, y+size], fill=(0, 0, 0))

# --- GÉNÉRATION ---

# 1. Fond (Gris moyen pour éviter l'éblouissement général, ou Blanc)
bg_color = (255, 255, 255) 
img = Image.new('RGB', (WIDTH, HEIGHT), bg_color)
draw = ImageDraw.Draw(img)

# 2. Marqueurs de détection (Résistants au flou - Disques noirs)
marker_radius = 40
corners = [
    (100, 100),                # HG
    (WIDTH-100, 100),          # HD
    (100, HEIGHT-100),         # BG
    (WIDTH-100, HEIGHT-100)    # BD
]

for x, y in corners:
    draw.ellipse((x - marker_radius, y - marker_radius, 
                  x + marker_radius, y + marker_radius), fill=(0, 0, 0))

# 3. Marqueur d'orientation (Le "Haut")
# Petit point au centre-haut pour indiquer le sens à l'algo
top_center_x = WIDTH // 2
top_center_y = 100
draw.ellipse((top_center_x - 20, top_center_y - 20, 
              top_center_x + 20, top_center_y + 20), fill=(0, 0, 0))

# Ajout : Point au milieu à droite
right_center_x = WIDTH - 100
right_center_y = HEIGHT // 2
draw.ellipse((right_center_x - 20, right_center_y - 20, 
              right_center_x + 20, right_center_y + 20), fill=(0, 0, 0))

# 4. Dessin de la Grille
grid_width = WIDTH - (2 * MARGIN_X)
grid_height = HEIGHT - (2 * MARGIN_Y) - 150 # Place pour le damier

# Taille d'un patch
patch_w = (grid_width - ((COLS - 1) * PATCH_SPACING)) / COLS
patch_h = (grid_height - ((ROWS - 1) * PATCH_SPACING)) / ROWS

# Fond noir global sous la grille (pour créer les séparations nettes)
grid_rect = [MARGIN_X, MARGIN_Y, MARGIN_X + grid_width, MARGIN_Y + grid_height]
draw.rectangle(grid_rect, fill=(0, 0, 0))

current_patch = 0

for row in range(ROWS):
    for col in range(COLS):
        # Coordonnées du patch
        x = MARGIN_X + (col * (patch_w + PATCH_SPACING))
        y = MARGIN_Y + (row * (patch_h + PATCH_SPACING))
        
        # Calcul de la couleur interpolée
        color = interpolate_color(START_COLOR, END_COLOR, current_patch, TOTAL_PATCHES)
        
        # Dessin
        draw.rectangle([x, y, x + patch_w, y + patch_h], fill=color)
        
        current_patch += 1

# 5. Mire de Focus (Damier) en bas
check_h = 80
check_w = grid_width // 2 
check_x = (WIDTH - check_w) // 2
check_y = HEIGHT - 120 

# On dessine un cadre noir autour du damier pour le contraste
draw.rectangle([check_x - 2, check_y - 2, check_x + check_w + 2, check_y + check_h + 2], outline=(0,0,0), width=2)
draw_checkerboard(draw, [check_x, check_y, check_x + check_w, check_y + check_h], size=4)

# --- FIN ---
output_filename = "DR_Grid_3.0_Calibrated.png"
img.save(output_filename)
print(f"Grille générée avec succès : {output_filename}")
print(f"Dégradé de {START_COLOR} à {END_COLOR}")
img.show()